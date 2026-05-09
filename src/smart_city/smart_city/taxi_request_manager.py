import json
import os
import time
from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from std_msgs.msg import String
from std_srvs.srv import Trigger

from smart_city_interfaces.action import NavigateToPose


class TaxiServiceState(str, Enum):
    IDLE = "IDLE"
    TO_PICKUP = "TO_PICKUP"
    TO_DROPOFF = "TO_DROPOFF"
    RETURNING_TO_PARKING = "RETURNING_TO_PARKING"
    PARKED = "PARKED"


class TaxiRequestManager(Node):

    def __init__(self):
        super().__init__("taxi_request_manager")

        self.declare_parameter("taxi_id", "taxi_1")
        self.declare_parameter("parkings_config_file", "config/parkings.json")
        self.declare_parameter("normal_speed", 1.0)
        self.declare_parameter("service_speed", 1.5)

        self.taxi_id = self.get_parameter("taxi_id").value
        self.normal_speed = float(self.get_parameter("normal_speed").value)
        self.service_speed = float(self.get_parameter("service_speed").value)

        self.state = TaxiServiceState.IDLE
        self.active_request = None
        self.navigation_busy = False
        self.claimed_parking_id = None

        self.parkings = self.load_parkings()
        self.known_parkings = {}

        self.assignment_sub = self.create_subscription(
            String,
            "/taxi_assignment",
            self.on_taxi_assignment,
            10
        )

        self.parking_status_sub = self.create_subscription(
            String,
            "/parking/status",
            self.on_parking_status,
            10
        )

        self.parking_status_pub = self.create_publisher(
            String,
            "/parking/status",
            10
        )

        self.status_service = self.create_service(
            Trigger,
            "/taxi_status",
            self.on_taxi_status_request
        )

        self.navigation_client = ActionClient(
            self,
            NavigateToPose,
            "/navigation_executor/navigate_to_pose"
        )

        self.get_logger().info(
            f"taxi_request_manager avviato per taxi_id={self.taxi_id}"
        )

    # ------------------------------------------------------------------
    # CONFIG
    # ------------------------------------------------------------------

    def load_parkings(self):
        file_path = self.get_parameter("parkings_config_file").value

        if not os.path.isabs(file_path):
            file_path = os.path.join(os.getcwd(), file_path)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File parcheggi non trovato: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "parkings" not in data:
            raise ValueError("Nel file manca la chiave 'parkings'")

        return data["parkings"]

    # ------------------------------------------------------------------
    # SERVICE /taxi_status
    # ------------------------------------------------------------------

    def on_taxi_status_request(self, request, response):
        available = self.state in [
            TaxiServiceState.IDLE,
            TaxiServiceState.PARKED
        ] and not self.navigation_busy

        response.success = available
        response.message = json.dumps({
            "taxi_id": self.taxi_id,
            "available": available,
            "state": self.state.value,
            "navigation_busy": self.navigation_busy
        })

        return response

    # ------------------------------------------------------------------
    # ASSIGNMENT DAL COORDINATOR
    # ------------------------------------------------------------------

    def on_taxi_assignment(self, msg):
        try:
            assignment = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn(f"Assignment non JSON: {msg.data}")
            return

        if assignment.get("taxi_id") != self.taxi_id:
            return

        if self.state not in [TaxiServiceState.IDLE, TaxiServiceState.PARKED]:
            self.get_logger().warn(
                f"{self.taxi_id}: assignment ricevuto ma taxi non libero"
            )
            return

        required = [
            "request_id",
            "pickup_x",
            "pickup_y",
            "dropoff_x",
            "dropoff_y"
        ]

        if not all(k in assignment for k in required):
            self.get_logger().warn(f"Assignment incompleto: {assignment}")
            return

        self.active_request = assignment
        self.claimed_parking_id = None

        self.get_logger().info(
            f"{self.taxi_id}: assignment ricevuto per request {assignment['request_id']}"
        )

        self.state = TaxiServiceState.TO_PICKUP

        self.send_navigation_goal(
            mission_id=f"taxi_pickup:{assignment['request_id']}",
            target_type="TAXI_PICKUP",
            x=float(assignment["pickup_x"]),
            y=float(assignment["pickup_y"]),
            max_speed=self.service_speed
        )

    # ------------------------------------------------------------------
    # ACTION CLIENT
    # ------------------------------------------------------------------

    def send_navigation_goal(self, mission_id, target_type, x, y, max_speed):
        if not self.navigation_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn("navigation_executor non disponibile")
            return

        goal = NavigateToPose.Goal()
        goal.vehicle_id = self.taxi_id
        goal.mission_id = mission_id
        goal.target_type = target_type
        goal.target_x = float(x)
        goal.target_y = float(y)
        goal.max_speed = float(max_speed)

        self.navigation_busy = True

        self.get_logger().info(
            f"{self.taxi_id}: invio navigation goal {mission_id}, "
            f"type={target_type}, target=({x}, {y})"
        )

        future = self.navigation_client.send_goal_async(
            goal,
            feedback_callback=self.on_navigation_feedback
        )

        future.add_done_callback(self.on_navigation_goal_response)

    def on_navigation_goal_response(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.navigation_busy = False
            self.get_logger().warn(f"{self.taxi_id}: goal rifiutato")
            self.reset_to_idle()
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.on_navigation_result)

    def on_navigation_feedback(self, feedback_msg):
        feedback = feedback_msg.feedback

        self.get_logger().debug(
            f"{self.taxi_id}: nav={feedback.status}, "
            f"dist={feedback.distance_remaining:.2f}"
        )

    def on_navigation_result(self, future):
        self.navigation_busy = False

        result = future.result().result

        if not result.success:
            self.get_logger().warn(
                f"{self.taxi_id}: navigazione fallita: {result.message}"
            )
            self.reset_to_idle()
            return

        if self.state == TaxiServiceState.TO_PICKUP:
            self.on_pickup_reached()
            return

        if self.state == TaxiServiceState.TO_DROPOFF:
            self.on_dropoff_reached()
            return

        if self.state == TaxiServiceState.RETURNING_TO_PARKING:
            self.on_parking_reached()
            return

    # ------------------------------------------------------------------
    # TRANSIZIONI SERVIZIO
    # ------------------------------------------------------------------

    def on_pickup_reached(self):
        request_id = self.active_request["request_id"]

        self.get_logger().info(
            f"{self.taxi_id}: pickup raggiunto per {request_id}"
        )

        self.state = TaxiServiceState.TO_DROPOFF

        self.send_navigation_goal(
            mission_id=f"taxi_dropoff:{request_id}",
            target_type="TAXI_DROPOFF",
            x=float(self.active_request["dropoff_x"]),
            y=float(self.active_request["dropoff_y"]),
            max_speed=self.service_speed
        )

    def on_dropoff_reached(self):
        request_id = self.active_request["request_id"]

        self.get_logger().info(
            f"{self.taxi_id}: dropoff raggiunto per {request_id}"
        )

        self.active_request = None
        self.return_to_parking()

    def return_to_parking(self):
        parking = self.find_free_parking()

        if parking is None:
            self.get_logger().warn(
                f"{self.taxi_id}: nessun parcheggio libero, resto IDLE"
            )
            self.state = TaxiServiceState.IDLE
            return

        parking_id, parking_data = parking
        self.claimed_parking_id = parking_id

        self.publish_parking_status(
            parking_id=parking_id,
            occupied=True,
            state="CLAIMED"
        )

        self.state = TaxiServiceState.RETURNING_TO_PARKING

        self.send_navigation_goal(
            mission_id=f"taxi_parking:{parking_id}",
            target_type="PARKING",
            x=float(parking_data["x"]),
            y=float(parking_data["y"]),
            max_speed=self.normal_speed
        )

    def on_parking_reached(self):
        if self.claimed_parking_id is not None:
            self.publish_parking_status(
                parking_id=self.claimed_parking_id,
                occupied=True,
                state="OCCUPIED"
            )

        self.state = TaxiServiceState.PARKED

        self.get_logger().info(
            f"{self.taxi_id}: parcheggiato in {self.claimed_parking_id}"
        )

    def reset_to_idle(self):
        self.active_request = None
        self.navigation_busy = False
        self.claimed_parking_id = None
        self.state = TaxiServiceState.IDLE

    # ------------------------------------------------------------------
    # PARKING STATUS
    # ------------------------------------------------------------------

    def on_parking_status(self, msg):
        try:
            status = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        parking_id = status.get("parking_id")

        if parking_id is None:
            return

        self.known_parkings[parking_id] = status

    def find_free_parking(self):
        for parking_id, parking_data in self.parkings.items():
            known = self.known_parkings.get(parking_id)

            if known is None:
                return parking_id, parking_data

            if known.get("occupied") is False:
                return parking_id, parking_data

        return None

    def publish_parking_status(self, parking_id, occupied, state):
        payload = {
            "parking_id": parking_id,
            "vehicle_id": self.taxi_id,
            "vehicle_type": "TAXI",
            "occupied": occupied,
            "state": state,
            "timestamp": time.time()
        }

        msg = String()
        msg.data = json.dumps(payload)

        self.parking_status_pub.publish(msg)

    # ------------------------------------------------------------------


def main(args=None):
    rclpy.init(args=args)

    node = TaxiRequestManager()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()