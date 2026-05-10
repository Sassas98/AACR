import os
import json
import math
import uuid
from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from std_msgs.msg import String

from smart_city_interfaces.action import NavigateToPose


class BusState(str, Enum):
    IDLE = "IDLE"
    ON_ROUTE = "ON_ROUTE"
    TO_BOOKED_STOP = "TO_BOOKED_STOP"
    RETURNING_TO_PARKING = "RETURNING_TO_PARKING"
    PARKED = "PARKED"


class BusPathManager(Node):

    def __init__(self):
        super().__init__("bus_path_manager")

        self.declare_parameter("bus_id", "bus_1")
        self.declare_parameter("initial_path_id", "path_A")
        self.declare_parameter("laps", 3)
        self.declare_parameter("idle_start_delay_sec", 5.0)
        self.declare_parameter("normal_speed", 1.0)
        self.declare_parameter("priority_speed", 1.8)

        self.bus_id = self.get_parameter("bus_id").value
        self.current_path_id = self.get_parameter("initial_path_id").value
        self.remaining_laps = int(self.get_parameter("laps").value)
        self.idle_start_delay_sec = float(self.get_parameter("idle_start_delay_sec").value)
        self.normal_speed = float(self.get_parameter("normal_speed").value)
        self.priority_speed = float(self.get_parameter("priority_speed").value)

        self.state = BusState.IDLE

        self.current_waypoint_index = 0
        self.active_booking = None
        self.active_goal_handle = None
        self.navigation_busy = False

        self.known_buses = {}
        self.known_parkings = {}

        self.handled_bookings = set()

        self.paths = self.load_paths()
        self.parkings = self.load_parkings()

        self.status_pub = self.create_publisher(
            String,
            "/bus/status",
            10
        )

        self.parking_claim_pub = self.create_publisher(
            String,
            "/parking/claim",
            10
        )

        self.booking_sub = self.create_subscription(
            String,
            "/gazebo/bus_booking",
            self.on_bus_booking,
            10
        )

        self.bus_status_sub = self.create_subscription(
            String,
            "/bus/status",
            self.on_bus_status,
            10
        )

        self.parking_status_sub = self.create_subscription(
            String,
            "/parking/status",
            self.on_parking_status,
            10
        )

        self.navigation_client = ActionClient(
            self,
            NavigateToPose,
            "/navigation_executor/navigate_to_pose"
        )

        self.status_timer = self.create_timer(
            1.0,
            self.publish_status
        )

        self.decision_timer = self.create_timer(
            1.0,
            self.decision_loop
        )

        self.start_timer = self.create_timer(
            self.idle_start_delay_sec,
            self.start_after_idle_delay
        )

        self.get_logger().info(
            f"bus_path_manager avviato: bus_id={self.bus_id}, "
            f"path={self.current_path_id}, laps={self.remaining_laps}"
        )

    # ---------------------------------------------------------------------
    # CONFIGURAZIONE LOCALE
    # ---------------------------------------------------------------------

    import os
    import json
    
    def load_paths(self):
        self.declare_parameter(
            "paths_config_file",
            "config/bus_paths.json"
        )

        file_path = self.get_parameter("paths_config_file").value

        return self.load_json_config(file_path, "paths")


    def load_parkings(self):
        self.declare_parameter(
            "parkings_config_file",
            "config/parkings.json"
        )

        file_path = self.get_parameter("parkings_config_file").value

        return self.load_json_config(file_path, "parkings")


    def load_json_config(self, file_path, root_key):
        if not os.path.isabs(file_path):
            file_path = os.path.join(os.getcwd(), file_path)

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"File di configurazione non trovato: {file_path}"
            )

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if root_key not in data:
            raise ValueError(
                f"Nel file {file_path} manca la chiave root '{root_key}'"
            )

        return data[root_key]

    # ---------------------------------------------------------------------
    # CALLBACK TOPIC
    # ---------------------------------------------------------------------

    def on_bus_booking(self, msg):
        try:
            booking = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn(f"Booking non JSON valido: {msg.data}")
            return

        required_fields = ["booking_id", "stop_id", "path_id", "x", "y"]
        if not all(field in booking for field in required_fields):
            self.get_logger().warn(f"Booking incompleto: {booking}")
            return

        booking_id = booking["booking_id"]

        if booking_id in self.handled_bookings:
            return

        path_id = booking["path_id"]

        if not self.should_take_booking(path_id):
            return

        self.handled_bookings.add(booking_id)
        self.active_booking = booking

        if path_id != self.current_path_id:
            self.get_logger().info(
                f"{self.bus_id}: cambio path da {self.current_path_id} a {path_id} "
                f"per booking {booking_id}"
            )
            self.current_path_id = path_id
            self.current_waypoint_index = self.find_nearest_waypoint_index(path_id, booking["x"], booking["y"])

        self.state = BusState.TO_BOOKED_STOP

        self.get_logger().info(
            f"{self.bus_id}: prendo booking {booking_id}, stop={booking['stop_id']}, "
            f"path={path_id}"
        )

        self.send_navigation_goal(
            target_x=float(booking["x"]),
            target_y=float(booking["y"]),
            target_type="BUS_STOP",
            max_speed=self.priority_speed,
            mission_id=f"booking:{booking_id}"
        )

    def on_bus_status(self, msg):
        try:
            status = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        other_bus_id = status.get("bus_id")

        if other_bus_id is None or other_bus_id == self.bus_id:
            return

        self.known_buses[other_bus_id] = status

    def on_parking_status(self, msg):
        try:
            status = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        parking_id = status.get("parking_id")

        if parking_id is None:
            return

        self.known_parkings[parking_id] = status

    # ---------------------------------------------------------------------
    # LOGICA DECISIONALE
    # ---------------------------------------------------------------------

    def start_after_idle_delay(self):
        self.start_timer.cancel()

        if self.state == BusState.IDLE:
            self.state = BusState.ON_ROUTE
            self.get_logger().info(
                f"{self.bus_id}: uscita da IDLE dopo timer, inizio path {self.current_path_id}"
            )

    def decision_loop(self):
        if self.navigation_busy:
            return

        if self.state == BusState.PARKED:
            return

        if self.remaining_laps <= 0 and self.state != BusState.RETURNING_TO_PARKING:
            self.return_to_parking()
            return

        uncovered_path = self.find_uncovered_path()

        if self.state == BusState.IDLE and uncovered_path is not None:
            self.current_path_id = uncovered_path
            self.current_waypoint_index = 0
            self.state = BusState.ON_ROUTE
            self.get_logger().info(
                f"{self.bus_id}: attivazione su path scoperto {uncovered_path}"
            )

        if self.state == BusState.ON_ROUTE:
            self.go_to_next_waypoint()
            return

        if self.state == BusState.TO_BOOKED_STOP:
            return

        if self.state == BusState.RETURNING_TO_PARKING:
            return

    def should_take_booking(self, path_id):
        if self.state in [BusState.RETURNING_TO_PARKING, BusState.PARKED]:
            return False

        if path_id == self.current_path_id:
            return True

        return self.is_path_uncovered(path_id)

    def is_path_uncovered(self, path_id):
        if self.current_path_id == path_id and self.state != BusState.PARKED:
            return False

        for _, bus in self.known_buses.items():
            bus_state = bus.get("state")
            bus_path = bus.get("current_path_id")

            if bus_state in ["ON_ROUTE", "TO_BOOKED_STOP", "IDLE"] and bus_path == path_id:
                return False

        return True

    def find_uncovered_path(self):
        for path_id in self.paths.keys():
            if self.is_path_uncovered(path_id):
                return path_id

        return None

    def go_to_next_waypoint(self):
        waypoints = self.paths.get(self.current_path_id)

        if not waypoints:
            self.get_logger().warn(f"Path inesistente: {self.current_path_id}")
            return

        waypoint = waypoints[self.current_waypoint_index]

        self.send_navigation_goal(
            target_x=waypoint["x"],
            target_y=waypoint["y"],
            target_type="WAYPOINT",
            max_speed=self.normal_speed,
            mission_id=f"path:{self.current_path_id}:wp:{waypoint['id']}"
        )

    def complete_current_waypoint(self):
        waypoints = self.paths[self.current_path_id]

        self.current_waypoint_index += 1

        if self.current_waypoint_index >= len(waypoints):
            self.current_waypoint_index = 0
            self.remaining_laps -= 1

            self.get_logger().info(
                f"{self.bus_id}: giro completato su {self.current_path_id}, "
                f"giri rimanenti={self.remaining_laps}"
            )

    def return_to_parking(self):
        parking = self.find_free_parking()

        if parking is None:
            self.get_logger().warn(
                f"{self.bus_id}: nessun parcheggio libero noto, resto in ON_ROUTE"
            )
            self.state = BusState.ON_ROUTE
            return

        parking_id, parking_data = parking

        self.claim_parking(parking_id)

        self.state = BusState.RETURNING_TO_PARKING

        self.send_navigation_goal(
            target_x=parking_data["x"],
            target_y=parking_data["y"],
            target_type="PARKING",
            max_speed=self.normal_speed,
            mission_id=f"parking:{parking_id}"
        )

    def find_free_parking(self):
        for parking_id, parking_data in self.parkings.items():
            known = self.known_parkings.get(parking_id)

            if known is None:
                return parking_id, parking_data

            if known.get("occupied") is False:
                return parking_id, parking_data

        return None

    def claim_parking(self, parking_id):
        payload = {
            "parking_id": parking_id,
            "vehicle_id": self.bus_id,
            "vehicle_type": "BUS",
            "claim": True
        }

        msg = String()
        msg.data = json.dumps(payload)
        self.parking_claim_pub.publish(msg)

        self.get_logger().info(
            f"{self.bus_id}: claim parcheggio {parking_id}"
        )

    def find_nearest_waypoint_index(self, path_id, x, y):
        waypoints = self.paths.get(path_id, [])

        if not waypoints:
            return 0

        best_index = 0
        best_distance = float("inf")

        for i, wp in enumerate(waypoints):
            dx = wp["x"] - x
            dy = wp["y"] - y
            d = math.sqrt(dx * dx + dy * dy)

            if d < best_distance:
                best_distance = d
                best_index = i

        return best_index

    # ---------------------------------------------------------------------
    # ACTION CLIENT
    # ---------------------------------------------------------------------

    def send_navigation_goal(self, target_x, target_y, target_type, max_speed, mission_id):
        if self.navigation_busy:
            return

        if not self.navigation_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn("navigation_executor non disponibile")
            return

        goal = NavigateToPose.Goal()
        goal.vehicle_id = self.bus_id
        goal.mission_id = mission_id
        goal.target_type = target_type
        goal.target_x = float(target_x)
        goal.target_y = float(target_y)
        goal.max_speed = float(max_speed)

        self.navigation_busy = True

        self.get_logger().info(
            f"{self.bus_id}: invio goal {mission_id}, "
            f"type={target_type}, target=({target_x}, {target_y}), speed={max_speed}"
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
            self.get_logger().warn(f"{self.bus_id}: goal rifiutato")
            return

        self.active_goal_handle = goal_handle

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.on_navigation_result)

    def on_navigation_feedback(self, feedback_msg):
        feedback = feedback_msg.feedback

        self.get_logger().debug(
            f"{self.bus_id}: feedback nav status={feedback.status}, "
            f"dist={feedback.distance_remaining:.2f}"
        )

    def on_navigation_result(self, future):
        self.navigation_busy = False

        result = future.result().result

        if not result.success:
            self.get_logger().warn(
                f"{self.bus_id}: navigazione fallita: {result.message}"
            )
            self.state = BusState.ON_ROUTE
            return

        self.get_logger().info(
            f"{self.bus_id}: navigazione completata: {result.message}"
        )

        if self.state == BusState.ON_ROUTE:
            self.complete_current_waypoint()
            return

        if self.state == BusState.TO_BOOKED_STOP:
            self.get_logger().info(
                f"{self.bus_id}: fermata prenotata raggiunta"
            )
            self.active_booking = None
            self.state = BusState.ON_ROUTE
            return

        if self.state == BusState.RETURNING_TO_PARKING:
            self.state = BusState.PARKED
            self.get_logger().info(
                f"{self.bus_id}: parcheggiato"
            )
            return

    # ---------------------------------------------------------------------
    # PUBBLICAZIONE STATO
    # ---------------------------------------------------------------------

    def publish_status(self):
        payload = {
            "bus_id": self.bus_id,
            "state": self.state.value,
            "current_path_id": self.current_path_id,
            "future_path_ids": self.estimate_future_paths(),
            "current_waypoint_index": self.current_waypoint_index,
            "remaining_laps": self.remaining_laps,
            "navigation_busy": self.navigation_busy,
            "active_booking_id": self.active_booking["booking_id"] if self.active_booking else None
        }

        msg = String()
        msg.data = json.dumps(payload)

        self.status_pub.publish(msg)

    def estimate_future_paths(self):
        uncovered = []

        for path_id in self.paths.keys():
            if self.is_path_uncovered(path_id):
                uncovered.append(path_id)

        return uncovered


def main(args=None):
    rclpy.init(args=args)

    node = BusPathManager()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()