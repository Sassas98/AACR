import os
import json
import math
import random
from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import String

from smart_city_interfaces.action import NavigateToPose


class BusState(str, Enum):
    IDLE = "IDLE"
    ON_ROUTE = "ON_ROUTE"
    RETURNING_TO_PARKING = "RETURNING_TO_PARKING"
    PARKED = "PARKED"


class BusPathManager(Node):

    def __init__(self):
        super().__init__("bus_path_manager")

        self.declare_parameter("bus_id", "bus_1")
        self.declare_parameter("initial_path_id", "path_A")
        self.declare_parameter("laps", 3)
        self.declare_parameter("idle_start_delay_sec", 5.0)
        self.declare_parameter("random_start_jitter_sec", 6.0)
        self.declare_parameter("normal_speed", 1.0)
        self.declare_parameter("paths_config_file", "config/bus_paths.json")
        self.declare_parameter("parkings_config_file", "config/parkings.json")

        self.bus_id = self.get_parameter("bus_id").value
        self.current_path_id = self.get_parameter("initial_path_id").value
        self.remaining_laps = int(self.get_parameter("laps").value)
        self.idle_start_delay_sec = float(self.get_parameter("idle_start_delay_sec").value)
        self.random_start_jitter_sec = float(self.get_parameter("random_start_jitter_sec").value)
        self.normal_speed = float(self.get_parameter("normal_speed").value)

        self.state = BusState.IDLE
        self.current_waypoint_index = 0
        self.active_goal_handle = None
        self.navigation_busy = False

        self.known_buses = {}
        self.known_parkings = {}

        self.paths = self.load_json_config(
            self.get_parameter("paths_config_file").value,
            "paths"
        )

        self.parkings = self.load_json_config(
            self.get_parameter("parkings_config_file").value,
            "parkings"
        )

        self.status_pub = self.create_publisher(String, "/bus/status", 10)
        self.parking_claim_pub = self.create_publisher(String, "/parking/claim", 10)

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

        self.status_timer = self.create_timer(1.0, self.publish_status)
        self.decision_timer = self.create_timer(1.0, self.decision_loop)

        start_delay = self.idle_start_delay_sec + random.uniform(
            0.0,
            self.random_start_jitter_sec
        )

        self.start_timer = self.create_timer(
            start_delay,
            self.start_after_idle_delay
        )

        self.get_logger().info(
            f"{self.bus_id}: bus_path_manager avviato | "
            f"path iniziale={self.current_path_id}, "
            f"laps={self.remaining_laps}, "
            f"start_delay={start_delay:.1f}s"
        )

    # ------------------------------------------------------------------
    # CONFIG
    # ------------------------------------------------------------------

    def load_json_config(self, file_path, root_key):
        if not os.path.isabs(file_path):
            file_path = os.path.join(os.getcwd(), file_path)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File configurazione non trovato: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if root_key not in data:
            raise ValueError(f"Nel file {file_path} manca la chiave '{root_key}'")

        return data[root_key]

    # ------------------------------------------------------------------
    # CALLBACK TOPIC
    # ------------------------------------------------------------------

    def on_bus_status(self, msg):
        try:
            status = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        other_bus_id = status.get("bus_id")

        if not other_bus_id or other_bus_id == self.bus_id:
            return

        self.known_buses[other_bus_id] = status

    def on_parking_status(self, msg):
        try:
            status = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        parking_id = status.get("parking_id")

        if parking_id:
            self.known_parkings[parking_id] = status

    # ------------------------------------------------------------------
    # DECISION LOOP
    # ------------------------------------------------------------------

    def start_after_idle_delay(self):
        self.start_timer.cancel()

        if self.state != BusState.IDLE:
            return

        best_path = self.find_best_path()

        if best_path is not None:
            self.current_path_id = best_path

        self.current_waypoint_index = 0
        self.state = BusState.ON_ROUTE

        self.get_logger().info(
            f"{self.bus_id}: partenza su {self.current_path_id}"
        )

    def decision_loop(self):
        if self.navigation_busy:
            return

        if self.state == BusState.PARKED:
            return

        if self.remaining_laps <= 0 and self.state != BusState.RETURNING_TO_PARKING:
            self.return_to_parking()
            return

        if self.state == BusState.IDLE:
            return

        if self.state == BusState.ON_ROUTE:
            self.go_to_next_waypoint()
            return

        if self.state == BusState.RETURNING_TO_PARKING:
            return

    def find_best_path(self):
        if not self.paths:
            return None

        scores = {path_id: 0 for path_id in self.paths.keys()}

        for _, bus in self.known_buses.items():
            state = bus.get("state")
            path_id = bus.get("current_path_id")

            if state in [BusState.ON_ROUTE.value, BusState.IDLE.value]:
                if path_id in scores:
                    scores[path_id] += 1

        return min(scores, key=scores.get)

    # ------------------------------------------------------------------
    # ROUTE
    # ------------------------------------------------------------------

    def go_to_next_waypoint(self):
        waypoints = self.paths.get(self.current_path_id)

        if not waypoints:
            self.get_logger().warn(f"{self.bus_id}: path inesistente {self.current_path_id}")
            return

        waypoint = waypoints[self.current_waypoint_index]

        self.send_navigation_goal(
            target_x=float(waypoint["x"]),
            target_y=float(waypoint["y"]),
            target_type="WAYPOINT",
            max_speed=self.normal_speed,
            mission_id=f"path:{self.current_path_id}:wp:{waypoint['id']}"
        )

    def complete_current_waypoint(self):
        waypoints = self.paths[self.current_path_id]

        self.current_waypoint_index += 1

        if self.current_waypoint_index < len(waypoints):
            return

        self.current_waypoint_index = 0
        self.remaining_laps -= 1

        self.get_logger().info(
            f"{self.bus_id}: giro completato su {self.current_path_id}, "
            f"giri rimanenti={self.remaining_laps}"
        )

        if self.remaining_laps > 0:
            best_path = self.find_best_path()

            if best_path and best_path != self.current_path_id:
                self.get_logger().info(
                    f"{self.bus_id}: prossimo giro cambio path "
                    f"{self.current_path_id} -> {best_path}"
                )
                self.current_path_id = best_path

    # ------------------------------------------------------------------
    # PARKING
    # ------------------------------------------------------------------

    def return_to_parking(self):
        parking = self.find_free_parking()

        if parking is None:
            self.get_logger().warn(
                f"{self.bus_id}: nessun parcheggio libero, continuo il giro"
            )
            self.remaining_laps = 1
            self.state = BusState.ON_ROUTE
            return

        parking_id, parking_data = parking

        self.claim_parking(parking_id)
        self.state = BusState.RETURNING_TO_PARKING

        self.send_navigation_goal(
            target_x=float(parking_data["x"]),
            target_y=float(parking_data["y"]),
            target_type="PARKING",
            max_speed=self.normal_speed,
            mission_id=f"parking:{parking_id}"
        )

    def find_free_parking(self):
        for parking_id, parking_data in self.parkings.items():
            if parking_data.get("vehicle_type") != "BUS":
                continue

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

        self.get_logger().info(f"{self.bus_id}: claim parcheggio {parking_id}")

    # ------------------------------------------------------------------
    # ACTION CLIENT
    # ------------------------------------------------------------------

    def send_navigation_goal(self, target_x, target_y, target_type, max_speed, mission_id):
        if self.navigation_busy:
            return

        if not self.navigation_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn(f"{self.bus_id}: navigation_executor non disponibile")
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
            f"type={target_type}, target=({target_x:.2f},{target_y:.2f}), "
            f"speed={max_speed:.2f}"
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

            if self.state == BusState.RETURNING_TO_PARKING:
                self.state = BusState.ON_ROUTE

            return

        self.get_logger().info(
            f"{self.bus_id}: navigazione completata: {result.message}"
        )

        if self.state == BusState.ON_ROUTE:
            self.complete_current_waypoint()
            return

        if self.state == BusState.RETURNING_TO_PARKING:
            self.state = BusState.PARKED
            self.get_logger().info(f"{self.bus_id}: parcheggiato")
            return

    # ------------------------------------------------------------------
    # STATUS
    # ------------------------------------------------------------------

    def publish_status(self):
        payload = {
            "bus_id": self.bus_id,
            "state": self.state.value,
            "current_path_id": self.current_path_id,
            "current_waypoint_index": self.current_waypoint_index,
            "remaining_laps": self.remaining_laps,
            "navigation_busy": self.navigation_busy
        }

        msg = String()
        msg.data = json.dumps(payload)

        self.status_pub.publish(msg)


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