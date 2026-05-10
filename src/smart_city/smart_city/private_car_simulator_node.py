import json
import math
import os
import random
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from nav_msgs.msg import Odometry

from smart_city_interfaces.action import NavigateToPose


class PrivateCarState:
    def __init__(self, vehicle_id):
        self.vehicle_id = vehicle_id
        self.x = 0.0
        self.y = 0.0
        self.busy = False
        self.last_goal_time = 0.0
        self.goal_handle = None


class PrivateCarSimulatorNode(Node):

    def __init__(self):
        super().__init__("private_car_simulator_node")

        self.declare_parameter("vehicles_config_file", "config/vehicles.json")
        self.declare_parameter("map_config_file", "config/city_map.json")
        self.declare_parameter("min_goal_interval_sec", 6.0)
        self.declare_parameter("max_goal_interval_sec", 14.0)
        self.declare_parameter("private_car_max_speed", 2.0)

        self.vehicles_config_file = self.get_parameter("vehicles_config_file").value
        self.map_config_file = self.get_parameter("map_config_file").value

        self.min_goal_interval_sec = float(
            self.get_parameter("min_goal_interval_sec").value
        )
        self.max_goal_interval_sec = float(
            self.get_parameter("max_goal_interval_sec").value
        )
        self.private_car_max_speed = float(
            self.get_parameter("private_car_max_speed").value
        )

        self.nodes = {}
        self.edges = []
        self.private_cars = {}
        self.action_clients = {}

        self.load_map()
        self.load_private_cars()

        self.timer = self.create_timer(1.0, self.loop)

        self.get_logger().info(
            f"private_car_simulator_node avviato: {len(self.private_cars)} auto private"
        )

    # ------------------------------------------------------------------
    # CONFIG
    # ------------------------------------------------------------------

    def load_json(self, file_path):
        if not os.path.isabs(file_path):
            file_path = os.path.join(os.getcwd(), file_path)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File non trovato: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_map(self):
        data = self.load_json(self.map_config_file)

        for node in data["nodes"]:
            self.nodes[node["id"]] = {
                "id": node["id"],
                "x": float(node["x"]),
                "y": float(node["y"])
            }

        self.edges = data["edges"]

    def load_private_cars(self):
        data = self.load_json(self.vehicles_config_file)

        for vehicle in data["vehicles"]:
            if vehicle.get("type") != "PRIVATE_CAR":
                continue

            vehicle_id = vehicle["id"]

            car = PrivateCarState(vehicle_id)

            spawn = vehicle.get("spawn", {})
            car.x = float(spawn.get("x", 0.0))
            car.y = float(spawn.get("y", 0.0))

            self.private_cars[vehicle_id] = car

            self.create_subscription(
                Odometry,
                f"/{vehicle_id}/odom",
                lambda msg, vid=vehicle_id: self.on_odom(msg, vid),
                10
            )

            self.action_clients[vehicle_id] = ActionClient(
                self,
                NavigateToPose,
                f"/{vehicle_id}/navigation_executor/navigate_to_pose"
            )

            self.get_logger().info(
                f"{vehicle_id}: registrata come auto privata"
            )

    # ------------------------------------------------------------------
    # CALLBACK
    # ------------------------------------------------------------------

    def on_odom(self, msg, vehicle_id):
        car = self.private_cars.get(vehicle_id)

        if car is None:
            return

        car.x = msg.pose.pose.position.x
        car.y = msg.pose.pose.position.y

    # ------------------------------------------------------------------
    # LOOP
    # ------------------------------------------------------------------

    def loop(self):
        now = time.time()

        for car in self.private_cars.values():
            if car.busy:
                continue

            elapsed = now - car.last_goal_time

            if elapsed < self.min_goal_interval_sec:
                continue

            if elapsed < random.uniform(
                self.min_goal_interval_sec,
                self.max_goal_interval_sec
            ):
                continue

            target = self.choose_random_road_point(car)

            self.send_goal(car, target)

    # ------------------------------------------------------------------
    # GOAL ACTION
    # ------------------------------------------------------------------

    def send_goal(self, car, target):
        client = self.action_clients[car.vehicle_id]

        if not client.wait_for_server(timeout_sec=0.2):
            self.get_logger().warn(
                f"{car.vehicle_id}: navigation_executor non disponibile"
            )
            return

        goal = NavigateToPose.Goal()
        goal.vehicle_id = car.vehicle_id
        goal.mission_id = f"private_car_{car.vehicle_id}_{int(time.time())}"
        goal.target_type = "PRIVATE_RANDOM_ROAD_TARGET"
        goal.target_x = float(target["x"])
        goal.target_y = float(target["y"])
        goal.max_speed = float(self.private_car_max_speed)

        car.busy = True
        car.last_goal_time = time.time()

        self.get_logger().info(
            f"{car.vehicle_id}: invio goal verso "
            f"({goal.target_x:.2f}, {goal.target_y:.2f})"
        )

        future = client.send_goal_async(goal)
        future.add_done_callback(
            lambda fut, vid=car.vehicle_id: self.on_goal_response(fut, vid)
        )

    def on_goal_response(self, future, vehicle_id):
        car = self.private_cars[vehicle_id]

        try:
            goal_handle = future.result()
        except Exception as ex:
            car.busy = False
            self.get_logger().error(
                f"{vehicle_id}: errore invio goal: {ex}"
            )
            return

        if not goal_handle.accepted:
            car.busy = False
            self.get_logger().warn(
                f"{vehicle_id}: goal rifiutato"
            )
            return

        car.goal_handle = goal_handle

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda fut, vid=vehicle_id: self.on_goal_result(fut, vid)
        )

    def on_goal_result(self, future, vehicle_id):
        car = self.private_cars[vehicle_id]
        car.busy = False
        car.last_goal_time = time.time()

        try:
            result = future.result().result
        except Exception as ex:
            self.get_logger().error(
                f"{vehicle_id}: errore risultato goal: {ex}"
            )
            return

        self.get_logger().info(
            f"{vehicle_id}: goal concluso success={result.success}, "
            f"message='{result.message}'"
        )

    # ------------------------------------------------------------------
    # TARGET SELECTION
    # ------------------------------------------------------------------

    def choose_random_road_point(self, car):
        edge = random.choice(self.edges)

        from_node = self.nodes[edge["from"]]
        to_node = self.nodes[edge["to"]]

        t = random.uniform(0.20, 0.80)

        center_x = from_node["x"] + (to_node["x"] - from_node["x"]) * t
        center_y = from_node["y"] + (to_node["y"] - from_node["y"]) * t

        lane_x, lane_y = self.apply_right_lane_offset(
            center_x,
            center_y,
            from_node["x"],
            from_node["y"],
            to_node["x"],
            to_node["y"]
        )

        return {
            "x": lane_x,
            "y": lane_y,
            "edge_id": edge["id"]
        }

    def apply_right_lane_offset(self, x, y, from_x, from_y, to_x, to_y):
        dx = to_x - from_x
        dy = to_y - from_y

        length = math.sqrt(dx * dx + dy * dy)

        if length <= 0.000001:
            return x, y

        ux = dx / length
        uy = dy / length

        right_x = uy
        right_y = -ux

        lane_width = 1.2
        offset = lane_width * 0.5

        return x + right_x * offset, y + right_y * offset


def main(args=None):
    rclpy.init(args=args)

    node = PrivateCarSimulatorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()