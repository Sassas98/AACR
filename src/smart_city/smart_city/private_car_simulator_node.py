import json
import math
import os
import random
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


class PrivateCar:
    def __init__(self, vehicle_id):
        self.vehicle_id = vehicle_id
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.route = []
        self.target_index = 0
        self.last_goal_time = 0.0


class PrivateCarSimulatorNode(Node):

    def __init__(self):
        super().__init__("private_car_simulator_node")

        self.declare_parameter("vehicles_config_file", "config/vehicles.json")
        self.declare_parameter("map_config_file", "config/city_map.json")
        self.declare_parameter("linear_speed", 0.7)
        self.declare_parameter("angular_k", 1.8)
        self.declare_parameter("target_tolerance", 0.8)

        self.vehicles_file = self.get_parameter("vehicles_config_file").value
        self.map_file = self.get_parameter("map_config_file").value

        self.linear_speed = float(self.get_parameter("linear_speed").value)
        self.angular_k = float(self.get_parameter("angular_k").value)
        self.target_tolerance = float(self.get_parameter("target_tolerance").value)

        self.nodes = {}
        self.edges = []
        self.adj = {}

        self.private_cars = {}
        self.cmd_publishers = {}

        self.load_map()
        self.load_private_cars()

        self.timer = self.create_timer(0.1, self.loop)

        self.get_logger().info(
            f"private_car_simulator_node avviato con {len(self.private_cars)} auto private"
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
        data = self.load_json(self.map_file)

        for node in data["nodes"]:
            self.nodes[node["id"]] = {
                "x": float(node["x"]),
                "y": float(node["y"])
            }

        self.edges = data["edges"]

        for node_id in self.nodes:
            self.adj[node_id] = []

        for edge in self.edges:
            a = edge["from"]
            b = edge["to"]

            self.adj[a].append(b)
            self.adj[b].append(a)

    def load_private_cars(self):
        data = self.load_json(self.vehicles_file)

        for vehicle in data["vehicles"]:
            if vehicle.get("type") != "PRIVATE_CAR":
                continue

            vehicle_id = vehicle["id"]

            car = PrivateCar(vehicle_id)

            spawn = vehicle.get("spawn", {})
            car.x = float(spawn.get("x", 0.0))
            car.y = float(spawn.get("y", 0.0))
            car.yaw = float(spawn.get("yaw", 0.0))

            self.private_cars[vehicle_id] = car

            self.cmd_publishers[vehicle_id] = self.create_publisher(
                Twist,
                f"/{vehicle_id}/cmd_vel",
                10
            )

            self.create_subscription(
                Odometry,
                f"/{vehicle_id}/odom",
                lambda msg, vid=vehicle_id: self.on_odom(msg, vid),
                10
            )

            self.assign_new_route(car)

    # ------------------------------------------------------------------
    # CALLBACK ODOM
    # ------------------------------------------------------------------

    def on_odom(self, msg, vehicle_id):
        car = self.private_cars.get(vehicle_id)

        if car is None:
            return

        car.x = msg.pose.pose.position.x
        car.y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation

        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)

        car.yaw = math.atan2(siny_cosp, cosy_cosp)

    # ------------------------------------------------------------------
    # LOGICA
    # ------------------------------------------------------------------

    def loop(self):
        for car in self.private_cars.values():
            self.update_car(car)

    def update_car(self, car):
        if not car.route:
            self.assign_new_route(car)
            return

        if car.target_index >= len(car.route):
            self.assign_new_route(car)
            return

        target_node_id = car.route[car.target_index]
        target = self.nodes[target_node_id]

        dx = target["x"] - car.x
        dy = target["y"] - car.y

        distance = math.sqrt(dx * dx + dy * dy)

        if distance <= self.target_tolerance:
            car.target_index += 1

            if car.target_index >= len(car.route):
                self.assign_new_route(car)

            return

        self.drive_towards(car, target["x"], target["y"])

    def assign_new_route(self, car):
        start = self.find_nearest_node(car.x, car.y)

        if start is None:
            return

        route = [start]
        current = start

        steps = random.randint(3, 8)

        for _ in range(steps):
            neighbors = self.adj.get(current, [])

            if not neighbors:
                break

            current = random.choice(neighbors)
            route.append(current)

        car.route = route
        car.target_index = 1 if len(route) > 1 else 0
        car.last_goal_time = time.time()

        self.get_logger().info(
            f"{car.vehicle_id}: nuova route casuale {car.route}"
        )

    def drive_towards(self, car, target_x, target_y):
        dx = target_x - car.x
        dy = target_y - car.y

        target_angle = math.atan2(dy, dx)
        angle_error = self.normalize_angle(target_angle - car.yaw)

        cmd = Twist()

        if abs(angle_error) > 0.7:
            cmd.linear.x = 0.0
        else:
            cmd.linear.x = self.linear_speed

        cmd.angular.z = self.angular_k * angle_error

        self.cmd_publishers[car.vehicle_id].publish(cmd)

    # ------------------------------------------------------------------
    # UTILITY
    # ------------------------------------------------------------------

    def find_nearest_node(self, x, y):
        best_node = None
        best_distance = float("inf")

        for node_id, node in self.nodes.items():
            dx = node["x"] - x
            dy = node["y"] - y
            d = math.sqrt(dx * dx + dy * dy)

            if d < best_distance:
                best_distance = d
                best_node = node_id

        return best_node

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi

        while angle < -math.pi:
            angle += 2.0 * math.pi

        return angle


def main(args=None):
    rclpy.init(args=args)

    node = PrivateCarSimulatorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    for publisher in node.cmd_publishers.values():
        publisher.publish(Twist())

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()