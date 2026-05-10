import json
import os
import random
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class TaxiRequestGenerator(Node):

    def __init__(self):
        super().__init__("taxi_request_generator")

        self.declare_parameter(
            "taxi_request_zones_config_file",
            "config/taxi_request_zones.json"
        )

        self.config_file = self.get_parameter("taxi_request_zones_config_file").value
        self.config = self.load_config(self.config_file)

        gen = self.config["request_generation"]

        self.min_interval = float(gen.get("min_interval_sec", 6.0))
        self.max_interval = float(gen.get("max_interval_sec", 18.0))

        self.pickup_points = self.config["pickup_points"]
        self.dropoff_points = self.config["dropoff_points"]

        self.pub = self.create_publisher(String, "/gazebo/taxi_request", 10)

        self.timer = self.create_timer(1.0, self.loop)
        self.next_request_time = time.time() + random.uniform(self.min_interval, self.max_interval)

        self.counter = 0

        self.get_logger().info("taxi_request_generator avviato")

    def load_config(self, file_path):
        if not os.path.isabs(file_path):
            file_path = os.path.join(os.getcwd(), file_path)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File taxi zones non trovato: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def loop(self):
        now = time.time()

        if now < self.next_request_time:
            return

        pickup = self.choose_pickup()
        dropoff = self.choose_dropoff(pickup)

        self.counter += 1

        payload = {
            "request_id": f"taxi_request_{self.counter}",
            "pickup_id": pickup["id"],
            "pickup_x": float(pickup["x"]),
            "pickup_y": float(pickup["y"]),
            "dropoff_id": dropoff["id"],
            "dropoff_x": float(dropoff["x"]),
            "dropoff_y": float(dropoff["y"]),
            "timestamp": now
        }

        msg = String()
        msg.data = json.dumps(payload)

        self.pub.publish(msg)

        self.get_logger().info(
            f"taxi request generata: {payload['request_id']} "
            f"{pickup['id']} -> {dropoff['id']}"
        )

        self.next_request_time = now + random.uniform(self.min_interval, self.max_interval)

    def choose_pickup(self):
        weights = [
            float(point.get("weight", 1.0))
            for point in self.pickup_points
        ]

        return random.choices(self.pickup_points, weights=weights, k=1)[0]

    def choose_dropoff(self, pickup):
        candidates = [
            point for point in self.dropoff_points
            if point["id"] != pickup["id"]
        ]

        return random.choice(candidates)


def main(args=None):
    rclpy.init(args=args)

    node = TaxiRequestGenerator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()