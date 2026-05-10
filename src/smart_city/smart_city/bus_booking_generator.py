import json
import os
import random
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class BusBookingGenerator(Node):

    def __init__(self):
        super().__init__("bus_booking_generator")

        self.declare_parameter("bus_stops_config_file", "config/bus_stops.json")

        self.config_file = self.get_parameter("bus_stops_config_file").value
        self.config = self.load_config(self.config_file)

        self.stops = self.config["bus_stops"]
        gen = self.config["request_generation"]

        self.min_interval = float(gen.get("min_interval_sec", 8.0))
        self.max_interval = float(gen.get("max_interval_sec", 22.0))

        self.pub = self.create_publisher(String, "/gazebo/bus_booking", 10)

        self.timer = self.create_timer(1.0, self.loop)
        self.next_request_time = time.time() + random.uniform(self.min_interval, self.max_interval)

        self.counter = 0

        self.get_logger().info("bus_booking_generator avviato")

    def load_config(self, file_path):
        if not os.path.isabs(file_path):
            file_path = os.path.join(os.getcwd(), file_path)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File bus stops non trovato: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def loop(self):
        now = time.time()

        if now < self.next_request_time:
            return

        stop = self.choose_stop()

        self.counter += 1

        payload = {
            "booking_id": f"bus_booking_{self.counter}",
            "stop_id": stop["stop_id"],
            "path_id": stop["path_id"],
            "x": float(stop["x"]),
            "y": float(stop["y"]),
            "priority": 2,
            "timestamp": now
        }

        msg = String()
        msg.data = json.dumps(payload)

        self.pub.publish(msg)

        self.get_logger().info(
            f"booking bus generato: {payload['booking_id']} stop={payload['stop_id']}"
        )

        self.next_request_time = now + random.uniform(self.min_interval, self.max_interval)

    def choose_stop(self):
        weights = [
            float(stop.get("spawn_probability", 1.0))
            for stop in self.stops
        ]

        return random.choices(self.stops, weights=weights, k=1)[0]


def main(args=None):
    rclpy.init(args=args)

    node = BusBookingGenerator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()