import os
import json
import math
import time
from collections import defaultdict, Counter

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from geometry_msgs.msg import Twist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class SimulationReportRecorder(Node):

    def __init__(self):
        super().__init__("simulation_report_recorder")

        self.declare_parameter("duration_sec", 600.0)
        self.declare_parameter("output_dir", "simulation_report_graphs")
        self.declare_parameter("vehicle_ids", [
            "bus_1", "bus_2", "bus_3",
            "taxi_1", "taxi_2", "taxi_3",
            "private_car_1", "private_car_2", "private_car_3"
        ])

        self.duration_sec = float(self.get_parameter("duration_sec").value)
        self.output_dir_base = self.get_parameter("output_dir").value
        self.vehicle_ids = list(self.get_parameter("vehicle_ids").value)

        self.start_time = time.time()
        self.finished = False

        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(os.getcwd(), f"{self.output_dir_base}_{stamp}")
        os.makedirs(self.output_dir, exist_ok=True)

        self.vehicle_positions = defaultdict(list)
        self.vehicle_speeds = defaultdict(list)
        self.vehicle_cmds = defaultdict(list)
        self.vehicle_stop_events = defaultdict(int)

        self.bus_status_samples = []
        self.taxi_requests = []
        self.traffic_light_samples = []
        self.traffic_priority_requests = []

        self.create_subscription(String, "/vehicle_states", self.on_vehicle_state, 100)
        self.create_subscription(String, "/bus/status", self.on_bus_status, 50)
        self.create_subscription(String, "/gazebo/taxi_request", self.on_taxi_request, 50)
        self.create_subscription(String, "/traffic_light/status", self.on_traffic_light_status, 100)
        self.create_subscription(String, "/traffic_light/priority_request", self.on_priority_request, 100)

        for vehicle_id in self.vehicle_ids:
            self.create_subscription(
                Twist,
                f"/{vehicle_id}/cmd_vel",
                lambda msg, vid=vehicle_id: self.on_cmd_vel(vid, msg),
                50
            )

        self.timer = self.create_timer(1.0, self.check_finish)

        self.get_logger().info(
            f"SimulationReportRecorder started. Recording for {self.duration_sec:.0f}s. "
            f"Output directory: {self.output_dir}"
        )

    def rel_time(self):
        return time.time() - self.start_time

    def on_vehicle_state(self, msg):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        vehicle_id = data.get("vehicle_id")
        if not vehicle_id:
            return

        t = self.rel_time()
        x = float(data.get("x", 0.0))
        y = float(data.get("y", 0.0))
        yaw = float(data.get("yaw", 0.0))

        previous = self.vehicle_positions[vehicle_id][-1] if self.vehicle_positions[vehicle_id] else None

        self.vehicle_positions[vehicle_id].append({
            "t": t,
            "x": x,
            "y": y,
            "yaw": yaw
        })

        if previous:
            dt = max(1e-6, t - previous["t"])
            dx = x - previous["x"]
            dy = y - previous["y"]
            speed = math.sqrt(dx * dx + dy * dy) / dt
            self.vehicle_speeds[vehicle_id].append({
                "t": t,
                "speed": speed
            })

    def on_cmd_vel(self, vehicle_id, msg):
        t = self.rel_time()

        linear = float(msg.linear.x)
        angular = float(msg.angular.z)

        self.vehicle_cmds[vehicle_id].append({
            "t": t,
            "linear": linear,
            "angular": angular
        })

        if abs(linear) < 0.01 and abs(angular) < 0.01:
            self.vehicle_stop_events[vehicle_id] += 1

    def on_bus_status(self, msg):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        data["t"] = self.rel_time()
        self.bus_status_samples.append(data)

    def on_taxi_request(self, msg):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        data["t"] = self.rel_time()
        self.taxi_requests.append(data)

    def on_traffic_light_status(self, msg):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        data["t"] = self.rel_time()
        self.traffic_light_samples.append(data)

    def on_priority_request(self, msg):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        data["t"] = self.rel_time()
        self.traffic_priority_requests.append(data)

    def check_finish(self):
        if self.finished:
            return

        elapsed = self.rel_time()

        if elapsed < self.duration_sec:
            remaining = self.duration_sec - elapsed
            if int(remaining) % 60 == 0:
                self.get_logger().info(f"Recording... {remaining:.0f}s remaining")
            return

        self.finished = True
        self.generate_all_graphs()
        self.get_logger().info(f"Report graphs generated in: {self.output_dir}")

    def savefig(self, name):
        path = os.path.join(self.output_dir, name)
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        self.get_logger().info(f"Saved {path}")

    def generate_all_graphs(self):
        self.plot_vehicle_trajectories()
        self.plot_vehicle_speed_over_time()
        self.plot_vehicle_distance_travelled()
        self.plot_cmd_vel_activity()
        self.plot_stop_events()
        self.plot_bus_path_usage()
        self.plot_taxi_request_timeline()
        self.plot_traffic_light_activity()
        self.plot_priority_requests()

    def plot_vehicle_trajectories(self):
        plt.figure(figsize=(10, 8))

        for vehicle_id, points in self.vehicle_positions.items():
            if len(points) < 2:
                continue

            xs = [p["x"] for p in points]
            ys = [p["y"] for p in points]

            plt.plot(xs, ys, linewidth=1.5, label=vehicle_id)
            plt.scatter(xs[0], ys[0], marker="o", s=25)
            plt.scatter(xs[-1], ys[-1], marker="x", s=35)

        plt.title("Vehicle Trajectories During Simulation")
        plt.xlabel("World X position [m]")
        plt.ylabel("World Y position [m]")
        plt.axis("equal")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)

        self.savefig("01_vehicle_trajectories.png")

    def plot_vehicle_speed_over_time(self):
        plt.figure(figsize=(12, 6))

        for vehicle_id, samples in self.vehicle_speeds.items():
            if not samples:
                continue

            ts = [s["t"] / 60.0 for s in samples]
            speeds = [s["speed"] for s in samples]

            plt.plot(ts, speeds, linewidth=1.2, label=vehicle_id)

        plt.title("Estimated Vehicle Speed Over Time")
        plt.xlabel("Simulation time [min]")
        plt.ylabel("Estimated speed [m/s]")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)

        self.savefig("02_vehicle_speed_over_time.png")

    def plot_vehicle_distance_travelled(self):
        labels = []
        distances = []

        for vehicle_id, points in self.vehicle_positions.items():
            if len(points) < 2:
                continue

            total = 0.0

            for a, b in zip(points[:-1], points[1:]):
                total += self.distance(a["x"], a["y"], b["x"], b["y"])

            labels.append(vehicle_id)
            distances.append(total)

        plt.figure(figsize=(10, 5))
        plt.bar(labels, distances)

        plt.title("Total Distance Travelled by Vehicle")
        plt.xlabel("Vehicle")
        plt.ylabel("Distance [m]")
        plt.xticks(rotation=35, ha="right")
        plt.grid(True, axis="y", alpha=0.3)

        self.savefig("03_total_distance_travelled.png")

    def plot_cmd_vel_activity(self):
        labels = []
        avg_linear = []
        avg_abs_angular = []

        for vehicle_id, samples in self.vehicle_cmds.items():
            if not samples:
                continue

            labels.append(vehicle_id)
            avg_linear.append(sum(abs(s["linear"]) for s in samples) / len(samples))
            avg_abs_angular.append(sum(abs(s["angular"]) for s in samples) / len(samples))

        x = range(len(labels))

        plt.figure(figsize=(10, 5))
        plt.bar(x, avg_linear, label="Average |linear.x|")
        plt.plot(x, avg_abs_angular, marker="o", label="Average |angular.z|")

        plt.title("Controller Command Activity")
        plt.xlabel("Vehicle")
        plt.ylabel("Command magnitude")
        plt.xticks(x, labels, rotation=35, ha="right")
        plt.grid(True, axis="y", alpha=0.3)
        plt.legend()

        self.savefig("04_controller_command_activity.png")

    def plot_stop_events(self):
        labels = list(self.vehicle_stop_events.keys())
        values = [self.vehicle_stop_events[v] for v in labels]

        plt.figure(figsize=(10, 5))
        plt.bar(labels, values)

        plt.title("Stop Command Frequency by Vehicle")
        plt.xlabel("Vehicle")
        plt.ylabel("Number of zero cmd_vel samples")
        plt.xticks(rotation=35, ha="right")
        plt.grid(True, axis="y", alpha=0.3)

        self.savefig("05_stop_command_frequency.png")

    def plot_bus_path_usage(self):
        counter = Counter()

        for sample in self.bus_status_samples:
            bus_id = sample.get("bus_id", "unknown")
            path_id = sample.get("current_path_id", "unknown")
            counter[(bus_id, path_id)] += 1

        labels = [f"{bus}\n{path}" for (bus, path) in counter.keys()]
        values = list(counter.values())

        plt.figure(figsize=(10, 5))
        plt.bar(labels, values)

        plt.title("Bus Path Occupancy Samples")
        plt.xlabel("Bus / Path")
        plt.ylabel("Status samples")
        plt.grid(True, axis="y", alpha=0.3)

        self.savefig("06_bus_path_occupancy.png")

    def plot_taxi_request_timeline(self):
        if not self.taxi_requests:
            plt.figure(figsize=(10, 4))
            plt.title("Taxi Request Timeline")
            plt.text(0.5, 0.5, "No taxi requests recorded", ha="center", va="center")
            plt.axis("off")
            self.savefig("07_taxi_request_timeline.png")
            return

        ts = [r["t"] / 60.0 for r in self.taxi_requests]
        ys = list(range(1, len(ts) + 1))

        plt.figure(figsize=(10, 4))
        plt.step(ts, ys, where="post")
        plt.scatter(ts, ys)

        plt.title("Cumulative Taxi Requests Over Time")
        plt.xlabel("Simulation time [min]")
        plt.ylabel("Cumulative requests")
        plt.grid(True, alpha=0.3)

        self.savefig("07_taxi_request_timeline.png")

    def plot_traffic_light_activity(self):
        counter = Counter()

        for sample in self.traffic_light_samples:
            node_id = sample.get("node_id", "unknown")
            phase = sample.get("phase", "unknown")
            counter[(node_id, phase)] += 1

        if not counter:
            plt.figure(figsize=(10, 4))
            plt.title("Traffic Light Phase Activity")
            plt.text(0.5, 0.5, "No traffic light samples recorded", ha="center", va="center")
            plt.axis("off")
            self.savefig("08_traffic_light_phase_activity.png")
            return

        labels = [f"{node}\n{phase}" for (node, phase) in counter.keys()]
        values = list(counter.values())

        plt.figure(figsize=(12, 5))
        plt.bar(labels, values)

        plt.title("Traffic Light Phase Activity")
        plt.xlabel("Intersection / Phase")
        plt.ylabel("Status samples")
        plt.xticks(rotation=45, ha="right")
        plt.grid(True, axis="y", alpha=0.3)

        self.savefig("08_traffic_light_phase_activity.png")

    def plot_priority_requests(self):
        counter = Counter()

        for sample in self.traffic_priority_requests:
            node_id = sample.get("node_id", "unknown")
            counter[node_id] += 1

        if not counter:
            plt.figure(figsize=(10, 4))
            plt.title("Traffic Priority Requests")
            plt.text(0.5, 0.5, "No priority requests recorded", ha="center", va="center")
            plt.axis("off")
            self.savefig("09_traffic_priority_requests.png")
            return

        labels = list(counter.keys())
        values = list(counter.values())

        plt.figure(figsize=(10, 5))
        plt.bar(labels, values)

        plt.title("Traffic Priority Requests by Intersection")
        plt.xlabel("Intersection node")
        plt.ylabel("Number of requests")
        plt.grid(True, axis="y", alpha=0.3)

        self.savefig("09_traffic_priority_requests.png")

    def distance(self, x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        return math.sqrt(dx * dx + dy * dy)


def main(args=None):
    rclpy.init(args=args)

    node = SimulationReportRecorder()

    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        node.get_logger().warn("Interrupted manually, generating partial report...")
        node.generate_all_graphs()
        node.get_logger().info(f"Partial report graphs generated in: {node.output_dir}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()