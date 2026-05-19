import json
import math
import subprocess
import textwrap
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class GazeboVisualController(Node):

    def __init__(self):
        super().__init__("gazebo_visual_controller")

        self.declare_parameter("world_name", "default")
        self.declare_parameter("pickup_reached_distance", 1.2)
        self.declare_parameter("bus_stop_reached_distance", 1.5)
        self.declare_parameter("city_map_file", "config/city_map.json")

        self.city_map_file = self.get_parameter("city_map_file").value
        self.intersection_positions = self.load_intersection_positions(
            self.city_map_file
        )

        self.world_name = self.get_parameter("world_name").value
        self.pickup_reached_distance = float(self.get_parameter("pickup_reached_distance").value)
        self.bus_stop_reached_distance = float(self.get_parameter("bus_stop_reached_distance").value)

        self.active_taxi_requests = {}
        self.active_bus_bookings = {}
        self.vehicle_states = {}
        self.traffic_light_markers = {}

        self.create_subscription(String, "/gazebo/taxi_request", self.on_taxi_request, 10)
        self.create_subscription(String, "/gazebo/bus_booking", self.on_bus_booking, 10)
        self.create_subscription(String, "/vehicle_states", self.on_vehicle_state, 50)
        self.create_subscription(String, "/traffic_light/status", self.on_traffic_light_status, 10)

        self.timer = self.create_timer(0.5, self.cleanup_loop)

        self.get_logger().info("gazebo_visual_controller avviato")

    def load_intersection_positions(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            city_map = json.load(f)

        result = {}

        for node in city_map["nodes"]:
            result[node["id"]] = {
                "x": float(node["x"]),
                "y": float(node["y"]),
            }

        return result

    # ------------------------------------------------------------
    # CALLBACK REQUEST
    # ------------------------------------------------------------

    def on_taxi_request(self, msg):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        request_id = data.get("request_id")
        x = float(data.get("pickup_x"))
        y = float(data.get("pickup_y"))

        if not request_id:
            return

        model_name = f"visual_taxi_request_{request_id}"

        self.active_taxi_requests[request_id] = {
            "model_name": model_name,
            "x": x,
            "y": y,
            "created_at": time.time(),
        }

        self.spawn_marker(
            model_name=model_name,
            x=x,
            y=y,
            z=0.35,
            color=(1.0, 0.85, 0.05, 1.0),
            size=(0.55, 0.55, 0.7)
        )

        self.get_logger().info(f"spawn taxi request marker {request_id}")

    def on_bus_booking(self, msg):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        booking_id = data.get("booking_id")
        x = float(data.get("x"))
        y = float(data.get("y"))

        if not booking_id:
            return

        model_name = f"visual_bus_booking_{booking_id}"

        self.active_bus_bookings[booking_id] = {
            "model_name": model_name,
            "x": x,
            "y": y,
            "created_at": time.time(),
        }

        self.spawn_marker(
            model_name=model_name,
            x=x,
            y=y,
            z=0.4,
            color=(1.0, 0.45, 0.05, 1.0),
            size=(0.75, 0.75, 0.8)
        )

        self.get_logger().info(f"spawn bus booking marker {booking_id}")

    def on_vehicle_state(self, msg):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        vehicle_id = data.get("vehicle_id")

        if not vehicle_id:
            return

        self.vehicle_states[vehicle_id] = data

    def on_traffic_light_status(self, msg):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        node_id = data.get("node_id")

        if not node_id:
            return

        # Se il manager pubblica un campo "color", lo uso.
        # Altrimenti deduco verde se ci sono movimenti consentiti.
        color_name = data.get("color")

        if color_name is None:
            allowed = data.get("allowed_movements", [])
            color_name = "green" if allowed else "red"

        marker_name = f"visual_traffic_light_{node_id}"

        old = self.traffic_light_markers.get(node_id)
        if old == color_name:
            return

        self.delete_model(marker_name)

        node = self.intersection_positions.get(node_id)

        if node is None:
            return

        x = float(node["x"])
        y = float(node["y"])

        color = self.color_for_traffic_light(color_name)

        self.spawn_marker(
            model_name=marker_name,
            x=x,
            y=y,
            z=2.2,
            color=color,
            size=(0.45, 0.45, 0.45)
        )

        self.traffic_light_markers[node_id] = color_name

    # ------------------------------------------------------------
    # CLEANUP MARKER
    # ------------------------------------------------------------

    def cleanup_loop(self):
        self.cleanup_taxi_requests()
        self.cleanup_bus_bookings()

    def cleanup_taxi_requests(self):
        to_remove = []

        for request_id, req in self.active_taxi_requests.items():
            for vehicle_id, vehicle in self.vehicle_states.items():
                if not vehicle_id.startswith("taxi_"):
                    continue

                d = self.distance(
                    float(vehicle["x"]),
                    float(vehicle["y"]),
                    req["x"],
                    req["y"]
                )

                if d <= self.pickup_reached_distance:
                    to_remove.append(request_id)
                    break

        for request_id in to_remove:
            req = self.active_taxi_requests.pop(request_id)
            self.delete_model(req["model_name"])
            self.get_logger().info(f"despawn taxi request marker {request_id}")

    def cleanup_bus_bookings(self):
        to_remove = []

        for booking_id, booking in self.active_bus_bookings.items():
            for vehicle_id, vehicle in self.vehicle_states.items():
                if not vehicle_id.startswith("bus_"):
                    continue

                d = self.distance(
                    float(vehicle["x"]),
                    float(vehicle["y"]),
                    booking["x"],
                    booking["y"]
                )

                if d <= self.bus_stop_reached_distance:
                    to_remove.append(booking_id)
                    break

        for booking_id in to_remove:
            booking = self.active_bus_bookings.pop(booking_id)
            self.delete_model(booking["model_name"])
            self.get_logger().info(f"despawn bus booking marker {booking_id}")

    # ------------------------------------------------------------
    # GAZEBO
    # ------------------------------------------------------------

    def spawn_marker(self, model_name, x, y, z, color, size):
        r, g, b, a = color
        sx, sy, sz = size

        sdf = f"""
        <sdf version='1.9'>
          <model name='{model_name}'>
            <static>true</static>
            <pose>{x} {y} {z} 0 0 0</pose>
            <link name='link'>
              <visual name='visual'>
                <geometry>
                  <box>
                    <size>{sx} {sy} {sz}</size>
                  </box>
                </geometry>
                <material>
                  <ambient>{r} {g} {b} {a}</ambient>
                  <diffuse>{r} {g} {b} {a}</diffuse>
                </material>
              </visual>
              <collision name='collision'>
                <geometry>
                  <box>
                    <size>{sx} {sy} {sz}</size>
                  </box>
                </geometry>
              </collision>
            </link>
          </model>
        </sdf>
        """

        sdf = " ".join(textwrap.dedent(sdf).split())

        req = f'sdf: "{sdf}", name: "{model_name}"'

        self.call_gz_service(
            service=f"/world/{self.world_name}/create",
            reqtype="gz.msgs.EntityFactory",
            reptype="gz.msgs.Boolean",
            req=req
        )

    def delete_model(self, model_name):
        req = f'name: "{model_name}", type: MODEL'

        self.call_gz_service(
            service=f"/world/{self.world_name}/remove",
            reqtype="gz.msgs.Entity",
            reptype="gz.msgs.Boolean",
            req=req
        )

    def call_gz_service(self, service, reqtype, reptype, req):
        cmd = [
            "gz", "service",
            "-s", service,
            "--reqtype", reqtype,
            "--reptype", reptype,
            "--timeout", "1000",
            "--req", req
        ]

        try:
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2.0
            )
        except Exception as ex:
            self.get_logger().warn(f"Gazebo service fallito {service}: {ex}")

    # ------------------------------------------------------------
    # UTILITY
    # ------------------------------------------------------------

    def color_for_traffic_light(self, color_name):
        if color_name == "green":
            return (0.0, 1.0, 0.1, 1.0)

        if color_name == "yellow":
            return (1.0, 0.8, 0.0, 1.0)

        return (1.0, 0.0, 0.0, 1.0)

    def distance(self, x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        return math.sqrt(dx * dx + dy * dy)


def main(args=None):
    rclpy.init(args=args)

    node = GazeboVisualController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()