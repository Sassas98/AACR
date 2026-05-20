import json
import math
import subprocess
import textwrap
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class GazeboVisualController(Node):
    """
    Nodo puramente visuale per Gazebo.

    Responsabilità:
    - mostrare marker per richieste taxi ricevute su /gazebo/taxi_request;
    - rimuovere i marker taxi quando un taxi raggiunge il pickup;
    - mostrare marker semaforici in base agli status ricevuti su /traffic_light/status.

    Nota:
    - non genera richieste taxi;
    - non decide i semafori;
    - non gestisce più bus booking.
    """

    def __init__(self):
        super().__init__("gazebo_visual_controller")

        self.declare_parameter("world_name", "smart_city_world")
        self.declare_parameter("pickup_reached_distance", 3.6)
        self.declare_parameter("city_map_file", "config/city_map.json")

        self.world_name = self.get_parameter("world_name").value
        self.pickup_reached_distance = float(
            self.get_parameter("pickup_reached_distance").value
        )
        self.city_map_file = self.get_parameter("city_map_file").value

        self.intersection_positions = self.load_intersection_positions(
            self.city_map_file
        )

        self.active_taxi_requests = {}
        self.vehicle_states = {}

        # node_id -> colore attuale del marker visuale
        # Serve anche per sapere se il marker è già stato creato:
        # se non è presente qui, NON provo a cancellarlo.
        self.traffic_light_markers = {}

        self.create_subscription(
            String,
            "/gazebo/taxi_request",
            self.on_taxi_request,
            10
        )

        self.create_subscription(
            String,
            "/vehicle_states",
            self.on_vehicle_state,
            50
        )

        self.create_subscription(
            String,
            "/traffic_light/status",
            self.on_traffic_light_status,
            50
        )

        self.timer = self.create_timer(0.5, self.cleanup_loop)

        self.get_logger().info(
            f"gazebo_visual_controller avviato | world={self.world_name}"
        )

    # ------------------------------------------------------------
    # MAPPA / POSIZIONI INCROCI
    # ------------------------------------------------------------

    def load_intersection_positions(self, file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                city_map = json.load(f)
        except Exception as ex:
            self.get_logger().warn(
                f"impossibile leggere city_map_file={file_path}: {ex}"
            )
            return {}

        result = {}

        for node in city_map.get("nodes", []):
            node_id = node.get("id")

            if not node_id:
                continue

            try:
                result[node_id] = {
                    "x": float(node["x"]),
                    "y": float(node["y"]),
                }
            except Exception:
                continue

        return result

    # ------------------------------------------------------------
    # CALLBACK REQUEST TAXI
    # ------------------------------------------------------------

    def on_taxi_request(self, msg):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        request_id = data.get("request_id")

        if not request_id:
            return

        try:
            x = float(data.get("pickup_x"))
            y = float(data.get("pickup_y"))
        except Exception:
            self.get_logger().warn(
                f"taxi request {request_id} senza pickup valido: {data}"
            )
            return

        model_name = f"visual_taxi_request_{request_id}"

        # Se esiste già un marker con lo stesso id, lo aggiorno senza
        # causare spam inutile: lo cancello solo se lo avevo già creato.
        previous = self.active_taxi_requests.get(request_id)
        if previous is not None:
            self.delete_model(previous["model_name"])

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
            color=(1.0, 0.45, 0.0, 1.0),
            size=(0.55, 0.55, 0.7)
        )

        self.get_logger().info(
            f"spawn taxi request marker {request_id} a ({x:.2f}, {y:.2f})"
        )

    def on_vehicle_state(self, msg):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        vehicle_id = data.get("vehicle_id")

        if not vehicle_id:
            return

        self.vehicle_states[vehicle_id] = data

    # ------------------------------------------------------------
    # CALLBACK SEMAFORI
    # ------------------------------------------------------------

    def on_traffic_light_status(self, msg):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        node_id = data.get("node_id")

        if not node_id:
            return

        node = self.intersection_positions.get(node_id)

        # Se il traffic_light_manager pubblica uno status per un nodo che
        # non è più nella mappa, ignoro. Non provo a creare/cancellare nulla.
        if node is None:
            return

        color_name = self.extract_traffic_light_color(data)
        marker_name = f"visual_traffic_light_{node_id}"

        old_color = self.traffic_light_markers.get(node_id)

        # Colore invariato: non faccio niente.
        if old_color == color_name:
            return

        # Cancello solo se so di averlo creato prima.
        # Questo evita:
        # Entity named [visual_traffic_light_X] not found, so not removed.
        if old_color is not None:
            self.delete_model(marker_name)

        self.spawn_marker(
            model_name=marker_name,
            x=float(node["x"]),
            y=float(node["y"]),
            z=2.2,
            color=self.color_for_traffic_light(color_name),
            size=(0.45, 0.45, 0.45)
        )

        self.traffic_light_markers[node_id] = color_name

    def extract_traffic_light_color(self, data):
        color_name = data.get("color")

        if color_name in ("green", "yellow", "red"):
            return color_name

        # Fallback per manager che non pubblicano "color".
        allowed = data.get("allowed_movements", [])

        if allowed:
            return "green"

        return "red"

    # ------------------------------------------------------------
    # CLEANUP MARKER TAXI
    # ------------------------------------------------------------

    def cleanup_loop(self):
        self.cleanup_taxi_requests()

    def cleanup_taxi_requests(self):
        to_remove = []

        for request_id, req in list(self.active_taxi_requests.items()):
            for vehicle_id, vehicle in list(self.vehicle_states.items()):
                if not vehicle_id.startswith("taxi_"):
                    continue

                try:
                    vehicle_x = float(vehicle["x"])
                    vehicle_y = float(vehicle["y"])
                except Exception:
                    continue

                d = self.distance(
                    vehicle_x,
                    vehicle_y,
                    req["x"],
                    req["y"]
                )

                if d <= self.pickup_reached_distance:
                    to_remove.append(request_id)
                    break

        for request_id in to_remove:
            req = self.active_taxi_requests.pop(request_id, None)

            if req is None:
                continue

            self.delete_model(req["model_name"])
            self.get_logger().info(
                f"despawn taxi request marker {request_id}"
            )

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
            self.get_logger().warn(
                f"Gazebo service fallito {service}: {ex}"
            )

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
