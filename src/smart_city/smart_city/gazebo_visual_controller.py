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
    Visual controller coerente con semafori ad assi.

    Mostra:
    - taxi request: cubi arancioni intangibili;
    - semafori: un cubo per ogni ramo dell'incrocio.

    Regola visuale semafori:
    - se asse X e' attivo: rami X verdi, rami Y rossi;
    - se asse Y e' attivo: rami Y verdi, rami X rossi;
    - NON visualizza mai tutti rossi.

    Nota: se il traffic_light_manager pubblica fasi transitorie tipo
    YELLOW_X o ALL_RED_AFTER_X, questo nodo le riconduce comunque a una
    configurazione a due assi, cosi' la visualizzazione resta semplice:
    sempre due verdi e due rossi quando l'incrocio ha 4 rami.
    """

    def __init__(self):
        super().__init__("gazebo_visual_controller")

        self.declare_parameter("world_name", "smart_city_world")
        self.declare_parameter("city_map_file", "config/city_map.json")
        self.declare_parameter("pickup_reached_distance", 3.6)

        self.declare_parameter("traffic_light_marker_distance", 2.4)
        self.declare_parameter("traffic_light_marker_z", 0.30)
        self.declare_parameter("traffic_light_marker_size", 0.65)

        self.world_name = str(self.get_parameter("world_name").value)
        self.city_map_file = str(self.get_parameter("city_map_file").value)
        self.pickup_reached_distance = float(
            self.get_parameter("pickup_reached_distance").value
        )

        self.traffic_light_marker_distance = float(
            self.get_parameter("traffic_light_marker_distance").value
        )
        self.traffic_light_marker_z = float(
            self.get_parameter("traffic_light_marker_z").value
        )
        self.traffic_light_marker_size = float(
            self.get_parameter("traffic_light_marker_size").value
        )

        self.nodes = {}
        self.load_city_map(self.city_map_file)

        self.active_taxi_requests = {}
        self.vehicle_states = {}

        # key=(node_id, from_node_id), value=color
        self.traffic_light_markers = {}

        # Ricordo l'ultimo asse visualizzato per nodo, per evitare flash strani.
        self.last_visual_green_axis_by_node = {}

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
            20
        )

        self.timer = self.create_timer(0.5, self.cleanup_loop)


    # ============================================================
    # MAPPA
    # ============================================================

    def load_city_map(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            city_map = json.load(f)

        self.nodes = {
            node["id"]: {
                "id": node["id"],
                "x": float(node["x"]),
                "y": float(node["y"]),
            }
            for node in city_map.get("nodes", [])
        }

    # ============================================================
    # TAXI REQUEST
    # ============================================================

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
        except (TypeError, ValueError):
            return

        model_name = f"visual_taxi_request_{request_id}"

        if request_id in self.active_taxi_requests:
            self.delete_model(self.active_taxi_requests[request_id]["model_name"])

        self.active_taxi_requests[request_id] = {
            "model_name": model_name,
            "x": x,
            "y": y,
            "created_at": time.time(),
        }

        self.spawn_box_marker(
            model_name=model_name,
            x=x,
            y=y,
            z=0.35,
            color=(1.0, 0.50, 0.0, 1.0),
            size=(0.55, 0.55, 0.7),
            collision=False
        )

    def cleanup_loop(self):
        self.cleanup_taxi_requests()

    def cleanup_taxi_requests(self):
        reached_requests = []

        for request_id, req in list(self.active_taxi_requests.items()):
            for vehicle_id, vehicle in self.vehicle_states.items():
                if not str(vehicle_id).startswith("taxi_"):
                    continue

                try:
                    d = self.distance(
                        float(vehicle["x"]),
                        float(vehicle["y"]),
                        req["x"],
                        req["y"]
                    )
                except (KeyError, TypeError, ValueError):
                    continue

                if d <= self.pickup_reached_distance:
                    reached_requests.append((request_id, vehicle_id))
                    break

        for request_id, vehicle_id in reached_requests:
            req = self.active_taxi_requests.pop(request_id, None)
            if req:
                self.get_logger().info(
                    f"obbiettivo raggiunto in "
                    f"({req['x']:.2f}, {req['y']:.2f}) "
                    f"da veicolo {vehicle_id}"
                )
                self.delete_model(req["model_name"])

    # ============================================================
    # VEHICLE STATE
    # ============================================================

    def on_vehicle_state(self, msg):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        vehicle_id = data.get("vehicle_id")
        if not vehicle_id:
            return

        self.vehicle_states[vehicle_id] = data

    # ============================================================
    # TRAFFIC LIGHTS AD ASSI
    # ============================================================

    def on_traffic_light_status(self, msg):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        node_id = data.get("node_id")
        if not node_id or node_id not in self.nodes:
            return

        signal_states = data.get("signal_states", [])
        if not signal_states:
            return

        branch_axis = self.get_branch_axis_map(data, signal_states)
        visual_green_axis = self.choose_visual_green_axis(data, node_id, branch_axis)

        if visual_green_axis is None:
            # Fallback estremo: scelgo l'asse del primo ramo.
            axes = sorted(set(branch_axis.values()))
            if axes:
                visual_green_axis = axes[0]
            else:
                return

        self.last_visual_green_axis_by_node[node_id] = visual_green_axis

        valid_keys = set()

        for state in signal_states:
            from_node_id = state.get("from_node_id")
            if not from_node_id or from_node_id not in self.nodes:
                continue

            axis = branch_axis.get(from_node_id)
            if axis is None:
                axis = self.compute_axis_from_geometry(node_id, from_node_id)

            color_name = "green" if axis == visual_green_axis else "red"

            key = (node_id, from_node_id)
            valid_keys.add(key)

            self.update_branch_traffic_light_marker(
                node_id=node_id,
                from_node_id=from_node_id,
                color_name=color_name
            )

        self.remove_missing_branch_markers(node_id, valid_keys)

    def get_branch_axis_map(self, data, signal_states):
        branch_axis = {}

        raw_branch_axis = data.get("branch_axis")
        if isinstance(raw_branch_axis, dict):
            for node_id, axis in raw_branch_axis.items():
                if axis in ("X", "Y"):
                    branch_axis[node_id] = axis

        for state in signal_states:
            from_node_id = state.get("from_node_id")
            axis = state.get("axis")

            if from_node_id and axis in ("X", "Y"):
                branch_axis[from_node_id] = axis

        node_id = data.get("node_id")
        if node_id:
            for state in signal_states:
                from_node_id = state.get("from_node_id")
                if from_node_id and from_node_id not in branch_axis:
                    branch_axis[from_node_id] = self.compute_axis_from_geometry(
                        node_id,
                        from_node_id
                    )

        return branch_axis

    def choose_visual_green_axis(self, data, node_id, branch_axis):
        green_axis = data.get("green_axis")
        if green_axis in ("X", "Y"):
            return green_axis

        phase = str(data.get("phase", ""))

        if phase == "GREEN_X":
            return "X"
        if phase == "GREEN_Y":
            return "Y"

        # Niente all-red visuale:
        # - YELLOW_X rimane visualizzato come X verde;
        # - ALL_RED_AFTER_X viene visualizzato come Y verde, cioe' il prossimo asse;
        # - YELLOW_Y rimane visualizzato come Y verde;
        # - ALL_RED_AFTER_Y viene visualizzato come X verde.
        if phase == "YELLOW_X":
            return "X"
        if phase == "YELLOW_Y":
            return "Y"
        if phase == "ALL_RED_AFTER_X":
            return "Y"
        if phase == "ALL_RED_AFTER_Y":
            return "X"

        # Fallback per altri nomi fase.
        if "X" in phase:
            return "X"
        if "Y" in phase:
            return "Y"

        old_axis = self.last_visual_green_axis_by_node.get(node_id)
        if old_axis in ("X", "Y"):
            return old_axis

        # Ultimo fallback: provo a dedurre dai colori pubblicati.
        green_votes = {"X": 0, "Y": 0}
        for state in data.get("signal_states", []):
            if state.get("color") != "green":
                continue
            from_node_id = state.get("from_node_id")
            axis = branch_axis.get(from_node_id)
            if axis in green_votes:
                green_votes[axis] += 1

        if green_votes["X"] > green_votes["Y"]:
            return "X"
        if green_votes["Y"] > green_votes["X"]:
            return "Y"

        return None

    def compute_axis_from_geometry(self, node_id, from_node_id):
        center = self.nodes.get(node_id)
        other = self.nodes.get(from_node_id)

        if center is None or other is None:
            return "X"

        dx = other["x"] - center["x"]
        dy = other["y"] - center["y"]

        return "X" if abs(dx) >= abs(dy) else "Y"

    def update_branch_traffic_light_marker(self, node_id, from_node_id, color_name):
        key = (node_id, from_node_id)
        old_color = self.traffic_light_markers.get(key)

        if old_color == color_name:
            return

        marker_name = self.branch_marker_name(node_id, from_node_id)

        if old_color is not None:
            self.delete_model(marker_name)

        x, y = self.compute_branch_marker_position(node_id, from_node_id)
        color = self.color_for_traffic_light(color_name)
        size = self.traffic_light_marker_size

        self.spawn_box_marker(
            model_name=marker_name,
            x=x,
            y=y,
            z=self.traffic_light_marker_z,
            color=color,
            size=(size, size, size),
            collision=False
        )

        self.traffic_light_markers[key] = color_name

    def remove_missing_branch_markers(self, node_id, valid_keys):
        for key in list(self.traffic_light_markers.keys()):
            marker_node_id, marker_from_id = key

            if marker_node_id != node_id:
                continue

            if key in valid_keys:
                continue

            marker_name = self.branch_marker_name(marker_node_id, marker_from_id)
            self.delete_model(marker_name)
            self.traffic_light_markers.pop(key, None)

    def compute_branch_marker_position(self, node_id, from_node_id):
        center = self.nodes[node_id]
        source = self.nodes[from_node_id]

        dx = source["x"] - center["x"]
        dy = source["y"] - center["y"]

        length = math.sqrt(dx * dx + dy * dy)
        if length < 0.000001:
            return center["x"], center["y"]

        ux = dx / length
        uy = dy / length

        return (
            center["x"] + ux * self.traffic_light_marker_distance,
            center["y"] + uy * self.traffic_light_marker_distance,
        )

    def branch_marker_name(self, node_id, from_node_id):
        return f"visual_traffic_light_{node_id}_from_{from_node_id}"

    # ============================================================
    # GAZEBO
    # ============================================================

    def spawn_box_marker(self, model_name, x, y, z, color, size, collision=False):
        r, g, b, a = color
        sx, sy, sz = size

        collision_xml = ""
        if collision:
            collision_xml = f"""
              <collision name='collision'>
                <geometry>
                  <box>
                    <size>{sx} {sy} {sz}</size>
                  </box>
                </geometry>
              </collision>
            """

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
              {collision_xml}
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
        except Exception:
            pass

    # ============================================================
    # UTILITY
    # ============================================================

    def color_for_traffic_light(self, color_name):
        if color_name == "green":
            return (0.0, 1.0, 0.1, 1.0)

        # niente gialli: visualizzazione sempre binaria rosso/verde
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
