import json
import math
import os
import time
from enum import Enum

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class LightPhase(str, Enum):
    GREEN_X = "GREEN_X"
    YELLOW_X = "YELLOW_X"
    ALL_RED_AFTER_X = "ALL_RED_AFTER_X"

    GREEN_Y = "GREEN_Y"
    YELLOW_Y = "YELLOW_Y"
    ALL_RED_AFTER_Y = "ALL_RED_AFTER_Y"


class TrafficLightManager(Node):
    """
    Semaforo semplice a due assi.

    - asse X = rami prevalentemente orizzontali dell'incrocio;
    - asse Y = rami prevalentemente verticali dell'incrocio;
    - GREEN_X: possono entrare solo i veicoli che arrivano dai rami X;
    - GREEN_Y: possono entrare solo i veicoli che arrivano dai rami Y;
    - YELLOW_*: nessun movimento consentito, ma l'asse uscente viene visualizzato giallo;
    - ALL_RED_*: tutti rossi.

    Durate base:
    - verde: 60s
    - giallo: 10s
    - rosso/all-red: 30s

    Priorità:
    - una richiesta sul prossimo asse può accorciare il rosso di massimo 20s.
    """

    def __init__(self):
        super().__init__("traffic_light_manager")

        self.declare_parameter("node_id", "n2")
        self.declare_parameter("map_config_file", "config/city_map.json")

        # Durate base del ciclo semaforico.
        # Verranno moltiplicate per phase_time_multiplier.
        self.declare_parameter("green_duration", 120.0)
        self.declare_parameter("yellow_duration", 10.0)
        self.declare_parameter("red_duration", 30.0)

        # Rallenta globalmente tutte le fasi.
        # 5.0 => semaforo 5 volte più lento.
        self.declare_parameter("phase_time_multiplier", 5.0)

        # Anche la priorità va scalata, altrimenti con fasi lente scade troppo presto.
        self.declare_parameter("priority_red_reduction_sec", 20.0)
        self.declare_parameter("priority_request_ttl_sec", 30.0)

        self.node_id = str(self.get_parameter("node_id").value)
        self.map_config_file = self.get_parameter("map_config_file").value

        self.phase_time_multiplier = float(
            self.get_parameter("phase_time_multiplier").value
        )
        self.phase_time_multiplier = max(0.1, self.phase_time_multiplier)

        base_green_duration = float(self.get_parameter("green_duration").value)
        base_yellow_duration = float(self.get_parameter("yellow_duration").value)
        base_red_duration = float(self.get_parameter("red_duration").value)

        base_priority_red_reduction = float(
            self.get_parameter("priority_red_reduction_sec").value
        )
        base_priority_request_ttl = float(
            self.get_parameter("priority_request_ttl_sec").value
        )

        self.green_duration = base_green_duration * self.phase_time_multiplier
        self.yellow_duration = base_yellow_duration * self.phase_time_multiplier
        self.red_duration = base_red_duration * self.phase_time_multiplier

        self.priority_red_reduction_sec = (
            base_priority_red_reduction * self.phase_time_multiplier
        )

        self.priority_request_ttl_sec = (
            base_priority_request_ttl * self.phase_time_multiplier
        )

        self.nodes = {}
        self.edges = []
        self.connected_nodes = []

        self.branch_axis = {}
        self.x_branches = []
        self.y_branches = []

        self.load_map()
        self.build_axes()

        self.current_phase = LightPhase.GREEN_X
        self.phase_started_at = time.time()
        self.priority_requests = []

        self.status_pub = self.create_publisher(
            String,
            "/traffic_light/status",
            10
        )

        self.priority_sub = self.create_subscription(
            String,
            "/traffic_light/priority_request",
            self.on_priority_request,
            10
        )

        self.timer = self.create_timer(0.25, self.update)

    # ============================================================
    # MAPPA
    # ============================================================

    def load_map(self):
        path = self.map_config_file

        if not os.path.isabs(path):
            path = os.path.join(os.getcwd(), path)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Mappa non trovata: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for node in data["nodes"]:
            self.nodes[node["id"]] = {
                "id": node["id"],
                "x": float(node["x"]),
                "y": float(node["y"]),
            }

        self.edges = data["edges"]

        if self.node_id not in self.nodes:
            raise ValueError(f"Nodo semaforo inesistente: {self.node_id}")

        connected = []

        for edge in self.edges:
            if edge["from"] == self.node_id:
                connected.append(edge["to"])
            elif edge["to"] == self.node_id:
                connected.append(edge["from"])

        self.connected_nodes = sorted(set(connected))

        if len(self.connected_nodes) < 2:
            raise ValueError(
                f"Il nodo {self.node_id} ha grado {len(self.connected_nodes)}: semaforo non utile"
            )

    def build_axes(self):
        center = self.nodes[self.node_id]

        for other_id in self.connected_nodes:
            other = self.nodes[other_id]

            dx = other["x"] - center["x"]
            dy = other["y"] - center["y"]

            axis = "X" if abs(dx) >= abs(dy) else "Y"
            self.branch_axis[other_id] = axis

            if axis == "X":
                self.x_branches.append(other_id)
            else:
                self.y_branches.append(other_id)

        # Fallback per incroci diagonali o mappe strane.
        if not self.x_branches or not self.y_branches:
            self.rebuild_axes_by_angle()

    def rebuild_axes_by_angle(self):
        center = self.nodes[self.node_id]

        ordered = sorted(
            self.connected_nodes,
            key=lambda node_id: math.atan2(
                self.nodes[node_id]["y"] - center["y"],
                self.nodes[node_id]["x"] - center["x"]
            )
        )

        self.branch_axis = {}
        self.x_branches = []
        self.y_branches = []

        for index, node_id in enumerate(ordered):
            axis = "X" if index % 2 == 0 else "Y"
            self.branch_axis[node_id] = axis

            if axis == "X":
                self.x_branches.append(node_id)
            else:
                self.y_branches.append(node_id)

    # ============================================================
    # PRIORITÀ
    # ============================================================

    def on_priority_request(self, msg):
        try:
            request = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn(f"Priority request non JSON: {msg.data}")
            return

        if request.get("node_id") != self.node_id:
            return

        required = [
            "vehicle_id",
            "mission_id",
            "from_node_id",
            "to_node_id",
            "priority",
        ]

        if not all(k in request for k in required):
            return

        from_node = request["from_node_id"]
        to_node = request["to_node_id"]

        if from_node not in self.connected_nodes:
            return

        if to_node is not None and to_node not in self.connected_nodes:
            return

        key = (
            request["vehicle_id"],
            request["mission_id"],
            from_node,
            to_node,
        )

        now = time.time()

        self.priority_requests = [
            r for r in self.priority_requests
            if (
                r["vehicle_id"],
                r["mission_id"],
                r["from_node_id"],
                r["to_node_id"],
            ) != key
        ]

        request["received_at"] = now
        request["axis"] = self.branch_axis.get(from_node)
        self.priority_requests.append(request)

    def cleanup_old_requests(self):
        now = time.time()

        self.priority_requests = [
            r for r in self.priority_requests
            if now - float(r.get("received_at", 0.0)) <= self.priority_request_ttl_sec
        ]

    def has_priority_for_axis(self, axis):
        return any(r.get("axis") == axis for r in self.priority_requests)

    # ============================================================
    # UPDATE FASI
    # ============================================================

    def update(self):
        self.cleanup_old_requests()

        elapsed = time.time() - self.phase_started_at

        if self.current_phase == LightPhase.GREEN_X:
            if elapsed >= self.green_duration:
                self.set_phase(LightPhase.YELLOW_X)

        elif self.current_phase == LightPhase.YELLOW_X:
            if elapsed >= self.yellow_duration:
                self.set_phase(LightPhase.ALL_RED_AFTER_X)

        elif self.current_phase == LightPhase.ALL_RED_AFTER_X:
            if elapsed >= self.effective_red_duration(next_axis="Y"):
                self.set_phase(LightPhase.GREEN_Y)

        elif self.current_phase == LightPhase.GREEN_Y:
            if elapsed >= self.green_duration:
                self.set_phase(LightPhase.YELLOW_Y)

        elif self.current_phase == LightPhase.YELLOW_Y:
            if elapsed >= self.yellow_duration:
                self.set_phase(LightPhase.ALL_RED_AFTER_Y)

        elif self.current_phase == LightPhase.ALL_RED_AFTER_Y:
            if elapsed >= self.effective_red_duration(next_axis="X"):
                self.set_phase(LightPhase.GREEN_X)

        self.publish_status()

    def effective_red_duration(self, next_axis):
        duration = self.red_duration

        if self.has_priority_for_axis(next_axis):
            duration -= self.priority_red_reduction_sec

        # Anche con priorità, non saltare mai completamente la fase rossa.
        # Serve a dare tempo all'incrocio di svuotarsi.
        min_red_duration = max(3.0, self.red_duration * 0.20)

        return max(min_red_duration, duration)

    def set_phase(self, phase):
        if phase == self.current_phase:
            return

        self.current_phase = phase
        self.phase_started_at = time.time()

    # ============================================================
    # LOGICA SEMAFORICA
    # ============================================================

    def active_green_axis(self):
        if self.current_phase == LightPhase.GREEN_X:
            return "X"
        if self.current_phase == LightPhase.GREEN_Y:
            return "Y"
        return None

    def active_yellow_axis(self):
        if self.current_phase == LightPhase.YELLOW_X:
            return "X"
        if self.current_phase == LightPhase.YELLOW_Y:
            return "Y"
        return None

    def phase_color(self):
        if self.active_green_axis() is not None:
            return "green"
        if self.active_yellow_axis() is not None:
            return "yellow"
        return "red"

    def get_allowed_movements(self):
        green_axis = self.active_green_axis()

        if green_axis is None:
            return []

        movements = []

        for from_node in self.connected_nodes:
            if self.branch_axis.get(from_node) != green_axis:
                continue

            for to_node in self.connected_nodes:
                if to_node == from_node:
                    continue

                movements.append({
                    "from": from_node,
                    "to": to_node,
                })

        return movements

    def get_signal_states(self):
        green_axis = self.active_green_axis()
        yellow_axis = self.active_yellow_axis()

        result = []
        center = self.nodes[self.node_id]

        for from_node in self.connected_nodes:
            axis = self.branch_axis.get(from_node)

            if axis == green_axis:
                color = "green"
            elif axis == yellow_axis:
                color = "yellow"
            else:
                color = "red"

            node = self.nodes[from_node]

            dx = node["x"] - center["x"]
            dy = node["y"] - center["y"]
            length = max(0.000001, math.sqrt(dx * dx + dy * dy))

            ux = dx / length
            uy = dy / length

            marker_distance = 2.0

            result.append({
                "node_id": self.node_id,
                "from_node_id": from_node,
                "axis": axis,
                "color": color,
                "x": center["x"] + ux * marker_distance,
                "y": center["y"] + uy * marker_distance,
            })

        return result

    def publish_status(self):
        payload = {
            "node_id": self.node_id,
            "phase": self.current_phase.value,
            "color": self.phase_color(),
            "green_axis": self.active_green_axis(),
            "yellow_axis": self.active_yellow_axis(),
            "connected_nodes": self.connected_nodes,
            "branch_axis": self.branch_axis,
            "x_branches": self.x_branches,
            "y_branches": self.y_branches,
            "allowed_movements": self.get_allowed_movements(),
            "signal_states": self.get_signal_states(),
            "pending_requests": [
                {
                    "vehicle_id": r["vehicle_id"],
                    "mission_id": r["mission_id"],
                    "from_node_id": r["from_node_id"],
                    "to_node_id": r["to_node_id"],
                    "priority": r["priority"],
                    "axis": r.get("axis"),
                }
                for r in self.priority_requests
            ],
        }

        msg = String()
        msg.data = json.dumps(payload)
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightManager()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
