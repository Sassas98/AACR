import json
import os
import time
from enum import Enum

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class LightPhase(str, Enum):
    PHASE_A = "PHASE_A"
    PHASE_B = "PHASE_B"


class TrafficLightManager(Node):

    def __init__(self):
        super().__init__("traffic_light_manager")

        self.declare_parameter("node_id", "n2")
        self.declare_parameter("map_config_file", "config/city_map.json")
        self.declare_parameter("base_phase_duration", 6.0)
        self.declare_parameter("min_phase_duration", 3.0)
        self.declare_parameter("max_phase_duration", 12.0)

        self.node_id = self.get_parameter("node_id").value
        self.base_phase_duration = float(self.get_parameter("base_phase_duration").value)
        self.min_phase_duration = float(self.get_parameter("min_phase_duration").value)
        self.max_phase_duration = float(self.get_parameter("max_phase_duration").value)

        self.nodes = {}
        self.edges = []
        self.connected_nodes = []

        self.load_map()

        self.current_phase = LightPhase.PHASE_A
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

        self.timer = self.create_timer(
            0.5,
            self.update
        )

        self.get_logger().info(
            f"traffic_light_manager avviato su nodo {self.node_id}, "
            f"grado={len(self.connected_nodes)}, connected={self.connected_nodes}"
        )

    # ------------------------------------------------------------------
    # MAPPA
    # ------------------------------------------------------------------

    def load_map(self):
        file_path = self.get_parameter("map_config_file").value

        if not os.path.isabs(file_path):
            file_path = os.path.join(os.getcwd(), file_path)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Mappa non trovata: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for node in data["nodes"]:
            self.nodes[node["id"]] = node

        self.edges = data["edges"]

        if self.node_id not in self.nodes:
            raise ValueError(f"Nodo semaforo inesistente: {self.node_id}")

        self.connected_nodes = []

        for edge in self.edges:
            if edge["from"] == self.node_id:
                self.connected_nodes.append(edge["to"])
            elif edge["to"] == self.node_id:
                self.connected_nodes.append(edge["from"])

        degree = len(self.connected_nodes)

        if degree < 3:
            raise ValueError(
                f"Il nodo {self.node_id} ha grado {degree}: "
                f"non richiede semaforo secondo la tua logica"
            )

        if degree > 4:
            raise ValueError(
                f"Il nodo {self.node_id} ha grado {degree}: "
                f"la logica prevista supporta massimo 4 archi"
            )

    # ------------------------------------------------------------------
    # PRIORITY REQUEST
    # ------------------------------------------------------------------

    def on_priority_request(self, msg):
        try:
            request = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn(f"Priority request non JSON: {msg.data}")
            return

        if request.get("node_id") != self.node_id:
            return

        required = ["vehicle_id", "mission_id", "from_node_id", "to_node_id", "priority"]

        if not all(k in request for k in required):
            self.get_logger().warn(f"Priority request incompleta: {request}")
            return

        from_node = request["from_node_id"]
        to_node = request["to_node_id"]

        if from_node not in self.connected_nodes:
            return

        if to_node is not None and to_node not in self.connected_nodes:
            return

        request["received_at"] = time.time()

        self.priority_requests.append(request)

        self.get_logger().info(
            f"{self.node_id}: richiesta priorità da {request['vehicle_id']} "
            f"{from_node}->{self.node_id}->{to_node}, priority={request['priority']}"
        )

    # ------------------------------------------------------------------
    # UPDATE
    # ------------------------------------------------------------------

    def update(self):
        self.cleanup_old_requests()

        desired_phase = self.choose_phase()

        elapsed = time.time() - self.phase_started_at

        if desired_phase != self.current_phase and elapsed >= self.min_phase_duration:
            self.current_phase = desired_phase
            self.phase_started_at = time.time()

        elif elapsed >= self.max_phase_duration:
            self.switch_phase()

        elif elapsed >= self.base_phase_duration and desired_phase != self.current_phase:
            self.current_phase = desired_phase
            self.phase_started_at = time.time()

        self.publish_status()

    def cleanup_old_requests(self):
        now = time.time()

        self.priority_requests = [
            r for r in self.priority_requests
            if now - r["received_at"] <= 10.0
        ]

    def choose_phase(self):
        if not self.priority_requests:
            elapsed = time.time() - self.phase_started_at

            if elapsed >= self.base_phase_duration:
                return self.other_phase()

            return self.current_phase

        score_a = 0
        score_b = 0

        for request in self.priority_requests:
            movement = {
                "from": request["from_node_id"],
                "to": request["to_node_id"]
            }

            priority = int(request.get("priority", 1))

            if self.movement_allowed_in_phase(movement, LightPhase.PHASE_A):
                score_a += priority

            if self.movement_allowed_in_phase(movement, LightPhase.PHASE_B):
                score_b += priority

        if score_a > score_b:
            return LightPhase.PHASE_A

        if score_b > score_a:
            return LightPhase.PHASE_B

        return self.current_phase

    def switch_phase(self):
        self.current_phase = self.other_phase()
        self.phase_started_at = time.time()

    def other_phase(self):
        if self.current_phase == LightPhase.PHASE_A:
            return LightPhase.PHASE_B

        return LightPhase.PHASE_A

    # ------------------------------------------------------------------
    # LOGICA SEMAFORO
    # ------------------------------------------------------------------

    def get_allowed_movements(self):
        degree = len(self.connected_nodes)

        if degree == 3:
            return self.get_allowed_movements_degree_3()

        if degree == 4:
            return self.get_allowed_movements_degree_4()

        return []

    def get_allowed_movements_degree_3(self):
        """
        Grado 3:
        - fase A: passano i due rami principali tra loro
        - fase B: passa il ramo laterale verso/da uno dei principali

        Per semplicità:
        connected_nodes[0] e connected_nodes[1] = via continua
        connected_nodes[2] = terza via
        """

        main_a = self.connected_nodes[0]
        main_b = self.connected_nodes[1]
        side = self.connected_nodes[2]

        if self.current_phase == LightPhase.PHASE_A:
            return [
                {"from": main_a, "to": main_b},
                {"from": main_b, "to": main_a}
            ]

        return [
            {"from": side, "to": main_a},
            {"from": side, "to": main_b},
            {"from": main_a, "to": side},
            {"from": main_b, "to": side}
        ]

    def get_allowed_movements_degree_4(self):
        """
        Grado 4:
        - fase A: passano connected_nodes[0] <-> connected_nodes[1]
        - fase B: passano connected_nodes[2] <-> connected_nodes[3]

        Blocca quindi due vie per volta.
        """

        a = self.connected_nodes[0]
        b = self.connected_nodes[1]
        c = self.connected_nodes[2]
        d = self.connected_nodes[3]

        if self.current_phase == LightPhase.PHASE_A:
            return [
                {"from": a, "to": b},
                {"from": b, "to": a}
            ]

        return [
            {"from": c, "to": d},
            {"from": d, "to": c}
        ]

    def movement_allowed_in_phase(self, movement, phase):
        current_phase_backup = self.current_phase

        self.current_phase = phase
        allowed = self.get_allowed_movements()

        self.current_phase = current_phase_backup

        for item in allowed:
            if item["from"] == movement["from"] and item["to"] == movement["to"]:
                return True

        return False

    # ------------------------------------------------------------------
    # STATUS
    # ------------------------------------------------------------------

    def publish_status(self):
        payload = {
            "node_id": self.node_id,
            "phase": self.current_phase.value,
            "degree": len(self.connected_nodes),
            "connected_nodes": self.connected_nodes,
            "allowed_movements": self.get_allowed_movements(),
            "pending_requests": [
                {
                    "vehicle_id": r["vehicle_id"],
                    "mission_id": r["mission_id"],
                    "from_node_id": r["from_node_id"],
                    "to_node_id": r["to_node_id"],
                    "priority": r["priority"]
                }
                for r in self.priority_requests
            ]
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