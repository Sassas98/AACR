import json
import os
import time
from enum import Enum

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class LightPhase(str, Enum):
    GREEN_A = "GREEN_A"
    YELLOW_A = "YELLOW_A"
    ALL_RED_A = "ALL_RED_A"
    GREEN_B = "GREEN_B"
    YELLOW_B = "YELLOW_B"
    ALL_RED_B = "ALL_RED_B"


class TrafficLightManager(Node):
    """
    Manager semaforico per un singolo incrocio.

    Logica:
    - divide i rami dell'incrocio in due gruppi compatibili: gruppo A e gruppo B;
    - durante GREEN_A passano solo i movimenti del gruppo A;
    - durante GREEN_B passano solo i movimenti del gruppo B;
    - durante YELLOW_* e ALL_RED_* non passa nessuno: scelta conservativa per evitare incidenti;
    - pubblica anche signal_states per visualizzare rosso/verde per ogni ramo.

    Compatibile sia con i vecchi parametri:
      base_phase_duration, min_phase_duration, max_phase_duration
    sia con quelli del launch attuale:
      green_duration, yellow_duration, min_green_duration, max_green_duration
    """

    def __init__(self):
        super().__init__("traffic_light_manager")

        self.declare_parameter("node_id", "n2")
        self.declare_parameter("map_config_file", "config/city_map.json")

        # Vecchi parametri
        self.declare_parameter("base_phase_duration", 60.0)
        self.declare_parameter("min_phase_duration", 30.0)
        self.declare_parameter("max_phase_duration", 120.0)

        # Parametri usati dal tuo launch attuale
        self.declare_parameter("green_duration", 60.0)
        self.declare_parameter("yellow_duration", 10.0)
        self.declare_parameter("all_red_duration", 30.0)
        self.declare_parameter("min_green_duration", 3200.0)
        self.declare_parameter("max_green_duration", 6400.0)

        self.declare_parameter("priority_request_ttl_sec", 50.0)

        self.node_id = self.get_parameter("node_id").value

        # Preferisco i parametri nuovi, ma lascio compatibilità.
        self.green_duration = float(self.get_parameter("green_duration").value)
        self.yellow_duration = float(self.get_parameter("yellow_duration").value)
        self.all_red_duration = float(self.get_parameter("all_red_duration").value)
        self.min_green_duration = float(self.get_parameter("min_green_duration").value)
        self.max_green_duration = float(self.get_parameter("max_green_duration").value)

        # Se qualcuno usa ancora i vecchi parametri, restano sensati.
        old_base = float(self.get_parameter("base_phase_duration").value)
        old_min = float(self.get_parameter("min_phase_duration").value)
        old_max = float(self.get_parameter("max_phase_duration").value)

        if self.green_duration <= 0.0:
            self.green_duration = old_base
        if self.min_green_duration <= 0.0:
            self.min_green_duration = old_min
        if self.max_green_duration <= 0.0:
            self.max_green_duration = old_max

        self.priority_request_ttl_sec = float(
            self.get_parameter("priority_request_ttl_sec").value
        )

        self.nodes = {}
        self.edges = []
        self.connected_nodes = []
        self.group_a = []
        self.group_b = []

        self.load_map()
        self.build_phase_groups()

        self.current_phase = LightPhase.GREEN_A
        self.phase_started_at = time.time()
        self.priority_requests = []

        self.status_pub = self.create_publisher(String, "/traffic_light/status", 10)
        self.priority_sub = self.create_subscription(
            String,
            "/traffic_light/priority_request",
            self.on_priority_request,
            10
        )

        self.timer = self.create_timer(0.25, self.update)

        self.get_logger().info(
            f"traffic_light_manager avviato su nodo {self.node_id}, "
            f"grado={len(self.connected_nodes)}, "
            f"connected={self.connected_nodes}, "
            f"group_a={self.group_a}, group_b={self.group_b}"
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

        # Ordino geometricamente per avere fasi stabili e non dipendenti
        # dall'ordine casuale degli edge nel JSON.
        center = self.nodes[self.node_id]

        def angle_of(node_id):
            n = self.nodes[node_id]
            return math_atan2(n["y"] - center["y"], n["x"] - center["x"])

        connected = sorted(set(connected), key=angle_of)
        self.connected_nodes = connected

        degree = len(self.connected_nodes)

        if degree < 3:
            raise ValueError(
                f"Il nodo {self.node_id} ha grado {degree}: "
                f"non richiede semaforo secondo la logica attuale"
            )

        if degree > 4:
            raise ValueError(
                f"Il nodo {self.node_id} ha grado {degree}: "
                f"supporto previsto massimo 4 archi"
            )

    def build_phase_groups(self):
        degree = len(self.connected_nodes)

        if degree == 3:
            # Per un T-junction:
            # - gruppo A: due rami più opposti/continui;
            # - gruppo B: ramo laterale.
            a, b, side = self.choose_main_pair_for_degree_3()
            self.group_a = [a, b]
            self.group_b = [side]
            return

        if degree == 4:
            # I nodi sono ordinati ad angolo attorno all'incrocio.
            # I rami opposti sono 0-2 e 1-3.
            self.group_a = [self.connected_nodes[0], self.connected_nodes[2]]
            self.group_b = [self.connected_nodes[1], self.connected_nodes[3]]
            return

    def choose_main_pair_for_degree_3(self):
        center = self.nodes[self.node_id]
        best_pair = None
        best_score = -1.0

        nodes = self.connected_nodes

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                a = self.nodes[nodes[i]]
                b = self.nodes[nodes[j]]

                ax = a["x"] - center["x"]
                ay = a["y"] - center["y"]
                bx = b["x"] - center["x"]
                by = b["y"] - center["y"]

                la = max(0.000001, (ax * ax + ay * ay) ** 0.5)
                lb = max(0.000001, (bx * bx + by * by) ** 0.5)

                dot = (ax * bx + ay * by) / (la * lb)

                # Più è vicino a -1, più sono opposti.
                score = -dot

                if score > best_score:
                    best_score = score
                    best_pair = (nodes[i], nodes[j])

        a, b = best_pair
        side = [n for n in nodes if n not in best_pair][0]
        return a, b, side

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

        # Evito accumuli duplicati dello stesso veicolo/missione/movimento.
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
        self.priority_requests.append(request)

    # ------------------------------------------------------------------
    # UPDATE
    # ------------------------------------------------------------------

    def update(self):
        self.cleanup_old_requests()

        elapsed = time.time() - self.phase_started_at

        if self.current_phase == LightPhase.GREEN_A:
            if self.should_leave_green(LightPhase.GREEN_A, elapsed):
                self.set_phase(LightPhase.YELLOW_A)

        elif self.current_phase == LightPhase.YELLOW_A:
            if elapsed >= self.yellow_duration:
                self.set_phase(LightPhase.ALL_RED_A)

        elif self.current_phase == LightPhase.ALL_RED_A:
            if elapsed >= self.all_red_duration:
                self.set_phase(LightPhase.GREEN_B)

        elif self.current_phase == LightPhase.GREEN_B:
            if self.should_leave_green(LightPhase.GREEN_B, elapsed):
                self.set_phase(LightPhase.YELLOW_B)

        elif self.current_phase == LightPhase.YELLOW_B:
            if elapsed >= self.yellow_duration:
                self.set_phase(LightPhase.ALL_RED_B)

        elif self.current_phase == LightPhase.ALL_RED_B:
            if elapsed >= self.all_red_duration:
                self.set_phase(LightPhase.GREEN_A)

        self.publish_status()

    def cleanup_old_requests(self):
        now = time.time()

        self.priority_requests = [
            r for r in self.priority_requests
            if now - r["received_at"] <= self.priority_request_ttl_sec
        ]

    def should_leave_green(self, phase, elapsed):
        BASE_GREEN = 60.0
        PRIORITY_ADVANTAGE = 20.0

        # Nessuna priorità: verde fisso 60s.
        if not self.priority_requests:
            return elapsed >= BASE_GREEN

        # Controllo se l'altro gruppo ha richieste.
        other_phase = (
            LightPhase.GREEN_B
            if phase == LightPhase.GREEN_A
            else LightPhase.GREEN_A
        )

        current_score = 0
        other_score = 0

        for request in self.priority_requests:
            movement = {
                "from": request["from_node_id"],
                "to": request["to_node_id"]
            }

            priority = int(request.get("priority", 1))

            if self.movement_allowed_in_green_phase(movement, phase):
                current_score += priority

            if self.movement_allowed_in_green_phase(movement, other_phase):
                other_score += priority

        # L'altro gruppo ha richieste:
        # posso anticipare il cambio di massimo 20s.
        if other_score > current_score:
            return elapsed >= (BASE_GREEN - PRIORITY_ADVANTAGE)

        # Altrimenti verde normale.
        return elapsed >= BASE_GREEN

    def choose_green_phase(self):
        if not self.priority_requests:
            if self.current_phase in (LightPhase.GREEN_A, LightPhase.YELLOW_A, LightPhase.ALL_RED_A):
                return LightPhase.GREEN_B
            return LightPhase.GREEN_A

        score_a = 0
        score_b = 0

        for request in self.priority_requests:
            movement = {
                "from": request["from_node_id"],
                "to": request["to_node_id"]
            }
            priority = int(request.get("priority", 1))

            if self.movement_allowed_in_green_phase(movement, LightPhase.GREEN_A):
                score_a += priority

            if self.movement_allowed_in_green_phase(movement, LightPhase.GREEN_B):
                score_b += priority

        if score_a > score_b:
            return LightPhase.GREEN_A

        if score_b > score_a:
            return LightPhase.GREEN_B

        if self.current_phase in (LightPhase.GREEN_A, LightPhase.YELLOW_A, LightPhase.ALL_RED_A):
            return LightPhase.GREEN_A

        return LightPhase.GREEN_B

    def set_phase(self, phase):
        if phase == self.current_phase:
            return

        self.current_phase = phase
        self.phase_started_at = time.time()

    # ------------------------------------------------------------------
    # LOGICA MOVIMENTI
    # ------------------------------------------------------------------

    def get_allowed_movements(self):
        if self.current_phase not in (LightPhase.GREEN_A, LightPhase.GREEN_B):
            return []

        return self.get_allowed_movements_for_green_phase(self.current_phase)

    def get_allowed_movements_for_green_phase(self, phase):
        if phase == LightPhase.GREEN_A:
            green_group = self.group_a
            red_group = self.group_b
        elif phase == LightPhase.GREEN_B:
            green_group = self.group_b
            red_group = self.group_a
        else:
            return []

        movements = []

        # Movimenti dentro il gruppo verde.
        # Esempio 4 vie: nord <-> sud, est <-> ovest.
        for from_node in green_group:
            for to_node in green_group:
                if from_node != to_node:
                    movements.append({"from": from_node, "to": to_node})

        # Nel grado 3, quando il gruppo verde è il ramo laterale singolo,
        # consentiamo entrata/uscita verso i due rami principali.
        if len(self.connected_nodes) == 3 and len(green_group) == 1:
            side = green_group[0]
            for main in red_group:
                movements.append({"from": side, "to": main})
                movements.append({"from": main, "to": side})

        return self.unique_movements(movements)

    def unique_movements(self, movements):
        seen = set()
        result = []

        for m in movements:
            key = (m["from"], m["to"])
            if key in seen:
                continue
            seen.add(key)
            result.append(m)

        return result

    def movement_allowed_in_green_phase(self, movement, phase):
        allowed = self.get_allowed_movements_for_green_phase(phase)

        for item in allowed:
            if item["from"] == movement["from"] and item["to"] == movement["to"]:
                return True

        return False

    # ------------------------------------------------------------------
    # STATUS / VISUALIZZAZIONE
    # ------------------------------------------------------------------

    def phase_color(self):
        if self.current_phase in (LightPhase.GREEN_A, LightPhase.GREEN_B):
            return "green"

        if self.current_phase in (LightPhase.YELLOW_A, LightPhase.YELLOW_B):
            return "yellow"

        return "red"

    def get_signal_states(self):
        states = []

        if self.current_phase == LightPhase.GREEN_A:
            green_nodes = set(self.group_a)
            yellow_nodes = set()
        elif self.current_phase == LightPhase.GREEN_B:
            green_nodes = set(self.group_b)
            yellow_nodes = set()
        elif self.current_phase == LightPhase.YELLOW_A:
            green_nodes = set()
            yellow_nodes = set(self.group_a)
        elif self.current_phase == LightPhase.YELLOW_B:
            green_nodes = set()
            yellow_nodes = set(self.group_b)
        else:
            green_nodes = set()
            yellow_nodes = set()

        for from_node in self.connected_nodes:
            if from_node in green_nodes:
                color = "green"
            elif from_node in yellow_nodes:
                color = "yellow"
            else:
                color = "red"

            states.append({
                "from_node_id": from_node,
                "node_id": self.node_id,
                "color": color
            })

        return states

    def publish_status(self):
        payload = {
            "node_id": self.node_id,
            "phase": self.current_phase.value,
            "color": self.phase_color(),
            "degree": len(self.connected_nodes),
            "connected_nodes": self.connected_nodes,
            "group_a": self.group_a,
            "group_b": self.group_b,
            "allowed_movements": self.get_allowed_movements(),
            "signal_states": self.get_signal_states(),
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


# Evito import math completo solo per non cambiare troppo stile del file.
def math_atan2(y, x):
    import math
    return math.atan2(y, x)


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
