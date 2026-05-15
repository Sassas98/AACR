import json
import math
import os
import heapq
from enum import Enum
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse

from geometry_msgs.msg import Twist
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from smart_city_interfaces.action import NavigateToPose
from geometry_msgs.msg import PoseStamped


class ExecutorState(Enum):
    IDLE = "IDLE"
    NAVIGATING = "NAVIGATING"
    WAITING_TRAFFIC_LIGHT = "WAITING_TRAFFIC_LIGHT"
    OBSTACLE_STOP = "OBSTACLE_STOP"


class NavigationExecutor(Node):

    def __init__(self):
        super().__init__("navigation_executor")

        # ------------------------------------------------------------
        # PARAMETRI
        # ------------------------------------------------------------
        self.declare_parameter("vehicle_id", "vehicle")
        self.declare_parameter("map_config_file", "config/city_map.json")
        self.declare_parameter("pose_entity_name", "")

        self.declare_parameter("default_max_speed", 1.2)
        self.declare_parameter("linear_k", 0.9)
        self.declare_parameter("angular_k", 0.7)
        self.declare_parameter("max_angular_speed", 0.25)

        self.declare_parameter("waypoint_tolerance", 0.30)
        self.declare_parameter("target_tolerance", 0.45)
        self.declare_parameter("lane_offset_ratio", 0.5)

        self.declare_parameter("intersection_clearance", 2.2)
        self.declare_parameter("diagnostic_log_period_sec", 1.0)
        self.declare_parameter("path_log_enabled", True)

        # Semafori
        self.declare_parameter("traffic_light_stop_distance", 1.5)

        # LiDAR anticollisione
        self.declare_parameter("obstacle_stop_distance", 0.8)
        self.declare_parameter("obstacle_slow_distance", 1.8)
        self.declare_parameter("obstacle_fov_deg", 60.0)

        self.vehicle_id = self.get_parameter("vehicle_id").value
        self.map_config_file = self.get_parameter("map_config_file").value
        self.pose_entity_name = self.get_parameter("pose_entity_name").value or self.vehicle_id

        self.default_max_speed = float(self.get_parameter("default_max_speed").value)
        self.linear_k = float(self.get_parameter("linear_k").value)
        self.angular_k = float(self.get_parameter("angular_k").value)
        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)

        self.waypoint_tolerance = float(self.get_parameter("waypoint_tolerance").value)
        self.target_tolerance = float(self.get_parameter("target_tolerance").value)
        self.lane_offset_ratio = float(self.get_parameter("lane_offset_ratio").value)

        self.intersection_clearance = float(self.get_parameter("intersection_clearance").value)
        self.diagnostic_log_period_sec = float(self.get_parameter("diagnostic_log_period_sec").value)
        self.path_log_enabled = bool(self.get_parameter("path_log_enabled").value)

        self.traffic_light_stop_distance = float(self.get_parameter("traffic_light_stop_distance").value)
        self.obstacle_stop_distance = float(self.get_parameter("obstacle_stop_distance").value)
        self.obstacle_slow_distance = float(self.get_parameter("obstacle_slow_distance").value)
        self.obstacle_fov_deg = float(self.get_parameter("obstacle_fov_deg").value)

        self.last_priority_request_time = {}
        # ------------------------------------------------------------
        # STATO
        # ------------------------------------------------------------
        self.state = ExecutorState.IDLE

        # True quando ho ricevuto almeno una posa reale del modello da Gazebo.
        # Tengo il nome has_odom solo per non riscrivere tutta la logica sotto.
        self.has_odom = False
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        # LiDAR
        self.obstacle_min_distance = float("inf")   # distanza minima ostacolo nel FOV frontale
        self.last_scan_stamp = None

        # Semafori: dizionario node_id -> status dict
        self.traffic_light_statuses = {}

        self.nodes = {}
        self.edges = []
        self.edge_by_id = {}
        self.adj = {}
        self.lane_width = 1.2
        self.default_map_speed_limit = 1.4

        self.current_path = []
        self.current_waypoint_index = 0
        self.current_mission_id = ""

        self.last_diag_time = self.get_clock().now()

        self.load_map()

        # ------------------------------------------------------------
        # TOPIC
        # ------------------------------------------------------------
        self.callback_group = ReentrantCallbackGroup()

        self.cmd_vel_pub = self.create_publisher(Twist, "cmd_vel", 10)

        self.pose_sub = self.create_subscription(
            PoseStamped,
            f"/gazebo/model_pose/{self.vehicle_id}",
            self.on_world_pose,
            10,
            callback_group=self.callback_group
        )
        self.scan_sub = self.create_subscription(
            LaserScan,
            "scan",
            self.on_scan,
            10,
            callback_group=self.callback_group
        )

        self.traffic_light_sub = self.create_subscription(
            String,
            "/traffic_light/status",
            self.on_traffic_light_status,
            10,
            callback_group=self.callback_group
        )

        self.priority_pub = self.create_publisher(
            String,
            "/traffic_light/priority_request",
            10
        )

        self.action_server = ActionServer(
            self,
            NavigateToPose,
            "navigation_executor/navigate_to_pose",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group
        )

    # ------------------------------------------------------------------
    # MAPPA
    # ------------------------------------------------------------------

    def load_map(self):
        path = self.map_config_file

        if not os.path.isabs(path):
            path = os.path.join(os.getcwd(), path)

        if not os.path.exists(path):
            raise FileNotFoundError(f"File mappa non trovato: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.lane_width = float(data.get("lane_width", 1.2))
        self.default_map_speed_limit = float(data.get("default_speed_limit", 1.4))

        for node in data["nodes"]:
            self.nodes[node["id"]] = {
                "id": node["id"],
                "x": float(node["x"]),
                "y": float(node["y"])
            }

        for edge in data["edges"]:
            edge_id = edge["id"]
            from_id = edge["from"]
            to_id = edge["to"]

            if from_id not in self.nodes or to_id not in self.nodes:
                raise RuntimeError(
                    f"Edge {edge_id} usa nodi non esistenti: {from_id}->{to_id}"
                )

            a = self.nodes[from_id]
            b = self.nodes[to_id]

            length = self.distance_xy(a["x"], a["y"], b["x"], b["y"])

            e = {
                "id": edge_id,
                "from": from_id,
                "to": to_id,
                "speed_limit": float(edge.get("speed_limit", self.default_map_speed_limit)),
                "length": length
            }

            self.edges.append(e)
            self.edge_by_id[edge_id] = e

            self.adj.setdefault(from_id, [])
            self.adj.setdefault(to_id, [])

            self.adj[from_id].append((to_id, edge_id, length))
            self.adj[to_id].append((from_id, edge_id, length))

    # ------------------------------------------------------------------
    # CALLBACK SENSORI
    # ------------------------------------------------------------------

    def on_world_pose(self, msg):
        p = msg.pose.position
        q = msg.pose.orientation

        self.current_x = p.x
        self.current_y = p.y

        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)

        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

        self.has_odom = True
        self.has_world_pose = True

    def _is_pose_for_this_vehicle(self, frame_name, wanted):
        if not frame_name or not wanted:
            return False

        f = frame_name.strip("/")
        w = wanted.strip("/")

        if f == w:
            return True

        parts = f.split("/")

        if w in parts:
            return True

        if parts and parts[-1] == w:
            return True

        # Caso abbastanza comune: child_frame_id tipo "taxi_1/base_link".
        if parts and parts[0] == w:
            return True

        return False

    def on_scan(self, msg):
        """
        Analizza il LiDAR tenendo solo i raggi nel cono frontale
        (±obstacle_fov_deg/2 rispetto al fronte del veicolo).
        Salva la distanza minima trovata in self.obstacle_min_distance.
        """
        self.last_scan_stamp = msg.header.stamp

        fov_rad = math.radians(self.obstacle_fov_deg / 2.0)

        min_dist = float("inf")

        angle = msg.angle_min

        for r in msg.ranges:
            if msg.range_min <= r <= msg.range_max:
                # Normalizza l'angolo del raggio nel frame del veicolo.
                # Il LiDAR pubblica angoli relativi al sensore: 0 = fronte.
                normalized = self.normalize_angle(angle)

                if abs(normalized) <= fov_rad:
                    if r < min_dist:
                        min_dist = r

            angle += msg.angle_increment

        self.obstacle_min_distance = min_dist

    def on_traffic_light_status(self, msg):
        try:
            status = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        node_id = status.get("node_id")

        if node_id:
            self.traffic_light_statuses[node_id] = status

    # ------------------------------------------------------------------
    # SEMAFORI
    # ------------------------------------------------------------------

    def publish_priority_request(self, from_node_id, to_node_id, intersection_node_id, mission_id):
        payload = {
            "vehicle_id": self.vehicle_id,
            "mission_id": mission_id,
            "node_id": intersection_node_id,
            "from_node_id": from_node_id,
            "to_node_id": to_node_id,
            "priority": 1
        }

        msg = String()
        msg.data = json.dumps(payload)
        self.priority_pub.publish(msg)
        self.log_navigation_event(
            f"chiedo priorità al semaforo {intersection_node_id}: movimento {from_node_id}->{intersection_node_id}->{to_node_id}"
        )

    def is_movement_allowed(self, from_node_id, to_node_id, intersection_node_id):
        """
        Controlla se il TrafficLightManager dell'incrocio consente
        il movimento from_node_id -> intersection_node_id -> to_node_id.
        Se non abbiamo ancora ricevuto lo status del semaforo, lasciamo passare
        (fail-open: meglio rischiare una piccola sovrapposizione che bloccarsi).
        """
        status = self.traffic_light_statuses.get(intersection_node_id)

        if status is None:
            return True

        allowed = status.get("allowed_movements", [])

        for m in allowed:
            if m["from"] == from_node_id and m["to"] == to_node_id:
                return True

        return False

    def distance_to_node(self, node_id):
        node = self.nodes.get(node_id)

        if node is None:
            return float("inf")

        return self.distance_xy(self.current_x, self.current_y, node["x"], node["y"])

    # ------------------------------------------------------------------
    # OSTACOLI
    # ------------------------------------------------------------------

    def get_obstacle_speed_factor(self):
        """
        Restituisce un fattore [0.0, 1.0] da applicare alla velocità lineare.
        0.0  → stop completo (ostacolo entro stop_distance)
        0.0..1.0 → rallentamento lineare tra slow_distance e stop_distance
        1.0  → nessun ostacolo rilevante
        """
        d = self.obstacle_min_distance

        if d <= self.obstacle_stop_distance:
            return 0.0

        if d <= self.obstacle_slow_distance:
            # Interpolazione lineare: più vicino = più lento.
            ratio = (d - self.obstacle_stop_distance) / (
                self.obstacle_slow_distance - self.obstacle_stop_distance
            )
            return max(0.0, min(1.0, ratio))

        return 1.0

    # ------------------------------------------------------------------
    # ACTION
    # ------------------------------------------------------------------

    def goal_callback(self, goal_request):

        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        goal = goal_handle.request
        self.current_mission_id = goal.mission_id

        if not self.has_odom:
            self.stop_vehicle()
            goal_handle.abort()
            result = NavigateToPose.Result()
            result.success = False
            result.message = "Posa reale Gazebo non ancora disponibile"
            return result

        self.state = ExecutorState.NAVIGATING
        self.log_navigation_event(
            f"parto da ({self.current_x:.2f},{self.current_y:.2f}) verso target ({goal.target_x:.2f},{goal.target_y:.2f})"
        )

        try:
            self.current_path, self.node_path = self.build_navigation_path(
                self.current_x, self.current_y,
                goal.target_x, goal.target_y
            )
            self.current_waypoint_index = 0

            if not self.current_path:
                raise RuntimeError("path vuoto")

            self.log_path(self.current_path)

        except Exception as ex:
            self.stop_vehicle()
            self.state = ExecutorState.IDLE
            goal_handle.abort()
            result = NavigateToPose.Result()
            result.success = False
            result.message = f"Errore calcolo path: {ex}"
            return result

        rate = self.create_rate(20)
        waypoint_start_time = self.get_clock().now()

        def compute_timeout(wp):
            """
            Timeout dinamico: almeno 8s, ma proporzionale alla distanza
            dal waypoint al momento in cui iniziamo a lavorarci.
            Usa velocità minima 0.5 m/s come riferimento pessimistico.
            """
            dx = wp["x"] - self.current_x
            dy = wp["y"] - self.current_y
            dist = math.sqrt(dx * dx + dy * dy)
            return max(8.0, dist / 0.5)

        waypoint_timeout_sec = compute_timeout(
            self.current_path[self.current_waypoint_index]
        )

        while rclpy.ok():

            # ── Cancellazione ──────────────────────────────────────────
            if goal_handle.is_cancel_requested:
                self.stop_vehicle()
                self.state = ExecutorState.IDLE
                goal_handle.canceled()
                result = NavigateToPose.Result()
                result.success = False
                result.message = "Navigazione cancellata"
                return result

            # ── Target raggiunto ───────────────────────────────────────
            if self.current_waypoint_index >= len(self.current_path):
                self.stop_vehicle()
                self.state = ExecutorState.IDLE
                goal_handle.succeed()
                result = NavigateToPose.Result()
                result.success = True
                result.message = "Target raggiunto"
                self.log_navigation_snapshot("target raggiunto", None, 0.0, force=True)
                return result

            current_wp = self.current_path[self.current_waypoint_index]

            dx = current_wp["x"] - self.current_x
            dy = current_wp["y"] - self.current_y
            distance = math.sqrt(dx * dx + dy * dy)

            is_last = self.current_waypoint_index == len(self.current_path) - 1
            tolerance = self.target_tolerance if is_last else self.waypoint_tolerance

            # ── Waypoint raggiunto ─────────────────────────────────────
            if distance <= tolerance:
                self.log_navigation_snapshot("waypoint raggiunto", current_wp, distance, force=True)
                self.current_waypoint_index += 1
                waypoint_start_time = self.get_clock().now()
                if self.current_waypoint_index < len(self.current_path):
                    waypoint_timeout_sec = compute_timeout(
                        self.current_path[self.current_waypoint_index]
                    )
                rate.sleep()
                continue

            # ── Timeout waypoint ───────────────────────────────────────
            elapsed_on_wp = (
                self.get_clock().now() - waypoint_start_time
            ).nanoseconds / 1e9

            if elapsed_on_wp > waypoint_timeout_sec:
                self.log_navigation_snapshot("timeout waypoint, passo al prossimo", current_wp, distance, force=True)
                self.current_waypoint_index += 1
                waypoint_start_time = self.get_clock().now()
                if self.current_waypoint_index < len(self.current_path):
                    waypoint_timeout_sec = compute_timeout(
                        self.current_path[self.current_waypoint_index]
                    )
                rate.sleep()
                continue

            # ── Controllo semaforo ─────────────────────────────────────
            if current_wp.get("kind") == "approach_intersection":
                intersection_node_id = current_wp.get("node_id")
                from_node_id, to_node_id = self.get_movement_for_intersection(
                    intersection_node_id
                )

                if intersection_node_id and from_node_id:
                    if distance <= self.traffic_light_stop_distance * 3:
                        last_sent = self.last_priority_request_time.get(intersection_node_id, 0.0)
                        now_sec = self.get_clock().now().nanoseconds / 1e9
                        if now_sec - last_sent >= 2.0:
                            self.publish_priority_request(
                                from_node_id,
                                to_node_id,
                                intersection_node_id,
                                goal.mission_id
                            )
                            self.last_priority_request_time[intersection_node_id] = now_sec

                    if distance <= self.traffic_light_stop_distance:
                        allowed = self.is_movement_allowed(
                            from_node_id, to_node_id, intersection_node_id
                        )

                        if not allowed:
                            if self.state != ExecutorState.WAITING_TRAFFIC_LIGHT:
                                self.state = ExecutorState.WAITING_TRAFFIC_LIGHT
                                self.log_navigation_snapshot(
                                    f"fermo al semaforo {intersection_node_id}: movimento {from_node_id}->{intersection_node_id}->{to_node_id} non consentito",
                                    current_wp, distance, force=True
                                )

                            self.stop_vehicle()
                            waypoint_start_time = self.get_clock().now()

                            feedback = self._make_feedback(goal, current_wp)
                            goal_handle.publish_feedback(feedback)

                            rate.sleep()
                            continue

                        if self.state == ExecutorState.WAITING_TRAFFIC_LIGHT:
                            self.state = ExecutorState.NAVIGATING
                            self.log_navigation_snapshot(
                                f"semaforo {intersection_node_id} verde: riparto",
                                current_wp, distance, force=True
                            )

            # ── Controllo ostacolo LiDAR ───────────────────────────────
            obstacle_factor = self.get_obstacle_speed_factor()

            if obstacle_factor == 0.0:
                if self.state != ExecutorState.OBSTACLE_STOP:
                    self.state = ExecutorState.OBSTACLE_STOP
                    self.log_navigation_snapshot(
                        f"ostacolo davanti a {self.obstacle_min_distance:.2f} m: stop",
                        current_wp, distance, force=True
                    )

                self.stop_vehicle()
                waypoint_start_time = self.get_clock().now()

                feedback = self._make_feedback(goal, current_wp)
                goal_handle.publish_feedback(feedback)

                rate.sleep()
                continue

            if self.state == ExecutorState.OBSTACLE_STOP:
                self.state = ExecutorState.NAVIGATING
                self.log_navigation_snapshot("via libera: riprendo navigazione", current_wp, distance, force=True)

            # ── Movimento ─────────────────────────────────────────────
            self.move_towards_waypoint(current_wp, goal.max_speed, obstacle_factor)

            feedback = self._make_feedback(goal, current_wp)
            goal_handle.publish_feedback(feedback)

            self.maybe_log_diagnostics(current_wp, distance)

            rate.sleep()

        self.stop_vehicle()
        self.state = ExecutorState.IDLE
        goal_handle.abort()

        result = NavigateToPose.Result()
        result.success = False
        result.message = "Navigazione interrotta"
        return result

    def _make_feedback(self, goal, current_wp):
        feedback = NavigateToPose.Feedback()
        feedback.current_x = float(self.current_x)
        feedback.current_y = float(self.current_y)
        feedback.distance_remaining = float(self.compute_remaining_distance())
        feedback.status = (
            f"mission={goal.mission_id} "
            f"wp={self.current_waypoint_index + 1}/{len(self.current_path)} "
            f"state={self.state.value}"
        )
        return feedback

    def get_movement_for_intersection(self, intersection_node_id):
        """
        Dato il node_id dell'incrocio in avvicinamento, risale al nodo
        precedente (from) e successivo (to) nel node_path pianificato.
        Restituisce (from_node_id, to_node_id).
        """
        if not hasattr(self, "node_path") or not self.node_path:
            return None, None

        try:
            idx = self.node_path.index(intersection_node_id)
        except ValueError:
            return None, None

        from_node_id = self.node_path[idx - 1] if idx > 0 else None
        to_node_id = self.node_path[idx + 1] if idx < len(self.node_path) - 1 else None

        return from_node_id, to_node_id

    # ------------------------------------------------------------------
    # PATH PLANNING
    # ------------------------------------------------------------------

    def build_navigation_path(self, start_x, start_y, target_x, target_y):
        """
        Restituisce (waypoints, node_path).
        node_path è la sequenza di nodi del grafo attraversati,
        necessaria per risalire ai movimenti agli incroci.
        """
        start_projection = self.find_nearest_edge_projection(start_x, start_y)
        target_projection = self.find_nearest_edge_projection(target_x, target_y)

        start_edge = start_projection["edge"]
        target_edge = target_projection["edge"]

        start_candidates = [start_edge["from"], start_edge["to"]]
        target_candidates = [target_edge["from"], target_edge["to"]]

        best_node_path = None
        best_cost = float("inf")

        for s in start_candidates:
            for t in target_candidates:
                node_path, cost = self.shortest_path(s, t)

                if node_path is None:
                    continue

                total_cost = (
                    cost
                    + self.distance_to_node_on_edge(start_projection, s)
                    + self.distance_to_node_on_edge(target_projection, t)
                )

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_node_path = node_path

        if not best_node_path:
            raise RuntimeError("nessun path trovato sul grafo")

        waypoints = []

        start_lane = self.project_point_to_lane(
            start_projection, best_node_path[0]
        )
        waypoints.append({
            "x": start_lane["x"],
            "y": start_lane["y"],
            "edge_id": start_edge["id"],
            "node_id": None,
            "kind": "start_projection",
            "speed_limit": start_edge["speed_limit"]
        })

        for i in range(len(best_node_path) - 1):
            current_node_id = best_node_path[i]
            next_node_id = best_node_path[i + 1]

            edge = self.get_edge_between(current_node_id, next_node_id)

            exit_from_current = self.node_to_lane_point(
                current_node_id, next_node_id, mode="exit"
            )
            waypoints.append({
                "x": exit_from_current["x"],
                "y": exit_from_current["y"],
                "edge_id": edge["id"],
                "node_id": current_node_id,
                "kind": "exit_intersection",
                "speed_limit": edge["speed_limit"]
            })

            approach_next = self.node_to_lane_point(
                next_node_id, current_node_id, mode="approach"
            )
            waypoints.append({
                "x": approach_next["x"],
                "y": approach_next["y"],
                "edge_id": edge["id"],
                "node_id": next_node_id,
                "kind": "approach_intersection",
                "speed_limit": edge["speed_limit"]
            })

        target_lane = self.project_point_to_lane(
            target_projection, best_node_path[-1]
        )
        waypoints.append({
            "x": target_lane["x"],
            "y": target_lane["y"],
            "edge_id": target_edge["id"],
            "node_id": None,
            "kind": "target_projection",
            "speed_limit": target_edge["speed_limit"]
        })

        waypoints = self.simplify_waypoints(waypoints)

        return waypoints, best_node_path

    def shortest_path(self, start_node_id, target_node_id):
        queue = [(0.0, start_node_id, [])]
        visited = set()

        while queue:
            cost, node_id, path = heapq.heappop(queue)

            if node_id in visited:
                continue

            visited.add(node_id)

            new_path = path + [node_id]

            if node_id == target_node_id:
                return new_path, cost

            for neighbor_id, edge_id, length in self.adj.get(node_id, []):
                if neighbor_id not in visited:
                    heapq.heappush(queue, (cost + length, neighbor_id, new_path))

        return None, float("inf")

    def find_nearest_edge_projection(self, x, y):
        best = None
        best_distance = float("inf")

        for edge in self.edges:
            a = self.nodes[edge["from"]]
            b = self.nodes[edge["to"]]

            proj = self.project_point_on_segment(
                x, y, a["x"], a["y"], b["x"], b["y"]
            )

            d = self.distance_xy(x, y, proj["x"], proj["y"])

            if d < best_distance:
                best_distance = d
                best = {
                    "edge": edge,
                    "x": proj["x"],
                    "y": proj["y"],
                    "t": proj["t"],
                    "distance": d
                }

        if best is None:
            raise RuntimeError("impossibile proiettare su una strada")

        return best

    def project_point_on_segment(self, px, py, ax, ay, bx, by):
        dx = bx - ax
        dy = by - ay

        denom = dx * dx + dy * dy

        if denom <= 0.000001:
            return {"x": ax, "y": ay, "t": 0.0}

        t = ((px - ax) * dx + (py - ay) * dy) / denom
        t = max(0.0, min(1.0, t))

        return {"x": ax + t * dx, "y": ay + t * dy, "t": t}

    def project_point_to_lane(self, projection, destination_node_id):
        edge = projection["edge"]

        a = self.nodes[edge["from"]]
        b = self.nodes[edge["to"]]

        if destination_node_id == edge["to"]:
            from_node, to_node = a, b
        else:
            from_node, to_node = b, a

        lane_x, lane_y = self.apply_right_lane_offset(
            projection["x"], projection["y"],
            from_node["x"], from_node["y"],
            to_node["x"], to_node["y"]
        )

        return {"x": lane_x, "y": lane_y}

    def node_to_lane_point(self, node_id, other_node_id, mode):
        node = self.nodes[node_id]
        other = self.nodes[other_node_id]

        dx = other["x"] - node["x"]
        dy = other["y"] - node["y"]

        length = math.sqrt(dx * dx + dy * dy)

        if length <= 0.000001:
            return {"x": node["x"], "y": node["y"]}

        ux = dx / length
        uy = dy / length

        c = self.intersection_clearance

        if mode == "exit":
            # Appena usciti dall'incrocio node verso other_node:
            # ci spostiamo AVANTI di c lungo la direzione node->other.
            base_x = node["x"] + ux * c
            base_y = node["y"] + uy * c

            lane_from_x, lane_from_y = node["x"], node["y"]
            lane_to_x, lane_to_y = other["x"], other["y"]

        elif mode == "approach":
            # In avvicinamento al nodo node provenendo da other_node:
            # la direzione di marcia è other -> node (ux,uy punta da node verso other,
            # quindi la direzione opposta è -ux, -uy).
            # Ci posizioniamo a distanza c dal nodo, sul lato da cui arriviamo.
            base_x = node["x"] - ux * c   # FIX: era +ux*c (direzione sbagliata)
            base_y = node["y"] - uy * c

            # La direzione di marcia reale è other -> node.
            lane_from_x, lane_from_y = other["x"], other["y"]
            lane_to_x, lane_to_y = node["x"], node["y"]

        else:
            raise RuntimeError(f"mode non valido: {mode}")

        lane_x, lane_y = self.apply_right_lane_offset(
            base_x, base_y,
            lane_from_x, lane_from_y,
            lane_to_x, lane_to_y
        )

        return {"x": lane_x, "y": lane_y}

    def apply_right_lane_offset(self, x, y, from_x, from_y, to_x, to_y):
        dx = to_x - from_x
        dy = to_y - from_y

        length = math.sqrt(dx * dx + dy * dy)

        if length <= 0.000001:
            return x, y

        ux = dx / length
        uy = dy / length

        # Vettore perpendicolare destra (rotazione +90°: right = (uy, -ux)).
        right_x = uy
        right_y = -ux

        offset = self.lane_width * self.lane_offset_ratio

        return x + right_x * offset, y + right_y * offset

    def distance_to_node_on_edge(self, projection, node_id):
        node = self.nodes[node_id]
        return self.distance_xy(
            projection["x"], projection["y"],
            node["x"], node["y"]
        )

    def get_edge_between(self, a, b):
        for edge in self.edges:
            if (edge["from"] == a and edge["to"] == b) or \
               (edge["from"] == b and edge["to"] == a):
                return edge

        raise RuntimeError(f"nessun edge tra {a} e {b}")

    def simplify_waypoints(self, waypoints):
        if not waypoints:
            return []

        result = [waypoints[0]]

        for wp in waypoints[1:]:
            last = result[-1]
            d = self.distance_xy(last["x"], last["y"], wp["x"], wp["y"])

            if d >= 0.35:
                result.append(wp)
            else:
                pass

        return result

    # ------------------------------------------------------------------
    # CONTROLLO MOVIMENTO
    # ------------------------------------------------------------------

    def move_towards_waypoint(self, waypoint, requested_max_speed, obstacle_factor=1.0):
        dx = waypoint["x"] - self.current_x
        dy = waypoint["y"] - self.current_y

        target_angle = math.atan2(dy, dx)
        angle_error = self.normalize_angle(target_angle - self.current_yaw)

        distance = math.sqrt(dx * dx + dy * dy)

        edge_speed_limit = float(
            waypoint.get("speed_limit", self.default_map_speed_limit)
        )

        max_speed = float(requested_max_speed) if float(requested_max_speed) > 0.0 else self.default_max_speed
        max_speed = min(max_speed, edge_speed_limit)

        angular_speed = self.angular_k * angle_error
        angular_speed = max(-self.max_angular_speed, min(self.max_angular_speed, angular_speed))

        abs_error = abs(angle_error)

        if abs_error > 0.60:
            linear_speed = 0.0
            motion_mode = "ROTATE_IN_PLACE"
        else:
            linear_speed = min(max_speed, self.linear_k * distance)
            motion_mode = "FORWARD"

        if distance < 0.8:
            linear_speed = min(linear_speed, 0.35)
        if distance < 0.4:
            linear_speed = min(linear_speed, 0.20)

        # Applica il fattore ostacolo.
        linear_speed *= obstacle_factor

        cmd = Twist()
        cmd.linear.x = float(linear_speed)
        cmd.angular.z = float(angular_speed)

        self.cmd_vel_pub.publish(cmd)

        waypoint["_last_control"] = {
            "target_angle": target_angle,
            "angle_error": angle_error,
            "distance": distance,
            "linear_speed": linear_speed,
            "angular_speed": angular_speed,
            "motion_mode": motion_mode,
            "max_speed": max_speed,
            "obstacle_factor": obstacle_factor
        }

    def stop_vehicle(self):
        self.cmd_vel_pub.publish(Twist())

    # ------------------------------------------------------------------
    # LOG DI NAVIGAZIONE / GRAFO
    # ------------------------------------------------------------------

    def log_navigation_event(self, message):
        self.get_logger().info(
            f"[NAV] vehicle={self.vehicle_id} | mission={self.current_mission_id} | {message}"
        )

    def describe_graph_position(self):
        if not self.edges:
            return "grafo non disponibile"

        projection = self.find_nearest_edge_projection(self.current_x, self.current_y)
        edge = projection["edge"]
        from_node_id = edge["from"]
        to_node_id = edge["to"]

        from_node = self.nodes[from_node_id]
        to_node = self.nodes[to_node_id]

        dist_from = self.distance_xy(self.current_x, self.current_y, from_node["x"], from_node["y"])
        dist_to = self.distance_xy(self.current_x, self.current_y, to_node["x"], to_node["y"])
        nearest_node = from_node_id if dist_from <= dist_to else to_node_id
        nearest_dist = min(dist_from, dist_to)

        return (
            f"sono a ({self.current_x:.2f},{self.current_y:.2f}), yaw={self.current_yaw:.2f}; "
            f"nel grafo sono sulla strada/edge {edge['id']} tra {from_node_id} e {to_node_id}; "
            f"proiezione=({projection['x']:.2f},{projection['y']:.2f}), "
            f"t={projection['t']:.2f}, fuori strada={projection['distance']:.2f} m; "
            f"nodo più vicino={nearest_node} ({nearest_dist:.2f} m)"
        )

    def describe_waypoint(self, waypoint, distance):
        if waypoint is None:
            return "nessun waypoint attivo"

        edge_id = waypoint.get("edge_id")
        edge = self.edge_by_id.get(edge_id)

        if edge:
            street = f"strada/edge {edge['id']} tra {edge['from']} e {edge['to']}"
        else:
            street = f"edge={edge_id}"

        node_id = waypoint.get("node_id") or "-"
        kind = waypoint.get("kind") or "-"

        return (
            f"sto puntando wp {self.current_waypoint_index + 1}/{len(self.current_path)} "
            f"[{kind}] a ({waypoint['x']:.2f},{waypoint['y']:.2f}), "
            f"dist={distance:.2f} m, node={node_id}, {street}"
        )

    def log_path(self, path):
        if not hasattr(self, "node_path") or not self.node_path:
            self.log_navigation_event(f"path calcolato con {len(path)} waypoint, ma senza node_path")
            return

        self.log_navigation_event(
            "percorso grafo scelto: " + " -> ".join(self.node_path)
        )

        for i in range(len(self.node_path) - 1):
            a = self.node_path[i]
            b = self.node_path[i + 1]
            edge = self.get_edge_between(a, b)
            self.log_navigation_event(
                f"tratto {i + 1}: giro/percorro la via tra nodo {a} e nodo {b} "
                f"sull'edge {edge['id']} lungo {edge['length']:.2f} m"
            )

        self.log_navigation_event(f"waypoint generati: {len(path)}")

    def log_navigation_snapshot(self, reason, waypoint=None, distance=None, force=False):
        now = self.get_clock().now()
        elapsed = (now - self.last_diag_time).nanoseconds / 1e9

        if not force and elapsed < self.diagnostic_log_period_sec:
            return

        self.last_diag_time = now

        if waypoint is None and self.current_waypoint_index < len(self.current_path):
            waypoint = self.current_path[self.current_waypoint_index]

        if distance is None and waypoint is not None:
            distance = self.distance_xy(self.current_x, self.current_y, waypoint["x"], waypoint["y"])
        elif distance is None:
            distance = 0.0

        ctrl = waypoint.get("_last_control", {}) if waypoint else {}

        self.log_navigation_event(
            f"{reason} | stato={self.state.value} | "
            f"{self.describe_graph_position()} | "
            f"{self.describe_waypoint(waypoint, distance)} | "
            f"cmd: mode={ctrl.get('motion_mode', '?')}, "
            f"v={ctrl.get('linear_speed', 0.0):.2f}, "
            f"w={ctrl.get('angular_speed', 0.0):.2f}"
        )

    def maybe_log_diagnostics(self, waypoint, distance):
        self.log_navigation_snapshot("navigazione in corso", waypoint, distance)

    # ------------------------------------------------------------------
    # UTILITY
    # ------------------------------------------------------------------

    def compute_remaining_distance(self):
        if not self.current_path:
            return 0.0

        if self.current_waypoint_index >= len(self.current_path):
            return 0.0

        total = 0.0
        current = {"x": self.current_x, "y": self.current_y}

        for i in range(self.current_waypoint_index, len(self.current_path)):
            wp = self.current_path[i]
            total += self.distance_xy(current["x"], current["y"], wp["x"], wp["y"])
            current = wp

        return total

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def distance_xy(self, x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        return math.sqrt(dx * dx + dy * dy)


def main(args=None):
    rclpy.init(args=args)

    node = NavigationExecutor()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    node.stop_vehicle()
    executor.shutdown()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
