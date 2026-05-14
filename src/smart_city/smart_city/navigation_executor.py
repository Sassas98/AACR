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
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from smart_city_interfaces.action import NavigateToPose


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

        self.declare_parameter("initial_x", 0.0)
        self.declare_parameter("initial_y", 0.0)
        self.declare_parameter("initial_yaw", 0.0)

        self.initial_x = float(self.get_parameter("initial_x").value)
        self.initial_y = float(self.get_parameter("initial_y").value)
        self.initial_yaw = float(self.get_parameter("initial_yaw").value)

        # ------------------------------------------------------------
        # STATO
        # ------------------------------------------------------------
        self.state = ExecutorState.IDLE

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

        self.odom_sub = self.create_subscription(
            Odometry,
            "odom",
            self.on_odom,
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

        self.get_logger().info(
            f"[INIT] NavigationExecutor avviato | "
            f"vehicle_id={self.vehicle_id} | "
            f"namespace={self.get_namespace()} | "
            f"map={self.map_config_file}"
        )

    # ------------------------------------------------------------------
    # MAPPA
    # ------------------------------------------------------------------

    def load_map(self):
        path = self.map_config_file

        if not os.path.isabs(path):
            path = os.path.join(os.getcwd(), path)

        self.get_logger().info(f"[MAP] Caricamento mappa da: {path}")

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

        self.get_logger().info(
            f"[MAP] Mappa caricata | "
            f"nodes={len(self.nodes)} | edges={len(self.edges)} | "
            f"lane_width={self.lane_width:.2f} | "
            f"default_speed_limit={self.default_map_speed_limit:.2f}"
        )

    # ------------------------------------------------------------------
    # CALLBACK SENSORI
    # ------------------------------------------------------------------

    def on_odom(self, msg):
        local_x = msg.pose.pose.position.x
        local_y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        local_yaw = math.atan2(siny_cosp, cosy_cosp)

        # Ruota il delta locale nel frame mondo usando lo yaw iniziale
        cos0 = math.cos(self.initial_yaw)
        sin0 = math.sin(self.initial_yaw)

        self.current_x = self.initial_x + cos0 * local_x - sin0 * local_y
        self.current_y = self.initial_y + sin0 * local_x + cos0 * local_y
        self.current_yaw = self.normalize_angle(self.initial_yaw + local_yaw)

        self.has_odom = True

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

        self.get_logger().info(
            f"[TL] Priority request inviata | "
            f"intersection={intersection_node_id} | "
            f"{from_node_id}->{intersection_node_id}->{to_node_id}"
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
        self.get_logger().info(
            f"[GOAL] Ricevuto goal | "
            f"vehicle={self.vehicle_id} | "
            f"mission={goal_request.mission_id} | "
            f"target=({goal_request.target_x:.2f}, {goal_request.target_y:.2f}) | "
            f"max_speed={goal_request.max_speed:.2f}"
        )

        if not self.has_odom:
            self.get_logger().warn(
                "[GOAL] Goal accettato, ma odom non ancora disponibile."
            )

        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().warn(f"[CANCEL] Cancellazione richiesta | vehicle={self.vehicle_id}")
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        goal = goal_handle.request
        self.current_mission_id = goal.mission_id

        if not self.has_odom:
            self.stop_vehicle()
            goal_handle.abort()
            result = NavigateToPose.Result()
            result.success = False
            result.message = "Odom non ancora disponibile"
            self.get_logger().error("[EXEC] Abort: odom non disponibile")
            return result

        self.state = ExecutorState.NAVIGATING

        self.get_logger().info(
            f"[EXEC] Avvio navigazione | mission={goal.mission_id} | "
            f"start=({self.current_x:.2f},{self.current_y:.2f}) | "
            f"yaw={self.current_yaw:.2f} | "
            f"target=({goal.target_x:.2f},{goal.target_y:.2f})"
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
            self.get_logger().error(f"[EXEC] Errore calcolo path: {ex}")
            return result

        rate = self.create_rate(20)

        while rclpy.ok():

            # ── Cancellazione ──────────────────────────────────────────
            if goal_handle.is_cancel_requested:
                self.stop_vehicle()
                self.state = ExecutorState.IDLE
                goal_handle.canceled()
                result = NavigateToPose.Result()
                result.success = False
                result.message = "Navigazione cancellata"
                self.get_logger().warn(f"[EXEC] Cancellata | mission={goal.mission_id}")
                return result

            # ── Target raggiunto ───────────────────────────────────────
            if self.current_waypoint_index >= len(self.current_path):
                self.stop_vehicle()
                self.state = ExecutorState.IDLE
                goal_handle.succeed()
                result = NavigateToPose.Result()
                result.success = True
                result.message = "Target raggiunto"
                self.get_logger().info(
                    f"[EXEC] Target raggiunto | mission={goal.mission_id} | "
                    f"final_pos=({self.current_x:.2f},{self.current_y:.2f})"
                )
                return result

            current_wp = self.current_path[self.current_waypoint_index]

            dx = current_wp["x"] - self.current_x
            dy = current_wp["y"] - self.current_y
            distance = math.sqrt(dx * dx + dy * dy)

            is_last = self.current_waypoint_index == len(self.current_path) - 1
            tolerance = self.target_tolerance if is_last else self.waypoint_tolerance

            # ── Waypoint raggiunto ─────────────────────────────────────
            if distance <= tolerance:
                self.get_logger().info(
                    f"[WP] Waypoint raggiunto | "
                    f"idx={self.current_waypoint_index + 1}/{len(self.current_path)} | "
                    f"dist={distance:.2f} | tol={tolerance:.2f} | "
                    f"wp=({current_wp['x']:.2f},{current_wp['y']:.2f})"
                )
                self.current_waypoint_index += 1
                continue

            # ── Controllo semaforo ─────────────────────────────────────
            #
            # Se il waypoint corrente è di tipo approach_intersection,
            # significa che stiamo avvicinandoci a un nodo-incrocio.
            # Usiamo node_path per risalire a from_node e to_node.
            #
            if current_wp.get("kind") == "approach_intersection":
                intersection_node_id = current_wp.get("node_id")
                from_node_id, to_node_id = self.get_movement_for_intersection(
                    intersection_node_id
                )

                if intersection_node_id and from_node_id:
                    # Pubblica la priority request a ogni ciclo finché siamo vicini.
                    if distance <= self.traffic_light_stop_distance * 3:
                        self.publish_priority_request(
                            from_node_id,
                            to_node_id,
                            intersection_node_id,
                            goal.mission_id
                        )

                    # Se siamo nella zona di stop, aspettiamo verde.
                    if distance <= self.traffic_light_stop_distance:
                        allowed = self.is_movement_allowed(
                            from_node_id, to_node_id, intersection_node_id
                        )

                        if not allowed:
                            if self.state != ExecutorState.WAITING_TRAFFIC_LIGHT:
                                self.state = ExecutorState.WAITING_TRAFFIC_LIGHT
                                self.get_logger().info(
                                    f"[TL] Stop al semaforo | "
                                    f"intersection={intersection_node_id} | "
                                    f"{from_node_id}->{to_node_id}"
                                )

                            self.stop_vehicle()

                            feedback = self._make_feedback(goal, current_wp)
                            goal_handle.publish_feedback(feedback)

                            rate.sleep()
                            continue

                        # Verde: riprendi.
                        if self.state == ExecutorState.WAITING_TRAFFIC_LIGHT:
                            self.state = ExecutorState.NAVIGATING
                            self.get_logger().info(
                                f"[TL] Verde | intersection={intersection_node_id}"
                            )

            # ── Controllo ostacolo LiDAR ───────────────────────────────
            obstacle_factor = self.get_obstacle_speed_factor()

            if obstacle_factor == 0.0:
                if self.state != ExecutorState.OBSTACLE_STOP:
                    self.state = ExecutorState.OBSTACLE_STOP
                    self.get_logger().warn(
                        f"[OBS] Ostacolo rilevato a {self.obstacle_min_distance:.2f} m — stop"
                    )

                self.stop_vehicle()

                feedback = self._make_feedback(goal, current_wp)
                goal_handle.publish_feedback(feedback)

                rate.sleep()
                continue

            if self.state == ExecutorState.OBSTACLE_STOP:
                self.state = ExecutorState.NAVIGATING
                self.get_logger().info("[OBS] Via libera — riprendo navigazione")

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

        self.get_logger().error("[EXEC] Loop ROS interrotto")
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

        self.get_logger().info(
            f"[PATH] Start projection | "
            f"pos=({start_x:.2f},{start_y:.2f}) | "
            f"edge={start_edge['id']} {start_edge['from']}->{start_edge['to']} | "
            f"proj=({start_projection['x']:.2f},{start_projection['y']:.2f}) | "
            f"distance_from_road={start_projection['distance']:.2f}"
        )

        self.get_logger().info(
            f"[PATH] Target projection | "
            f"target=({target_x:.2f},{target_y:.2f}) | "
            f"edge={target_edge['id']} {target_edge['from']}->{target_edge['to']} | "
            f"proj=({target_projection['x']:.2f},{target_projection['y']:.2f}) | "
            f"distance_from_road={target_projection['distance']:.2f}"
        )

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

                self.get_logger().info(
                    f"[PATH] Candidate | "
                    f"{s}->{t} | nodes={node_path} | cost={total_cost:.2f}"
                )

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_node_path = node_path

        if not best_node_path:
            raise RuntimeError("nessun path trovato sul grafo")

        self.get_logger().info(
            f"[PATH] Best node path | cost={best_cost:.2f} | nodes={best_node_path}"
        )

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
                self.get_logger().debug(
                    f"[PATH] Waypoint rimosso (troppo vicino) | "
                    f"d={d:.2f} | wp=({wp['x']:.2f},{wp['y']:.2f})"
                )

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

        if distance < 1.5:
            linear_speed = min(linear_speed, 0.35)
        if distance < 0.8:
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
    # LOG
    # ------------------------------------------------------------------

    def log_path(self, path):
        self.get_logger().info(f"[PATH] Path calcolato | waypoints={len(path)}")

        if not self.path_log_enabled:
            return

        for i, wp in enumerate(path):
            self.get_logger().info(
                f"[PATH] wp[{i}] | "
                f"kind={wp.get('kind')} | "
                f"pos=({wp['x']:.2f},{wp['y']:.2f}) | "
                f"edge={wp.get('edge_id')} | "
                f"node={wp.get('node_id')} | "
                f"speed_limit={wp.get('speed_limit')}"
            )

    def maybe_log_diagnostics(self, waypoint, distance):
        now = self.get_clock().now()
        elapsed = (now - self.last_diag_time).nanoseconds / 1e9

        if elapsed < self.diagnostic_log_period_sec:
            return

        self.last_diag_time = now

        ctrl = waypoint.get("_last_control", {})

        self.get_logger().info(
            f"[DIAG] mission={self.current_mission_id} | "
            f"state={self.state.value} | "
            f"pos=({self.current_x:.2f},{self.current_y:.2f}) | "
            f"yaw={self.current_yaw:.2f} | "
            f"dist_to_wp={distance:.2f} | "
            f"mode={ctrl.get('motion_mode', '?')} | "
            f"v={ctrl.get('linear_speed', 0.0):.2f} | "
            f"w={ctrl.get('angular_speed', 0.0):.2f} | "
            f"obs={self.obstacle_min_distance:.2f} | "
            f"obs_factor={ctrl.get('obstacle_factor', 1.0):.2f}"
        )

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
