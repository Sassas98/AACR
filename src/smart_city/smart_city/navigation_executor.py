import json
import math
import os
import heapq
from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from smart_city_interfaces.action import NavigateToPose


class ExecutorState(Enum):
    IDLE = "IDLE"
    NAVIGATING = "NAVIGATING"
    WAITING_TRAFFIC_LIGHT = "WAITING_TRAFFIC_LIGHT"
    OBSTACLE_STOP = "OBSTACLE_STOP"
    RECALCULATING = "RECALCULATING"


class NavigationExecutor(Node):

    # ============================================================
    # INIT
    # ============================================================

    def __init__(self):
        super().__init__("navigation_executor")

        self.declare_parameters(
            namespace="",
            parameters=[
                ("vehicle_id", "vehicle"),
                ("map_config_file", "config/city_map.json"),
                ("pose_entity_name", ""),

                ("default_max_speed", 2.4),
                ("linear_k", 1.6),
                ("angular_k", 1.6),
                ("max_angular_speed", 0.9),
                ("lookahead_distance", 4.0),
                ("lane_recovery_threshold", 0.35),

                ("waypoint_tolerance", 0.30),
                ("target_tolerance", 0.45),

                ("lane_width", 1.2),
                ("lane_offset_ratio", 1.0),

                ("intersection_clearance", 2.2),
                ("traffic_light_stop_distance", 1.5),

                ("obstacle_stop_distance", 3.0),
                ("obstacle_slow_distance", 5.0),
                ("obstacle_fov_deg", 60.0),
                ("obstacle_replan_timeout_sec", 15.0),

                ("diagnostic_log_period_sec", 1.0),
                ("path_log_enabled", True),
            ]
        )

        self.vehicle_id = self.get_parameter("vehicle_id").value
        self.map_config_file = self.get_parameter("map_config_file").value
        self.pose_entity_name = self.get_parameter("pose_entity_name").value or self.vehicle_id

        self.default_max_speed = float(self.get_parameter("default_max_speed").value)
        self.linear_k = float(self.get_parameter("linear_k").value)
        self.angular_k = float(self.get_parameter("angular_k").value)
        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)

        self.waypoint_tolerance = float(self.get_parameter("waypoint_tolerance").value)
        self.target_tolerance = float(self.get_parameter("target_tolerance").value)

        self.lane_width = float(self.get_parameter("lane_width").value)
        self.lane_offset_ratio = float(self.get_parameter("lane_offset_ratio").value)
        self.lane_recovery_threshold = float(self.get_parameter("lane_recovery_threshold").value)
        self.lookahead_distance = float(self.get_parameter("lookahead_distance").value)

        self.intersection_clearance = float(self.get_parameter("intersection_clearance").value)
        self.traffic_light_stop_distance = float(self.get_parameter("traffic_light_stop_distance").value)

        self.obstacle_stop_distance = float(self.get_parameter("obstacle_stop_distance").value)
        self.obstacle_slow_distance = float(self.get_parameter("obstacle_slow_distance").value)
        self.obstacle_fov_deg = float(self.get_parameter("obstacle_fov_deg").value)
        self.obstacle_replan_timeout_sec = float(self.get_parameter("obstacle_replan_timeout_sec").value)

        self.diagnostic_log_period_sec = float(self.get_parameter("diagnostic_log_period_sec").value)
        self.path_log_enabled = bool(self.get_parameter("path_log_enabled").value)

        # Stato veicolo
        self.state = ExecutorState.IDLE
        self.has_odom = False
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        # Sensori
        self.obstacle_min_distance = float("inf")
        self.last_scan_stamp = None
        self.obstacle_stop_start_time = None

        self.other_vehicles = {}
        self.vehicle_state_publish_period_sec = 0.2
        self.vehicle_stop_distance = 4.0
        self.vehicle_slow_distance = 7.0
        self.vehicle_corridor_width = 2.0
        self.vehicle_state_pub = self.create_publisher(
            String,
            "/vehicle_states",
            100
        )

        self.vehicle_state_sub = self.create_subscription(
            String,
            "/vehicle_states",
            self.on_vehicle_state,
            100
        )

        self.vehicle_state_timer = self.create_timer(
            self.vehicle_state_publish_period_sec,
            self.publish_vehicle_state
        )

        # Semafori
        self.traffic_light_statuses = {}
        self.last_priority_request_time = {}

        # Mappa
        self.nodes = {}
        self.edges = []
        self.edge_by_id = {}
        self.adj = {}
        self.default_map_speed_limit = 1.4
        self.blocked_edges = {}
        self.blocked_edge_ttl_sec = 45.0

        # Navigazione
        self.current_path = []
        self.node_path = []
        self.current_waypoint_index = 0
        self.current_mission_id = ""

        self.last_diag_time = self.get_clock().now()

        self.load_map()

        # ROS
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

    # ============================================================
    # MAPPA
    # ============================================================

    def load_map(self):
        path = self.map_config_file

        if not os.path.isabs(path):
            path = os.path.join(os.getcwd(), path)

        if not os.path.exists(path):
            raise FileNotFoundError(f"File mappa non trovato: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.lane_width = float(data.get("lane_width", self.lane_width))
        self.default_map_speed_limit = float(data.get("default_speed_limit", 1.4))

        for node in data["nodes"]:
            self.nodes[node["id"]] = {
                "id": node["id"],
                "x": float(node["x"]),
                "y": float(node["y"]),
            }

        for edge in data["edges"]:
            edge_id = edge["id"]
            from_id = edge["from"]
            to_id = edge["to"]

            if from_id not in self.nodes or to_id not in self.nodes:
                raise RuntimeError(f"Edge {edge_id} usa nodi non esistenti: {from_id}->{to_id}")

            a = self.nodes[from_id]
            b = self.nodes[to_id]

            length = self.distance_xy(a["x"], a["y"], b["x"], b["y"])

            e = {
                "id": edge_id,
                "from": from_id,
                "to": to_id,
                "speed_limit": float(edge.get("speed_limit", self.default_map_speed_limit)),
                "length": length,
            }

            self.edges.append(e)
            self.edge_by_id[edge_id] = e

            self.adj.setdefault(from_id, [])
            self.adj.setdefault(to_id, [])

            self.adj[from_id].append((to_id, edge_id, length))
            self.adj[to_id].append((from_id, edge_id, length))

    # ============================================================
    # SENSORI
    # ============================================================

    def on_world_pose(self, msg):
        p = msg.pose.position
        q = msg.pose.orientation

        self.current_x = float(p.x)
        self.current_y = float(p.y)

        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)

        raw_yaw = math.atan2(siny_cosp, cosy_cosp)

        # Compensazione modello: il frontale reale del bus è ruotato di 180°.
        self.current_yaw = self.normalize_angle(raw_yaw + math.pi)

        self.has_odom = True

    def on_scan(self, msg):
        self.last_scan_stamp = msg.header.stamp

        fov_rad = math.radians(self.obstacle_fov_deg / 2.0)
        min_dist = float("inf")

        angle = msg.angle_min

        for r in msg.ranges:
            if msg.range_min <= r <= msg.range_max:
                normalized = self.normalize_angle(angle)

                if abs(normalized) <= fov_rad:
                    min_dist = min(min_dist, r)

            angle += msg.angle_increment

        self.obstacle_min_distance = min_dist

    def publish_vehicle_state(self):
        if not self.has_odom:
            return

        payload = {
            "vehicle_id": self.vehicle_id,
            "x": self.current_x,
            "y": self.current_y,
            "yaw": self.current_yaw,
            "stamp": self.get_clock().now().nanoseconds / 1e9,
        }

        msg = String()
        msg.data = json.dumps(payload)
        self.vehicle_state_pub.publish(msg)


    def on_vehicle_state(self, msg):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        vehicle_id = data.get("vehicle_id")

        if not vehicle_id or vehicle_id == self.vehicle_id:
            return

        self.other_vehicles[vehicle_id] = data

    def on_traffic_light_status(self, msg):
        try:
            status = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        node_id = status.get("node_id")

        if node_id:
            self.traffic_light_statuses[node_id] = status

    # ============================================================
    # ACTION SERVER
    # ============================================================

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

        try:
            self.state = ExecutorState.NAVIGATING
            self.plan_path_to_goal(goal)

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
        waypoint_timeout_sec = self.compute_waypoint_timeout(self.current_path[0])

        while rclpy.ok():

            if goal_handle.is_cancel_requested:
                self.stop_vehicle()
                self.state = ExecutorState.IDLE
                goal_handle.canceled()

                result = NavigateToPose.Result()
                result.success = False
                result.message = "Navigazione cancellata"
                return result

            if self.current_waypoint_index >= len(self.current_path):
                final_wp = self.current_path[-1]

                final_distance = self.distance_xy(
                    self.current_x,
                    self.current_y,
                    final_wp["x"],
                    final_wp["y"]
                )

                if final_distance > self.target_tolerance:
                    self.stop_vehicle()
                    self.state = ExecutorState.IDLE
                    goal_handle.abort()

                    result = NavigateToPose.Result()
                    result.success = False
                    result.message = (
                        f"Path terminato ma target non raggiunto: distanza={final_distance:.2f} m"
                    )
                    return result

                self.stop_vehicle()
                self.state = ExecutorState.IDLE
                goal_handle.succeed()

                result = NavigateToPose.Result()
                result.success = True
                result.message = "Target raggiunto"
                return result

            current_wp = self.current_path[self.current_waypoint_index]

            distance_to_wp = self.distance_xy(
                self.current_x,
                self.current_y,
                current_wp["x"],
                current_wp["y"]
            )

            is_last = self.current_waypoint_index == len(self.current_path) - 1
            tolerance = self.target_tolerance if is_last else self.waypoint_tolerance

            if distance_to_wp <= tolerance:
                self.log_navigation_snapshot("waypoint raggiunto", current_wp, distance_to_wp, force=True)

                self.current_waypoint_index += 1
                waypoint_start_time = self.get_clock().now()

                if self.current_waypoint_index < len(self.current_path):
                    waypoint_timeout_sec = self.compute_waypoint_timeout(
                        self.current_path[self.current_waypoint_index]
                    )

                rate.sleep()
                continue

            elapsed_on_wp = (self.get_clock().now() - waypoint_start_time).nanoseconds / 1e9

            if elapsed_on_wp > waypoint_timeout_sec:
                if self.obstacle_min_distance <= self.obstacle_stop_distance:
                    self.log_navigation_snapshot(
                        f"timeout con ostacolo: blocco strada e ricalcolo",
                        current_wp,
                        distance_to_wp,
                        force=True
                    )

                    self.stop_vehicle()
                    self.mark_current_road_blocked(current_wp)
                    self.replan_after_obstacle(goal)

                else:
                    self.log_navigation_snapshot(
                        f"timeout senza ostacolo: continuo, non blocco strade",
                        current_wp,
                        distance_to_wp,
                        force=True
                    )

                    waypoint_start_time = self.get_clock().now()
                    waypoint_timeout_sec = self.compute_waypoint_timeout(current_wp)

                rate.sleep()
                continue

            if self.must_wait_at_traffic_light(current_wp, goal, distance_to_wp):
                self.stop_vehicle()
                waypoint_start_time = self.get_clock().now()

                feedback = self.make_feedback(goal)
                goal_handle.publish_feedback(feedback)

                rate.sleep()
                continue

            obstacle_factor = self.get_obstacle_speed_factor()

            if obstacle_factor == 0.0:
                handled = self.handle_obstacle_stop(goal, current_wp)

                waypoint_start_time = self.get_clock().now()

                feedback = self.make_feedback(goal)
                goal_handle.publish_feedback(feedback)

                rate.sleep()
                continue

            if self.state == ExecutorState.OBSTACLE_STOP:
                self.state = ExecutorState.NAVIGATING
                self.obstacle_stop_start_time = None
                self.log_navigation_snapshot("via libera: riprendo navigazione", current_wp, distance_to_wp, force=True)

            self.move_towards_waypoint(current_wp, goal.max_speed, obstacle_factor)

            feedback = self.make_feedback(goal)
            goal_handle.publish_feedback(feedback)

            self.maybe_log_diagnostics(current_wp, distance_to_wp)

            rate.sleep()

        self.stop_vehicle()
        self.state = ExecutorState.IDLE
        goal_handle.abort()

        result = NavigateToPose.Result()
        result.success = False
        result.message = "Navigazione interrotta"
        return result

    def plan_path_to_goal(self, goal):
        self.log_navigation_event(
            f"parto da ({self.current_x:.2f},{self.current_y:.2f}) "
            f"verso target ({goal.target_x:.2f},{goal.target_y:.2f})"
        )

        self.current_path, self.node_path = self.build_navigation_path(
            self.current_x,
            self.current_y,
            goal.target_x,
            goal.target_y
        )

        self.current_waypoint_index = 0

        if not self.current_path:
            raise RuntimeError("path vuoto")

        self.log_path(self.current_path)

    def make_feedback(self, goal):
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

    def compute_waypoint_timeout(self, waypoint):
        speed_limit = max(
            0.5,
            float(waypoint.get("speed_limit", self.default_max_speed))
        )

        if self.current_waypoint_index == 0:
            previous_x = self.current_x
            previous_y = self.current_y
        else:
            previous_wp = self.current_path[self.current_waypoint_index - 1]
            previous_x = previous_wp["x"]
            previous_y = previous_wp["y"]

        segment_length = self.distance_xy(
            previous_x,
            previous_y,
            waypoint["x"],
            waypoint["y"]
        )

        expected_time = segment_length / speed_limit

        return max(8.0, expected_time * 4.0)

    # ============================================================
    # OSTACOLI
    # ============================================================

    def get_vehicle_proximity_factor(self):
        if not self.has_odom:
            return 1.0

        now = self.get_clock().now().nanoseconds / 1e9

        forward_x = math.cos(self.current_yaw)
        forward_y = math.sin(self.current_yaw)

        min_forward_dist = float("inf")
        closest_vehicle = None

        for vehicle_id, other in list(self.other_vehicles.items()):
            stamp = float(other.get("stamp", 0.0))

            if now - stamp > 1.0:
                continue

            dx = float(other["x"]) - self.current_x
            dy = float(other["y"]) - self.current_y

            forward_dist = dx * forward_x + dy * forward_y
            side_dist = abs(-forward_y * dx + forward_x * dy)

            if forward_dist <= 0.0:
                continue

            if side_dist > self.vehicle_corridor_width:
                continue

            if forward_dist < min_forward_dist:
                min_forward_dist = forward_dist
                closest_vehicle = vehicle_id

        if min_forward_dist <= self.vehicle_stop_distance:
            self.log_navigation_event(
                f"vehicle proximity stop: davanti ho {closest_vehicle} a {min_forward_dist:.2f} m"
            )
            return 0.0

        if min_forward_dist <= self.vehicle_slow_distance:
            ratio = (
                (min_forward_dist - self.vehicle_stop_distance)
                / (self.vehicle_slow_distance - self.vehicle_stop_distance)
            )

            self.log_navigation_event(
                f"vehicle proximity slow: davanti ho {closest_vehicle} a {min_forward_dist:.2f} m"
            )

            return max(0.0, min(1.0, ratio))

        return 1.0

    def get_obstacle_speed_factor(self):
        d = self.obstacle_min_distance

        lidar_factor = 1.0

        if d <= self.obstacle_stop_distance:
            lidar_factor = 0.0
        elif d <= self.obstacle_slow_distance:
            lidar_factor = (d - self.obstacle_stop_distance) / (
                self.obstacle_slow_distance - self.obstacle_stop_distance
            )
            lidar_factor = max(0.0, min(1.0, lidar_factor))

        vehicle_factor = self.get_vehicle_proximity_factor()

        return min(lidar_factor, vehicle_factor)

    def handle_obstacle_stop(self, goal, current_wp):
        now = self.get_clock().now()

        if self.obstacle_stop_start_time is None:
            self.obstacle_stop_start_time = now

        stopped_for = (now - self.obstacle_stop_start_time).nanoseconds / 1e9

        if self.state != ExecutorState.OBSTACLE_STOP:
            self.state = ExecutorState.OBSTACLE_STOP
            self.log_navigation_snapshot(
                f"ostacolo davanti a {self.obstacle_min_distance:.2f} m: stop",
                current_wp,
                None,
                force=True
            )

        self.stop_vehicle()

        if stopped_for < self.obstacle_replan_timeout_sec:
            return False

        self.log_navigation_snapshot(
            f"ostacolo persistente da {stopped_for:.1f}s: blocco strada e ricalcolo",
            current_wp,
            None,
            force=True
        )

        self.mark_current_road_blocked(current_wp)
        self.replan_after_obstacle(goal)

        self.obstacle_stop_start_time = None
        return True
    
    def block_edge_temporarily(self, edge_id):
        if not edge_id:
            return

        expire_at = self.get_clock().now().nanoseconds / 1e9 + self.blocked_edge_ttl_sec
        self.blocked_edges[edge_id] = expire_at

    def cleanup_expired_blocked_edges(self):
        now = self.get_clock().now().nanoseconds / 1e9

        expired = [
            edge_id
            for edge_id, expire_at in self.blocked_edges.items()
            if expire_at <= now
        ]

        for edge_id in expired:
            del self.blocked_edges[edge_id]

    def is_edge_blocked(self, edge_id):
        self.cleanup_expired_blocked_edges()
        return edge_id in self.blocked_edges

    def mark_current_road_blocked(self, current_wp):
        edge_id = current_wp.get("edge_id")

        if edge_id:
            self.block_edge_temporarily(edge_id)

        try:
            lane_projection = self.find_nearest_lane_projection(
                self.current_x,
                self.current_y,
                preferred_edge_id=edge_id,
                allow_blocked=True
            )

            if lane_projection and lane_projection["edge"]:
                self.block_edge_temporarily(lane_projection["edge"]["id"])

        except Exception as ex:
            self.log_navigation_event(
                f"impossibile localizzare corsia durante blocco strada: {ex}"
            )

        self.log_navigation_event(
            "strade bloccate: " + ", ".join(sorted(self.blocked_edges))
        )

    def replan_after_obstacle(self, goal):
        self.state = ExecutorState.RECALCULATING

        try:
            self.current_path, self.node_path = self.build_navigation_path(
                self.current_x,
                self.current_y,
                goal.target_x,
                goal.target_y
            )

            self.current_waypoint_index = 0
            self.state = ExecutorState.NAVIGATING
            self.log_path(self.current_path)

        except Exception as ex:
            self.stop_vehicle()
            self.state = ExecutorState.OBSTACLE_STOP
            self.log_navigation_event(f"ricalcolo fallito: {ex}")

    # ============================================================
    # SEMAFORI
    # ============================================================

    def must_wait_at_traffic_light(self, current_wp, goal, distance_to_wp):
        if current_wp.get("kind") != "approach_intersection":
            return False

        intersection_node_id = current_wp.get("node_id")
        from_node_id, to_node_id = self.get_movement_for_intersection(intersection_node_id)

        if not intersection_node_id or not from_node_id:
            return False

        if distance_to_wp <= self.traffic_light_stop_distance * 3:
            self.maybe_publish_priority_request(
                from_node_id,
                to_node_id,
                intersection_node_id,
                goal.mission_id
            )

        if distance_to_wp > self.traffic_light_stop_distance:
            return False

        allowed = self.is_movement_allowed(
            from_node_id,
            to_node_id,
            intersection_node_id
        )

        if allowed:
            if self.state == ExecutorState.WAITING_TRAFFIC_LIGHT:
                self.state = ExecutorState.NAVIGATING
                self.log_navigation_snapshot(
                    f"semaforo {intersection_node_id} verde: riparto",
                    current_wp,
                    distance_to_wp,
                    force=True
                )

            return False

        if self.state != ExecutorState.WAITING_TRAFFIC_LIGHT:
            self.state = ExecutorState.WAITING_TRAFFIC_LIGHT
            self.log_navigation_snapshot(
                f"fermo al semaforo {intersection_node_id}: "
                f"movimento {from_node_id}->{intersection_node_id}->{to_node_id} non consentito",
                current_wp,
                distance_to_wp,
                force=True
            )

        return True

    def maybe_publish_priority_request(self, from_node_id, to_node_id, intersection_node_id, mission_id):
        now_sec = self.get_clock().now().nanoseconds / 1e9
        last_sent = self.last_priority_request_time.get(intersection_node_id, 0.0)

        if now_sec - last_sent < 2.0:
            return

        payload = {
            "vehicle_id": self.vehicle_id,
            "mission_id": mission_id,
            "node_id": intersection_node_id,
            "from_node_id": from_node_id,
            "to_node_id": to_node_id,
            "priority": 1,
        }

        msg = String()
        msg.data = json.dumps(payload)
        self.priority_pub.publish(msg)

        self.last_priority_request_time[intersection_node_id] = now_sec

        self.log_navigation_event(
            f"chiedo priorità al semaforo {intersection_node_id}: "
            f"movimento {from_node_id}->{intersection_node_id}->{to_node_id}"
        )

    def is_movement_allowed(self, from_node_id, to_node_id, intersection_node_id):
        status = self.traffic_light_statuses.get(intersection_node_id)

        if status is None:
            return True

        allowed = status.get("allowed_movements", [])

        for movement in allowed:
            if movement.get("from") == from_node_id and movement.get("to") == to_node_id:
                return True

        return False

    def get_movement_for_intersection(self, intersection_node_id):
        if not self.node_path:
            return None, None

        try:
            idx = self.node_path.index(intersection_node_id)
        except ValueError:
            return None, None

        from_node_id = self.node_path[idx - 1] if idx > 0 else None
        to_node_id = self.node_path[idx + 1] if idx < len(self.node_path) - 1 else None

        return from_node_id, to_node_id

    # ============================================================
    # PATH PLANNING
    # ============================================================

    def path_uses_blocked_edge(self, node_path):
        if len(node_path) < 2:
            return False

        for i in range(len(node_path) - 1):
            edge = self.get_edge_between(node_path[i], node_path[i + 1])

            if edge["id"] in self.blocked_edges:
                return True

        return False

    def find_nearest_lane_projection(
        self,
        x,
        y,
        preferred_edge_id=None,
        destination_node_id=None,
        allow_blocked=False
    ):
        best = None
        best_distance = float("inf")

        candidate_edges = self.edges

        if preferred_edge_id in self.edge_by_id:
            candidate_edges = [self.edge_by_id[preferred_edge_id]]

        for edge in candidate_edges:
            if not allow_blocked and edge["id"] in self.blocked_edges:
                continue

            a = self.nodes[edge["from"]]
            b = self.nodes[edge["to"]]

            center_projection = self.project_point_on_segment(
                x, y,
                a["x"], a["y"],
                b["x"], b["y"]
            )

            possible_destinations = []

            if destination_node_id in (edge["from"], edge["to"]):
                possible_destinations = [destination_node_id]
            else:
                # Se non so il verso, provo entrambe le direzioni:
                # edge[from]->edge[to] e edge[to]->edge[from].
                possible_destinations = [edge["from"], edge["to"]]

            for dest in possible_destinations:
                lane_projection = self.project_center_projection_to_right_lane(
                    {
                        "edge": edge,
                        "center_x": center_projection["x"],
                        "center_y": center_projection["y"],
                        "x": center_projection["x"],
                        "y": center_projection["y"],
                        "t": center_projection["t"],
                    },
                    dest
                )

                distance = self.distance_xy(
                    x,
                    y,
                    lane_projection["x"],
                    lane_projection["y"]
                )

                if distance < best_distance:
                    best_distance = distance
                    best = {
                        "edge": edge,
                        "center_x": center_projection["x"],
                        "center_y": center_projection["y"],
                        "x": lane_projection["x"],
                        "y": lane_projection["y"],
                        "t": center_projection["t"],
                        "distance": distance,
                        "destination_node_id": dest,
                    }

        if best is None:
            if allow_blocked:
                raise RuntimeError("impossibile proiettare sulla corsia")
            return self.find_nearest_lane_projection(
                x,
                y,
                preferred_edge_id=preferred_edge_id,
                destination_node_id=destination_node_id,
                allow_blocked=True
            )

        return best

    def build_navigation_path(self, start_x, start_y, target_x, target_y):
        # ------------------------------------------------------------
        # 1. Proiezioni grezze: servono solo per capire quali edge usare
        # ------------------------------------------------------------
        start_projection_raw = self.find_nearest_lane_projection(
            start_x,
            start_y,
            allow_blocked=True
        )

        target_projection_raw = self.find_nearest_lane_projection(
            target_x,
            target_y,
            allow_blocked=False
        )

        start_edge_raw = start_projection_raw["edge"]
        target_edge_raw = target_projection_raw["edge"]

        start_candidates = [
            start_edge_raw["from"],
            start_edge_raw["to"]
        ]

        target_candidates = [
            target_edge_raw["from"],
            target_edge_raw["to"]
        ]

        # ------------------------------------------------------------
        # 2. Scelta del miglior percorso tra candidati
        # ------------------------------------------------------------
        best_node_path = None
        best_cost = float("inf")

        for s in start_candidates:
            for t in target_candidates:
                node_path, graph_cost = self.shortest_path(s, t)

                if node_path is None:
                    continue

                if self.path_uses_blocked_edge(node_path):
                    continue

                total_cost = (
                    graph_cost
                    + self.distance_xy(
                        start_x,
                        start_y,
                        self.nodes[s]["x"],
                        self.nodes[s]["y"]
                    )
                    + self.distance_xy(
                        target_x,
                        target_y,
                        self.nodes[t]["x"],
                        self.nodes[t]["y"]
                    )
                )

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_node_path = node_path

        if not best_node_path:
            raise RuntimeError("nessun path trovato sul grafo")

        # ------------------------------------------------------------
        # 3. Ora che conosco il verso reale, rifaccio le proiezioni corsia
        # ------------------------------------------------------------
        first_destination = (
            best_node_path[1]
            if len(best_node_path) > 1
            else best_node_path[0]
        )

        final_destination = best_node_path[-1]

        start_projection = self.find_nearest_lane_projection(
            start_x,
            start_y,
            allow_blocked=True,
            destination_node_id=first_destination
        )

        target_projection = self.find_nearest_lane_projection(
            target_x,
            target_y,
            allow_blocked=False,
            destination_node_id=final_destination
        )

        start_edge = start_projection["edge"]
        target_edge = target_projection["edge"]

        # ------------------------------------------------------------
        # 4. Costruzione waypoint
        # ------------------------------------------------------------
        waypoints = []

        waypoints.append({
            "x": start_projection["x"],
            "y": start_projection["y"],
            "edge_id": start_edge["id"],
            "node_id": None,
            "kind": "start_lane_projection",
            "speed_limit": start_edge["speed_limit"],
        })

        for i in range(len(best_node_path) - 1):
            current_node_id = best_node_path[i]
            next_node_id = best_node_path[i + 1]

            edge = self.get_edge_between(current_node_id, next_node_id)

            approach = self.node_to_right_lane_point(
                node_id=next_node_id,
                other_node_id=current_node_id,
                mode="approach"
            )

            waypoints.append({
                "x": approach["x"],
                "y": approach["y"],
                "edge_id": edge["id"],
                "node_id": next_node_id,
                "kind": "approach_intersection",
                "speed_limit": edge["speed_limit"],
            })

        waypoints.append({
            "x": target_projection["x"],
            "y": target_projection["y"],
            "edge_id": target_edge["id"],
            "node_id": None,
            "kind": "target_lane_projection",
            "speed_limit": target_edge["speed_limit"],
        })

        # ------------------------------------------------------------
        # 5. Pulizia e log
        # ------------------------------------------------------------
        waypoints = self.simplify_waypoints(waypoints)

        self.log_built_path(
            start_x,
            start_y,
            target_x,
            target_y,
            best_node_path,
            waypoints
        )

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
                if self.is_edge_blocked(edge_id):
                    continue

                if neighbor_id not in visited:
                    heapq.heappush(queue, (cost + length, neighbor_id, new_path))

        return None, float("inf")
    # ============================================================
    # CORSIE / GEOMETRIA STRADALE
    # ============================================================

    def project_center_projection_to_right_lane(self, projection, destination_node_id):
        edge = projection["edge"]

        a = self.nodes[edge["from"]]
        b = self.nodes[edge["to"]]

        if destination_node_id == edge["to"]:
            from_node = a
            to_node = b
        else:
            from_node = b
            to_node = a

        base_x = projection.get("center_x", projection["x"])
        base_y = projection.get("center_y", projection["y"])

        lane_x, lane_y = self.apply_right_lane_offset(
            base_x,
            base_y,
            from_node["x"],
            from_node["y"],
            to_node["x"],
            to_node["y"]
        )

        return {
            "x": lane_x,
            "y": lane_y,
            "edge_id": edge["id"],
            "destination_node_id": destination_node_id,
        }

    def node_to_right_lane_point(self, node_id, other_node_id, mode):
        node = self.nodes[node_id]
        other = self.nodes[other_node_id]

        dx = other["x"] - node["x"]
        dy = other["y"] - node["y"]

        length = math.sqrt(dx * dx + dy * dy)

        if length <= 0.000001:
            return {
                "x": node["x"],
                "y": node["y"],
            }

        ux = dx / length
        uy = dy / length

        clearance = self.intersection_clearance

        if mode == "approach":
            base_x = node["x"] - ux * clearance
            base_y = node["y"] - uy * clearance

            lane_from_x = other["x"]
            lane_from_y = other["y"]
            lane_to_x = node["x"]
            lane_to_y = node["y"]

        elif mode == "exit":
            base_x = node["x"] + ux * clearance
            base_y = node["y"] + uy * clearance

            lane_from_x = node["x"]
            lane_from_y = node["y"]
            lane_to_x = other["x"]
            lane_to_y = other["y"]

        else:
            raise RuntimeError(f"mode non valido: {mode}")

        lane_x, lane_y = self.apply_right_lane_offset(
            base_x,
            base_y,
            lane_from_x,
            lane_from_y,
            lane_to_x,
            lane_to_y
        )

        return {
            "x": lane_x,
            "y": lane_y,
        }

    def apply_right_lane_offset(self, x, y, from_x, from_y, to_x, to_y):
        dx = to_x - from_x
        dy = to_y - from_y

        length = math.sqrt(dx * dx + dy * dy)

        if length <= 0.000001:
            return x, y

        ux = dx / length
        uy = dy / length

        right_x = uy
        right_y = -ux

        offset = self.lane_width * self.lane_offset_ratio

        return (
            x + right_x * offset,
            y + right_y * offset
        )

    def get_lane_follow_target(self, waypoint):
        """
        Target usato dal controller.

        Regola:
        - se sono fuori corsia, punto alla corsia più vicina coerente con l'edge del waypoint;
        - se sono già in corsia, punto a un lookahead sulla corsia verso il waypoint;
        - così il bus non taglia più in diagonale tra waypoint lontani.
        """
        preferred_edge_id = waypoint.get("edge_id")
        destination_node_id = waypoint.get("node_id")

        lane_projection = self.find_nearest_lane_projection(
            self.current_x,
            self.current_y,
            preferred_edge_id=preferred_edge_id,
            destination_node_id=destination_node_id,
            allow_blocked=True
        )

        edge = lane_projection["edge"]

        dist_current_to_wp = self.distance_xy(
            self.current_x,
            self.current_y,
            waypoint["x"],
            waypoint["y"]
        )

        # SE SONO VICINO AL WAYPOINT:
        # smetto di fare lane recovery/lookahead
        # e punto direttamente il waypoint
        if dist_current_to_wp <= max(
            self.lookahead_distance,
            self.waypoint_tolerance * 4.0
        ):
            return {
                "x": waypoint["x"],
                "y": waypoint["y"],
                "mode_prefix": "WAYPOINT_FINAL_APPROACH",
                "lane_error": lane_projection["distance"],
                "edge_id": edge["id"],
            }

        # SOLO SE SONO LONTANO E FUORI CORSIA:
        # faccio lane recovery
        if lane_projection["distance"] > self.lane_recovery_threshold:
            return {
                "x": lane_projection["x"],
                "y": lane_projection["y"],
                "mode_prefix": "LANE_RECOVERY",
                "lane_error": lane_projection["distance"],
                "edge_id": edge["id"],
            }

        # altrimenti lookahead normale...

        lookahead = self.compute_lane_lookahead_point(
            edge,
            lane_projection["t"],
            destination_node_id,
            self.lookahead_distance
        )

        dist_lookahead_to_wp = self.distance_xy(
            lookahead["x"],
            lookahead["y"],
            waypoint["x"],
            waypoint["y"]
        )

        dist_current_to_wp = self.distance_xy(
            self.current_x,
            self.current_y,
            waypoint["x"],
            waypoint["y"]
        )

        if dist_current_to_wp <= max(self.lookahead_distance, self.waypoint_tolerance * 2.0):
            return {
                "x": waypoint["x"],
                "y": waypoint["y"],
                "mode_prefix": "WAYPOINT_FINAL_APPROACH",
                "lane_error": lane_projection["distance"],
                "edge_id": edge["id"],
            }

        if dist_lookahead_to_wp > dist_current_to_wp:
            return {
                "x": waypoint["x"],
                "y": waypoint["y"],
                "mode_prefix": "WAYPOINT_APPROACH",
                "lane_error": lane_projection["distance"],
                "edge_id": edge["id"],
            }

        return {
            "x": lookahead["x"],
            "y": lookahead["y"],
            "mode_prefix": "LANE_FOLLOW",
            "lane_error": lane_projection["distance"],
            "edge_id": edge["id"],
        }

    def compute_lane_lookahead_point(self, edge, current_t, destination_node_id, lookahead_distance):
        a = self.nodes[edge["from"]]
        b = self.nodes[edge["to"]]

        edge_length = max(edge["length"], 0.000001)
        delta_t = lookahead_distance / edge_length

        if destination_node_id == edge["to"]:
            next_t = min(1.0, current_t + delta_t)
            lane_destination = edge["to"]
        else:
            next_t = max(0.0, current_t - delta_t)
            lane_destination = edge["from"]

        center_x = a["x"] + (b["x"] - a["x"]) * next_t
        center_y = a["y"] + (b["y"] - a["y"]) * next_t

        lane = self.project_center_projection_to_right_lane(
            {
                "edge": edge,
                "x": center_x,
                "y": center_y,
                "t": next_t,
            },
            lane_destination
        )

        return lane

    def project_point_on_segment(self, px, py, ax, ay, bx, by):
        dx = bx - ax
        dy = by - ay

        denom = dx * dx + dy * dy

        if denom <= 0.000001:
            return {
                "x": ax,
                "y": ay,
                "t": 0.0,
            }

        t = ((px - ax) * dx + (py - ay) * dy) / denom
        t = max(0.0, min(1.0, t))

        return {
            "x": ax + t * dx,
            "y": ay + t * dy,
            "t": t,
        }

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

            d = self.distance_xy(
                last["x"],
                last["y"],
                wp["x"],
                wp["y"]
            )

            if d >= 0.35:
                result.append(wp)

        return result

    # ============================================================
    # CONTROLLO MOVIMENTO
    # ============================================================

    def move_towards_waypoint(self, waypoint, requested_max_speed, obstacle_factor=1.0):
        follow_target = self.get_lane_follow_target(waypoint)

        target_x = follow_target["x"]
        target_y = follow_target["y"]

        dx = target_x - self.current_x
        dy = target_y - self.current_y

        target_angle = math.atan2(dy, dx)

        angle_error = math.atan2(
            math.sin(target_angle - self.current_yaw),
            math.cos(target_angle - self.current_yaw)
        )

        distance = math.sqrt(dx * dx + dy * dy)

        edge_speed_limit = float(
            waypoint.get("speed_limit", self.default_map_speed_limit)
        )

        max_speed = (
            float(requested_max_speed)
            if float(requested_max_speed) > 0.0
            else self.default_max_speed
        )

        max_speed = min(max_speed, edge_speed_limit)

        angular_speed = self.angular_k * angle_error
        angular_speed = -self.clamp(
            angular_speed,
            -self.max_angular_speed,
            self.max_angular_speed
        )

        abs_error = abs(angle_error)

        if abs_error > 0.35:
            linear_speed = 0.0
            motion_mode = "TURN_IN_PLACE"
        elif abs_error > 0.15:
            linear_speed = min(max_speed, self.linear_k * distance, 0.25)
            motion_mode = "SLOW_TURN"
        else:
            linear_speed = min(max_speed, self.linear_k * distance)
            motion_mode = "FORWARD"

        if distance < 0.8:
            linear_speed = min(linear_speed, 0.35)

        if distance < 0.4:
            linear_speed = min(linear_speed, 0.20)

        linear_speed *= obstacle_factor

        if follow_target["mode_prefix"]:
            motion_mode = f'{follow_target["mode_prefix"]}_{motion_mode}'

        cmd = Twist()
        cmd.linear.x = float(linear_speed)
        cmd.angular.z = float(angular_speed)

        self.cmd_vel_pub.publish(cmd)

        waypoint["_last_control"] = {
            "target_x": target_x,
            "target_y": target_y,
            "target_angle": target_angle,
            "angle_error": angle_error,
            "distance": distance,
            "lane_error": follow_target["lane_error"],
            "linear_speed": linear_speed,
            "angular_speed": angular_speed,
            "motion_mode": motion_mode,
            "max_speed": max_speed,
            "obstacle_factor": obstacle_factor,
            "edge_id": follow_target["edge_id"],
        }

    def stop_vehicle(self):
        self.cmd_vel_pub.publish(Twist())

    # ============================================================
    # LOGGING
    # ============================================================

    def log_navigation_event(self, message):
        self.get_logger().info(
            f"[NAV] vehicle={self.vehicle_id} | mission={self.current_mission_id} | {message}"
        )

    def log_built_path(self, start_x, start_y, target_x, target_y, node_path, waypoints):
        if not self.path_log_enabled:
            return

        self.log_navigation_event(
            f"path build | start=({start_x:.2f},{start_y:.2f}) "
            f"target=({target_x:.2f},{target_y:.2f}) | "
            f"node_path={' -> '.join(node_path)} | "
            f"waypoints={len(waypoints)}"
        )

        for i, wp in enumerate(waypoints):
            self.log_navigation_event(
                f"wp[{i}] kind={wp.get('kind')} "
                f"node={wp.get('node_id')} "
                f"edge={wp.get('edge_id')} "
                f"pos=({wp['x']:.2f},{wp['y']:.2f}) "
                f"speed={wp.get('speed_limit', -1):.2f}"
            )

    def log_path(self, path):
        if not self.node_path:
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
                f"tratto {i + 1}: nodo {a} -> nodo {b} "
                f"sull'edge {edge['id']} lungo {edge['length']:.2f} m"
            )

        self.log_navigation_event(f"waypoint generati: {len(path)}")

    def describe_graph_position(self):
        if not self.edges:
            return "grafo non disponibile"

        lane_projection = self.find_nearest_lane_projection(
            self.current_x,
            self.current_y
        )

        edge = lane_projection["edge"]
        from_node_id = edge["from"]
        to_node_id = edge["to"]

        from_node = self.nodes[from_node_id]
        to_node = self.nodes[to_node_id]

        dist_from = self.distance_xy(
            self.current_x,
            self.current_y,
            from_node["x"],
            from_node["y"]
        )

        dist_to = self.distance_xy(
            self.current_x,
            self.current_y,
            to_node["x"],
            to_node["y"]
        )

        nearest_node = from_node_id if dist_from <= dist_to else to_node_id
        nearest_dist = min(dist_from, dist_to)

        return (
            f"sono a ({self.current_x:.2f},{self.current_y:.2f}), "
            f"yaw={self.current_yaw:.2f}; "
            f"corsia più vicina=edge {edge['id']} tra {from_node_id} e {to_node_id}; "
            f"lane_projection=({lane_projection['x']:.2f},{lane_projection['y']:.2f}), "
            f"center_projection=({lane_projection['center_x']:.2f},{lane_projection['center_y']:.2f}), "
            f"t={lane_projection['t']:.2f}, "
            f"fuori corsia={lane_projection['distance']:.2f} m; "
            f"nodo più vicino={nearest_node} ({nearest_dist:.2f} m)"
        )

    def describe_waypoint(self, waypoint, distance):
        if waypoint is None:
            return "nessun waypoint attivo"

        edge_id = waypoint.get("edge_id")
        edge = self.edge_by_id.get(edge_id)

        if edge:
            street = f"edge {edge['id']} tra {edge['from']} e {edge['to']}"
        else:
            street = f"edge={edge_id}"

        node_id = waypoint.get("node_id") or "-"
        kind = waypoint.get("kind") or "-"

        return (
            f"sto puntando wp {self.current_waypoint_index + 1}/{len(self.current_path)} "
            f"[{kind}] a ({waypoint['x']:.2f},{waypoint['y']:.2f}), "
            f"dist={distance:.2f} m, node={node_id}, {street}"
        )

    def log_navigation_snapshot(self, reason, waypoint=None, distance=None, force=False):
        now = self.get_clock().now()
        elapsed = (now - self.last_diag_time).nanoseconds / 1e9

        if not force and elapsed < self.diagnostic_log_period_sec:
            return

        self.last_diag_time = now

        if waypoint is None and self.current_waypoint_index < len(self.current_path):
            waypoint = self.current_path[self.current_waypoint_index]

        if distance is None and waypoint is not None:
            distance = self.distance_xy(
                self.current_x,
                self.current_y,
                waypoint["x"],
                waypoint["y"]
            )
        elif distance is None:
            distance = 0.0

        ctrl = waypoint.get("_last_control", {}) if waypoint else {}
        #return
        self.log_navigation_event(
            f"{reason} | stato={self.state.value} | "
            f"{self.describe_graph_position()} | "
            f"{self.describe_waypoint(waypoint, distance)} | "
            f"cmd: mode={ctrl.get('motion_mode', '?')}, "
            f"target=({ctrl.get('target_x', 0.0):.2f},{ctrl.get('target_y', 0.0):.2f}), "
            f"lane_err={ctrl.get('lane_error', 0.0):.2f}, "
            f"v={ctrl.get('linear_speed', 0.0):.2f}, "
            f"w={ctrl.get('angular_speed', 0.0):.2f}"
        )

    def maybe_log_diagnostics(self, waypoint, distance):
        self.log_navigation_snapshot("navigazione in corso", waypoint, distance)

    # ============================================================
    # UTILITY
    # ============================================================

    def compute_remaining_distance(self):
        if not self.current_path:
            return 0.0

        if self.current_waypoint_index >= len(self.current_path):
            return 0.0

        total = 0.0
        current = {
            "x": self.current_x,
            "y": self.current_y,
        }

        for i in range(self.current_waypoint_index, len(self.current_path)):
            wp = self.current_path[i]

            total += self.distance_xy(
                current["x"],
                current["y"],
                wp["x"],
                wp["y"]
            )

            current = wp

        return total

    def normalize_angle(self, angle):
        return math.atan2(
            math.sin(angle),
            math.cos(angle)
        )

    def distance_xy(self, x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1

        return math.sqrt(dx * dx + dy * dy)

    def clamp(self, value, min_value, max_value):
        return max(min_value, min(max_value, value))


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