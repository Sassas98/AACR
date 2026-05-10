import json
import math
import os
import heapq
import time
from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from smart_city_interfaces.action import NavigateToPose


class ExecutorState(str, Enum):
    IDLE = "IDLE"
    NAVIGATING = "NAVIGATING"
    BLOCKED = "BLOCKED"
    WAITING_TRAFFIC_LIGHT = "WAITING_TRAFFIC_LIGHT"
    GOAL_REACHED = "GOAL_REACHED"
    FAILED = "FAILED"


class NavigationExecutor(Node):

    def __init__(self):
        super().__init__("navigation_executor")

        self.declare_parameter("vehicle_id", "bus_1")
        self.declare_parameter("map_config_file", "config/city_map.json")
        self.declare_parameter("position_tolerance", 0.35)
        self.declare_parameter("waypoint_tolerance", 0.45)
        self.declare_parameter("obstacle_distance_threshold", 0.8)
        self.declare_parameter("linear_k", 0.8)
        self.declare_parameter("angular_k", 0.8)
        self.declare_parameter("default_max_speed", 1.0)
        self.declare_parameter("traffic_light_distance", 2.0)

        self.vehicle_id = self.get_parameter("vehicle_id").value
        self.position_tolerance = float(self.get_parameter("position_tolerance").value)
        self.waypoint_tolerance = float(self.get_parameter("waypoint_tolerance").value)
        self.obstacle_distance_threshold = float(self.get_parameter("obstacle_distance_threshold").value)
        self.linear_k = float(self.get_parameter("linear_k").value)
        self.angular_k = float(self.get_parameter("angular_k").value)
        self.default_max_speed = float(self.get_parameter("default_max_speed").value)
        self.traffic_light_distance = float(self.get_parameter("traffic_light_distance").value)

        self.state = ExecutorState.IDLE

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.last_scan = None

        self.nodes = {}
        self.edges = []
        self.adj = {}
        self.lane_width = 1.2

        self.load_map()

        self.current_route_points = []
        self.current_route_edges = []
        self.route_index = 0

        self.traffic_lights = {}
        self.priority_sent = set()

        self.cmd_vel_pub = self.create_publisher(
            Twist,
            "/cmd_vel",
            10
        )

        self.priority_request_pub = self.create_publisher(
            String,
            "/traffic_light/priority_request",
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            "/odom",
            self.on_odom,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            "/scan",
            self.on_scan,
            10
        )

        self.traffic_light_status_sub = self.create_subscription(
            String,
            "/traffic_light/status",
            self.on_traffic_light_status,
            10
        )

        self.action_server = ActionServer(
            self,
            NavigateToPose,
            "/navigation_executor/navigate_to_pose",
            self.execute_callback
        )

        self.get_logger().info(
            f"navigation_executor avviato per {self.vehicle_id}"
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

        self.lane_width = float(data.get("lane_width", 1.2))

        for node in data["nodes"]:
            self.nodes[node["id"]] = {
                "id": node["id"],
                "x": float(node["x"]),
                "y": float(node["y"])
            }

        self.edges = data["edges"]

        for node_id in self.nodes:
            self.adj[node_id] = []

        for edge in self.edges:
            a = edge["from"]
            b = edge["to"]
            speed_limit = float(edge.get("speed_limit", self.default_max_speed))
            distance = self.distance_between_nodes(a, b)

            self.adj[a].append({
                "to": b,
                "edge_id": edge["id"],
                "distance": distance,
                "speed_limit": speed_limit
            })

            self.adj[b].append({
                "to": a,
                "edge_id": edge["id"],
                "distance": distance,
                "speed_limit": speed_limit
            })

    def node_degree(self, node_id):
        return len(self.adj.get(node_id, []))

    def has_traffic_light(self, node_id):
        return self.node_degree(node_id) >= 3

    # ------------------------------------------------------------------
    # SENSORI
    # ------------------------------------------------------------------

    def on_odom(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation

        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)

        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def on_scan(self, msg):
        self.last_scan = msg

    def on_traffic_light_status(self, msg):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        node_id = data.get("node_id")

        if node_id is None:
            return

        self.traffic_lights[node_id] = data

    # ------------------------------------------------------------------
    # ACTION
    # ------------------------------------------------------------------

    def execute_callback(self, goal_handle):
        goal = goal_handle.request

        if goal.vehicle_id != self.vehicle_id:
            goal_handle.abort()

            result = NavigateToPose.Result()
            result.success = False
            result.message = f"Goal destinato a {goal.vehicle_id}, non a {self.vehicle_id}"
            return result

        target_edge = self.find_edge_containing_point(goal.target_x, goal.target_y)

        if target_edge is None:
            goal_handle.abort()

            result = NavigateToPose.Result()
            result.success = False
            result.message = "Target non appartenente ad alcuna strada"
            return result

        start_edge = self.find_edge_containing_point(self.current_x, self.current_y)

        if start_edge is None:
            start_node = self.find_nearest_node(self.current_x, self.current_y)
        else:
            start_node = self.closest_endpoint_of_edge(start_edge, self.current_x, self.current_y)

        target_entry_node = self.closest_endpoint_of_edge(
            target_edge,
            goal.target_x,
            goal.target_y
        )

        node_route = self.compute_node_route(start_node, target_entry_node)

        if len(node_route) == 0:
            goal_handle.abort()

            result = NavigateToPose.Result()
            result.success = False
            result.message = "Nessun percorso trovato"
            return result

        self.current_route_points = self.build_route_points(
            node_route,
            goal.target_x,
            goal.target_y
        )

        self.current_route_edges = self.build_route_edges(node_route, target_edge)
        self.route_index = 0
        self.priority_sent.clear()

        self.state = ExecutorState.NAVIGATING

        feedback = NavigateToPose.Feedback()

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                self.stop_vehicle()
                goal_handle.canceled()

                result = NavigateToPose.Result()
                result.success = False
                result.message = "Navigazione cancellata"

                self.state = ExecutorState.IDLE
                return result

            target_point = self.current_route_points[self.route_index]
            distance = self.distance_to_point(target_point["x"], target_point["y"])

            feedback.current_x = self.current_x
            feedback.current_y = self.current_y
            feedback.distance_remaining = self.remaining_route_distance()
            feedback.status = self.state.value

            goal_handle.publish_feedback(feedback)

            is_final_point = self.route_index == len(self.current_route_points) - 1

            if is_final_point and distance <= self.position_tolerance:
                self.stop_vehicle()
                goal_handle.succeed()

                result = NavigateToPose.Result()
                result.success = True
                result.message = "Target raggiunto"

                self.state = ExecutorState.IDLE
                return result

            if not is_final_point and distance <= self.waypoint_tolerance:
                self.route_index += 1
                continue

            if self.has_front_obstacle():
                self.state = ExecutorState.BLOCKED
                self.stop_vehicle()
                time.sleep(0.1)
                continue

            upcoming_node = self.get_upcoming_intersection_node()

            if upcoming_node is not None:
                self.send_priority_request_if_needed(goal, upcoming_node)

                if self.must_wait_for_traffic_light(upcoming_node):
                    self.state = ExecutorState.WAITING_TRAFFIC_LIGHT
                    self.stop_vehicle()
                    time.sleep(0.1)
                    continue

            self.state = ExecutorState.NAVIGATING
            self.move_towards_point(target_point, goal.max_speed)

            time.sleep(0.05)

        self.stop_vehicle()
        goal_handle.abort()

        result = NavigateToPose.Result()
        result.success = False
        result.message = "ROS shutdown"

        return result

    # ------------------------------------------------------------------
    # COSTRUZIONE PERCORSO
    # ------------------------------------------------------------------

    def build_route_points(self, node_route, target_x, target_y):
        points = []

        for node_id in node_route:
            node = self.nodes[node_id]
            points.append({
                "kind": "NODE_PASSAGE",
                "node_id": node_id,
                "x": node["x"],
                "y": node["y"]
            })

        points.append({
            "kind": "FINAL_TARGET",
            "node_id": None,
            "x": target_x,
            "y": target_y
        })

        return points

    def build_route_edges(self, node_route, target_edge):
        route_edges = []

        for i in range(len(node_route) - 1):
            edge = self.get_edge_between(node_route[i], node_route[i + 1])
            route_edges.append(edge)

        route_edges.append(target_edge)

        return route_edges

    def compute_node_route(self, start_node, target_node):
        queue = []
        heapq.heappush(queue, (0.0, start_node))

        distances = {node_id: float("inf") for node_id in self.nodes}
        previous = {node_id: None for node_id in self.nodes}

        distances[start_node] = 0.0

        while queue:
            current_distance, current = heapq.heappop(queue)

            if current == target_node:
                break

            if current_distance > distances[current]:
                continue

            for edge in self.adj[current]:
                neighbor = edge["to"]
                new_distance = current_distance + edge["distance"]

                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous[neighbor] = current
                    heapq.heappush(queue, (new_distance, neighbor))

        if distances[target_node] == float("inf"):
            return []

        route = []
        current = target_node

        while current is not None:
            route.append(current)
            current = previous[current]

        route.reverse()
        return route

    # ------------------------------------------------------------------
    # STRADE / ARCHI
    # ------------------------------------------------------------------

    def find_edge_containing_point(self, x, y):
        best_edge = None
        best_distance = float("inf")

        max_distance_from_centerline = self.lane_width

        for edge in self.edges:
            a = self.nodes[edge["from"]]
            b = self.nodes[edge["to"]]

            distance = self.distance_point_to_segment(
                x,
                y,
                a["x"],
                a["y"],
                b["x"],
                b["y"]
            )

            if distance < best_distance:
                best_distance = distance
                best_edge = edge

        if best_distance <= max_distance_from_centerline:
            return best_edge

        return None

    def closest_endpoint_of_edge(self, edge, x, y):
        a = self.nodes[edge["from"]]
        b = self.nodes[edge["to"]]

        da = math.sqrt((x - a["x"]) ** 2 + (y - a["y"]) ** 2)
        db = math.sqrt((x - b["x"]) ** 2 + (y - b["y"]) ** 2)

        if da <= db:
            return edge["from"]

        return edge["to"]

    def get_edge_between(self, a, b):
        for edge in self.edges:
            if edge["from"] == a and edge["to"] == b:
                return edge

            if edge["from"] == b and edge["to"] == a:
                return edge

        return None

    def distance_between_nodes(self, a, b):
        na = self.nodes[a]
        nb = self.nodes[b]

        return math.sqrt(
            (nb["x"] - na["x"]) ** 2 +
            (nb["y"] - na["y"]) ** 2
        )

    def distance_point_to_segment(self, px, py, ax, ay, bx, by):
        abx = bx - ax
        aby = by - ay

        apx = px - ax
        apy = py - ay

        ab_len_sq = abx * abx + aby * aby

        if ab_len_sq == 0:
            return math.sqrt((px - ax) ** 2 + (py - ay) ** 2)

        t = (apx * abx + apy * aby) / ab_len_sq
        t = max(0.0, min(1.0, t))

        closest_x = ax + t * abx
        closest_y = ay + t * aby

        return math.sqrt(
            (px - closest_x) ** 2 +
            (py - closest_y) ** 2
        )

    # ------------------------------------------------------------------
    # SEMAFORI
    # ------------------------------------------------------------------

    def get_upcoming_intersection_node(self):
        if self.route_index >= len(self.current_route_points):
            return None

        point = self.current_route_points[self.route_index]

        if point["kind"] != "NODE_PASSAGE":
            return None

        node_id = point["node_id"]

        if not self.has_traffic_light(node_id):
            return None

        distance = self.distance_to_point(point["x"], point["y"])

        if distance <= self.traffic_light_distance:
            return node_id

        return None

    def send_priority_request_if_needed(self, goal, node_id):
        if goal.target_type == "PARKING":
            return

        if node_id in self.priority_sent:
            return

        movement = self.get_current_movement_through_node(node_id)

        payload = {
            "vehicle_id": self.vehicle_id,
            "mission_id": goal.mission_id,
            "target_type": goal.target_type,
            "node_id": node_id,
            "from_node_id": movement["from"],
            "to_node_id": movement["to"],
            "priority": self.compute_priority(goal)
        }

        msg = String()
        msg.data = json.dumps(payload)

        self.priority_request_pub.publish(msg)
        self.priority_sent.add(node_id)

    def must_wait_for_traffic_light(self, node_id):
        status = self.traffic_lights.get(node_id)

        if status is None:
            return False

        movement = self.get_current_movement_through_node(node_id)

        allowed_movements = status.get("allowed_movements", [])

        for allowed in allowed_movements:
            if allowed.get("from") == movement["from"] and allowed.get("to") == movement["to"]:
                return False

        return True

    def get_current_movement_through_node(self, node_id):
        previous_node = None
        next_node = None

        for i, point in enumerate(self.current_route_points):
            if point.get("node_id") == node_id:
                if i > 0:
                    previous_node = self.current_route_points[i - 1].get("node_id")

                if i + 1 < len(self.current_route_points):
                    next_node = self.current_route_points[i + 1].get("node_id")

                break

        return {
            "from": previous_node,
            "to": next_node
        }

    def compute_priority(self, goal):
        if goal.target_type == "BUS_STOP":
            return 3

        if goal.target_type == "TAXI_PICKUP":
            return 3

        if goal.target_type == "TAXI_DROPOFF":
            return 2

        if goal.target_type == "WAYPOINT":
            return 1

        return 1

    # ------------------------------------------------------------------
    # MOVIMENTO
    # ------------------------------------------------------------------

    def move_towards_point(self, point, requested_max_speed):
        dx = point["x"] - self.current_x
        dy = point["y"] - self.current_y

        target_angle = math.atan2(dy, dx)
        angle_error = self.normalize_angle(target_angle - self.current_yaw)

        distance = math.sqrt(dx * dx + dy * dy)

        edge_speed_limit = self.get_current_speed_limit()

        max_speed = requested_max_speed

        if max_speed <= 0.0:
            max_speed = self.default_max_speed

        max_speed = min(max_speed, edge_speed_limit)

        linear_speed = min(max_speed, self.linear_k * distance)
        max_angular_speed = 1.0
        angular_speed = max(
            -max_angular_speed,
            min(max_angular_speed, self.angular_k * angle_error)
        )

        if abs(angle_error) > 0.7:
            linear_speed = 0.15

        cmd = Twist()
        cmd.linear.x = linear_speed
        cmd.angular.z = angular_speed

        self.cmd_vel_pub.publish(cmd)

    def get_current_speed_limit(self):
        if self.route_index >= len(self.current_route_edges):
            return self.default_max_speed

        edge = self.current_route_edges[self.route_index]

        if edge is None:
            return self.default_max_speed

        return float(edge.get("speed_limit", self.default_max_speed))

    def stop_vehicle(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)

    # ------------------------------------------------------------------
    # OSTACOLI
    # ------------------------------------------------------------------

    def has_front_obstacle(self):
        if self.last_scan is None:
            return False

        ranges = list(self.last_scan.ranges)

        if len(ranges) == 0:
            return False

        center = len(ranges) // 2
        window = ranges[max(0, center - 10):min(len(ranges), center + 10)]

        valid = [
            r for r in window
            if not math.isnan(r) and not math.isinf(r)
        ]

        if not valid:
            return False

        return min(valid) < self.obstacle_distance_threshold

    # ------------------------------------------------------------------
    # UTILITY
    # ------------------------------------------------------------------

    def find_nearest_node(self, x, y):
        best_node = None
        best_distance = float("inf")

        for node_id, node in self.nodes.items():
            distance = math.sqrt(
                (x - node["x"]) ** 2 +
                (y - node["y"]) ** 2
            )

            if distance < best_distance:
                best_distance = distance
                best_node = node_id

        return best_node

    def distance_to_point(self, x, y):
        return math.sqrt(
            (x - self.current_x) ** 2 +
            (y - self.current_y) ** 2
        )

    def remaining_route_distance(self):
        if self.route_index >= len(self.current_route_points):
            return 0.0

        current_target = self.current_route_points[self.route_index]

        total = self.distance_to_point(
            current_target["x"],
            current_target["y"]
        )

        for i in range(self.route_index, len(self.current_route_points) - 1):
            a = self.current_route_points[i]
            b = self.current_route_points[i + 1]

            total += math.sqrt(
                (b["x"] - a["x"]) ** 2 +
                (b["y"] - a["y"]) ** 2
            )

        return total

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi

        while angle < -math.pi:
            angle += 2.0 * math.pi

        return angle


def main(args=None):
    rclpy.init(args=args)

    node = NavigationExecutor()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()