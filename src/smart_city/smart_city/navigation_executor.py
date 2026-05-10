import json
import math
import os
import heapq
from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from smart_city_interfaces.action import NavigateToPose


class ExecutorState(Enum):
    IDLE = "IDLE"
    NAVIGATING = "NAVIGATING"


class NavigationExecutor(Node):

    def __init__(self):
        super().__init__("navigation_executor")

        self.declare_parameter("vehicle_id", "vehicle")
        self.declare_parameter("map_config_file", "config/city_map.json")

        self.declare_parameter("default_max_speed", 2.8)
        self.declare_parameter("linear_k", 2.0)
        self.declare_parameter("angular_k", 1.4)
        self.declare_parameter("max_angular_speed", 0.75)

        self.declare_parameter("waypoint_tolerance", 0.75)
        self.declare_parameter("target_tolerance", 0.85)
        self.declare_parameter("lane_offset_ratio", 0.5)

        self.vehicle_id = self.get_parameter("vehicle_id").value
        self.map_config_file = self.get_parameter("map_config_file").value

        self.default_max_speed = float(self.get_parameter("default_max_speed").value)
        self.linear_k = float(self.get_parameter("linear_k").value)
        self.angular_k = float(self.get_parameter("angular_k").value)
        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)

        self.waypoint_tolerance = float(self.get_parameter("waypoint_tolerance").value)
        self.target_tolerance = float(self.get_parameter("target_tolerance").value)
        self.lane_offset_ratio = float(self.get_parameter("lane_offset_ratio").value)

        self.state = ExecutorState.IDLE

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        self.nodes = {}
        self.edges = []
        self.edge_by_id = {}
        self.adj = {}
        self.lane_width = 1.2
        self.default_map_speed_limit = 1.4

        self.current_path = []
        self.current_waypoint_index = 0

        self.load_map()

        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

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

        self.action_server = ActionServer(
            self,
            NavigateToPose,
            "/navigation_executor/navigate_to_pose",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        self.get_logger().info(
            f"navigation_executor avviato per {self.vehicle_id}"
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
            f"mappa caricata: {len(self.nodes)} nodi, {len(self.edges)} archi"
        )

    # ------------------------------------------------------------------
    # CALLBACK
    # ------------------------------------------------------------------

    def on_odom(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation

        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)

        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def on_scan(self, msg):
        pass

    # ------------------------------------------------------------------
    # ACTION
    # ------------------------------------------------------------------

    def goal_callback(self, goal_request):
        self.get_logger().info(
            f"{self.vehicle_id}: ricevuto goal "
            f"mission={goal_request.mission_id}, "
            f"target=({goal_request.target_x:.2f}, {goal_request.target_y:.2f})"
        )

        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info(f"{self.vehicle_id}: cancellazione richiesta")
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        goal = goal_handle.request

        self.state = ExecutorState.NAVIGATING

        try:
            self.current_path = self.build_navigation_path(
                self.current_x,
                self.current_y,
                goal.target_x,
                goal.target_y
            )

            self.current_waypoint_index = 0

            self.get_logger().info(
                f"{self.vehicle_id}: path calcolato con {len(self.current_path)} waypoint"
            )

            for i, wp in enumerate(self.current_path):
                self.get_logger().info(
                    f"{self.vehicle_id}: wp[{i}] "
                    f"({wp['x']:.2f}, {wp['y']:.2f}) "
                    f"edge={wp.get('edge_id')} node={wp.get('node_id')}"
                )

        except Exception as ex:
            self.stop_vehicle()
            self.state = ExecutorState.IDLE
            goal_handle.abort()

            result = NavigateToPose.Result()
            result.success = False
            result.message = f"Errore calcolo path: {ex}"
            return result

        rate = self.create_rate(20)

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
                self.stop_vehicle()
                self.state = ExecutorState.IDLE
                goal_handle.succeed()

                result = NavigateToPose.Result()
                result.success = True
                result.message = "Target raggiunto"
                return result

            current_wp = self.current_path[self.current_waypoint_index]

            dx = current_wp["x"] - self.current_x
            dy = current_wp["y"] - self.current_y
            distance = math.sqrt(dx * dx + dy * dy)

            is_last = self.current_waypoint_index == len(self.current_path) - 1
            tolerance = self.target_tolerance if is_last else self.waypoint_tolerance

            if distance <= tolerance:
                self.current_waypoint_index += 1
                continue

            self.move_towards_waypoint(current_wp, goal.max_speed)

            feedback = NavigateToPose.Feedback()
            feedback.current_x = float(self.current_x)
            feedback.current_y = float(self.current_y)
            feedback.distance_remaining = float(self.compute_remaining_distance())
            feedback.status = (
                f"wp {self.current_waypoint_index + 1}/{len(self.current_path)}"
            )

            goal_handle.publish_feedback(feedback)

            rate.sleep()

        self.stop_vehicle()
        self.state = ExecutorState.IDLE
        goal_handle.abort()

        result = NavigateToPose.Result()
        result.success = False
        result.message = "Navigazione interrotta"
        return result

    # ------------------------------------------------------------------
    # PATH PLANNING SU GRAFO
    # ------------------------------------------------------------------

    def build_navigation_path(self, start_x, start_y, target_x, target_y):
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

                cost += self.distance_to_node_on_edge(start_projection, s)
                cost += self.distance_to_node_on_edge(target_projection, t)

                if node_path and cost < best_cost:
                    best_cost = cost
                    best_node_path = node_path

        if not best_node_path:
            raise RuntimeError("nessun path trovato sul grafo")

        waypoints = []

        start_lane = self.project_point_to_lane(
            start_projection,
            best_node_path[0]
        )

        waypoints.append({
            "x": start_lane["x"],
            "y": start_lane["y"],
            "edge_id": start_edge["id"],
            "node_id": None,
            "speed_limit": start_edge["speed_limit"]
        })

        for i in range(len(best_node_path) - 1):
            current_node_id = best_node_path[i]
            next_node_id = best_node_path[i + 1]

            current_node = self.nodes[current_node_id]
            edge = self.get_edge_between(current_node_id, next_node_id)

            lane_point = self.node_to_lane_point(
                current_node_id,
                next_node_id,
                approach=False
            )

            waypoints.append({
                "x": lane_point["x"],
                "y": lane_point["y"],
                "edge_id": edge["id"],
                "node_id": current_node_id,
                "speed_limit": edge["speed_limit"]
            })

            exit_point = self.node_to_lane_point(
                next_node_id,
                current_node_id,
                approach=True
            )

            waypoints.append({
                "x": exit_point["x"],
                "y": exit_point["y"],
                "edge_id": edge["id"],
                "node_id": next_node_id,
                "speed_limit": edge["speed_limit"]
            })

        target_lane = self.project_point_to_lane(
            target_projection,
            best_node_path[-1]
        )

        waypoints.append({
            "x": target_lane["x"],
            "y": target_lane["y"],
            "edge_id": target_edge["id"],
            "node_id": None,
            "speed_limit": target_edge["speed_limit"]
        })

        return self.simplify_waypoints(waypoints)

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
                    heapq.heappush(
                        queue,
                        (cost + length, neighbor_id, new_path)
                    )

        return None, float("inf")

    def find_nearest_edge_projection(self, x, y):
        best = None
        best_distance = float("inf")

        for edge in self.edges:
            a = self.nodes[edge["from"]]
            b = self.nodes[edge["to"]]

            proj = self.project_point_on_segment(
                x,
                y,
                a["x"],
                a["y"],
                b["x"],
                b["y"]
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

        return best

    def project_point_on_segment(self, px, py, ax, ay, bx, by):
        dx = bx - ax
        dy = by - ay

        denom = dx * dx + dy * dy

        if denom <= 0.000001:
            return {"x": ax, "y": ay, "t": 0.0}

        t = ((px - ax) * dx + (py - ay) * dy) / denom
        t = max(0.0, min(1.0, t))

        return {
            "x": ax + t * dx,
            "y": ay + t * dy,
            "t": t
        }

    def project_point_to_lane(self, projection, destination_node_id):
        edge = projection["edge"]

        a = self.nodes[edge["from"]]
        b = self.nodes[edge["to"]]

        if destination_node_id == edge["to"]:
            from_node = a
            to_node = b
        else:
            from_node = b
            to_node = a

        lane_x, lane_y = self.apply_right_lane_offset(
            projection["x"],
            projection["y"],
            from_node["x"],
            from_node["y"],
            to_node["x"],
            to_node["y"]
        )

        return {"x": lane_x, "y": lane_y}

    def node_to_lane_point(self, node_id, other_node_id, approach):
        node = self.nodes[node_id]
        other = self.nodes[other_node_id]

        dx = other["x"] - node["x"]
        dy = other["y"] - node["y"]

        length = math.sqrt(dx * dx + dy * dy)

        if length <= 0.000001:
            return {"x": node["x"], "y": node["y"]}

        ux = dx / length
        uy = dy / length

        intersection_clearance = 2.2

        if approach:
            base_x = node["x"] + ux * intersection_clearance
            base_y = node["y"] + uy * intersection_clearance
        else:
            base_x = node["x"] + ux * intersection_clearance
            base_y = node["y"] + uy * intersection_clearance

        lane_x, lane_y = self.apply_right_lane_offset(
            base_x,
            base_y,
            node["x"],
            node["y"],
            other["x"],
            other["y"]
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

        right_x = uy
        right_y = -ux

        offset = self.lane_width * self.lane_offset_ratio

        return x + right_x * offset, y + right_y * offset

    def distance_to_node_on_edge(self, projection, node_id):
        node = self.nodes[node_id]
        return self.distance_xy(
            projection["x"],
            projection["y"],
            node["x"],
            node["y"]
        )

    def get_edge_between(self, a, b):
        for edge in self.edges:
            if edge["from"] == a and edge["to"] == b:
                return edge
            if edge["from"] == b and edge["to"] == a:
                return edge

        raise RuntimeError(f"nessun edge tra {a} e {b}")

    def simplify_waypoints(self, waypoints):
        if not waypoints:
            return []

        result = [waypoints[0]]

        for wp in waypoints[1:]:
            last = result[-1]
            d = self.distance_xy(last["x"], last["y"], wp["x"], wp["y"])

            if d >= 0.4:
                result.append(wp)

        return result

    # ------------------------------------------------------------------
    # CONTROLLO MOVIMENTO
    # ------------------------------------------------------------------

    def move_towards_waypoint(self, waypoint, requested_max_speed):
        dx = waypoint["x"] - self.current_x
        dy = waypoint["y"] - self.current_y

        target_angle = math.atan2(dy, dx)
        angle_error = self.normalize_angle(target_angle - self.current_yaw)

        distance = math.sqrt(dx * dx + dy * dy)

        edge_speed_limit = float(
            waypoint.get("speed_limit", self.default_map_speed_limit)
        )

        max_speed = requested_max_speed
        if max_speed <= 0.0:
            max_speed = self.default_max_speed

        max_speed = min(max_speed, edge_speed_limit * 2.0)

        angular_speed = self.angular_k * angle_error
        angular_speed = max(
            -self.max_angular_speed,
            min(self.max_angular_speed, angular_speed)
        )

        abs_error = abs(angle_error)

        if abs_error > 2.6:
            linear_speed = 0.15
        elif abs_error > 1.2:
            linear_speed = 0.45
        elif abs_error > 0.55:
            linear_speed = 0.75
        else:
            linear_speed = min(max_speed, self.linear_k * distance)

        if distance > 1.5 and linear_speed < 0.65:
            linear_speed = 0.65

        self.get_logger().info(
            f"{self.vehicle_id}: "
            f"wp={self.current_waypoint_index + 1}/{len(self.current_path)} "
            f"target=({waypoint['x']:.2f},{waypoint['y']:.2f}) "
            f"pos=({self.current_x:.2f},{self.current_y:.2f}) "
            f"yaw={self.current_yaw:.2f} "
            f"angle_error={angle_error:.2f} "
            f"linear={linear_speed:.2f} angular={angular_speed:.2f}"
        )

        cmd = Twist()
        cmd.linear.x = float(linear_speed)
        cmd.angular.z = float(angular_speed)

        self.cmd_vel_pub.publish(cmd)

    def stop_vehicle(self):
        self.cmd_vel_pub.publish(Twist())

    # ------------------------------------------------------------------
    # UTILITY
    # ------------------------------------------------------------------

    def compute_remaining_distance(self):
        if not self.current_path:
            return 0.0

        if self.current_waypoint_index >= len(self.current_path):
            return 0.0

        total = 0.0

        current = {
            "x": self.current_x,
            "y": self.current_y
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
        while angle > math.pi:
            angle -= 2.0 * math.pi

        while angle < -math.pi:
            angle += 2.0 * math.pi

        return angle

    def distance_xy(self, x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        return math.sqrt(dx * dx + dy * dy)


def main(args=None):
    rclpy.init(args=args)

    node = NavigationExecutor()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.stop_vehicle()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()