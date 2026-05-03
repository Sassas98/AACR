import math
import time
import rclpy

from rclpy.action import ActionServer
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from nav2_msgs.action import NavigateToPose


class NavigatorNode(Node):

    def __init__(self):
        super().__init__('navigator_node')

        self.declare_parameter('robot_id', 'robot_1')
        self.declare_parameter('goal_tolerance', 0.25)
        self.declare_parameter('angle_tolerance', 0.15)
        self.declare_parameter('linear_speed', 0.25)
        self.declare_parameter('angular_speed', 0.8)
        self.declare_parameter('obstacle_distance', 0.45)
        self.declare_parameter('control_frequency', 10.0)
        self.declare_parameter('navigation_timeout_sec', 60.0)

        self.robot_id = self.get_parameter('robot_id').value
        self.goal_tolerance = float(self.get_parameter('goal_tolerance').value)
        self.angle_tolerance = float(self.get_parameter('angle_tolerance').value)
        self.linear_speed = float(self.get_parameter('linear_speed').value)
        self.angular_speed = float(self.get_parameter('angular_speed').value)
        self.obstacle_distance = float(self.get_parameter('obstacle_distance').value)
        self.control_frequency = float(self.get_parameter('control_frequency').value)
        self.navigation_timeout_sec = float(
            self.get_parameter('navigation_timeout_sec').value
        )

        self.pose = {
            "x": 0.0,
            "y": 0.0,
            "theta": 0.0
        }

        self.front_obstacle = False
        self.latest_scan_time = None
        self.latest_odom_time = None

        self.odom_sub = self.create_subscription(
            Odometry,
            f'/{self.robot_id}/odom',
            self.odom_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            f'/{self.robot_id}/scan',
            self.scan_callback,
            10
        )

        self.cmd_vel_pub = self.create_publisher(
            Twist,
            f'/{self.robot_id}/cmd_vel',
            10
        )

        self.action_server = ActionServer(
            self,
            NavigateToPose,
            f'/{self.robot_id}/navigate_to_pose',
            self.execute_navigation
        )

        self.get_logger().info(
            f'Navigator Node avviato per {self.robot_id}'
        )

    def odom_callback(self, msg: Odometry):
        self.pose["x"] = float(msg.pose.pose.position.x)
        self.pose["y"] = float(msg.pose.pose.position.y)
        self.pose["theta"] = self.quaternion_to_yaw(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        )

        self.latest_odom_time = self.get_clock().now().nanoseconds

    def scan_callback(self, msg: LaserScan):
        if not msg.ranges:
            self.front_obstacle = False
            return

        center_index = len(msg.ranges) // 2
        window = max(1, len(msg.ranges) // 12)

        front_ranges = msg.ranges[
            max(0, center_index - window):
            min(len(msg.ranges), center_index + window)
        ]

        front_valid = [
            r for r in front_ranges
            if not math.isinf(r) and not math.isnan(r)
        ]

        self.front_obstacle = (
            len(front_valid) > 0 and min(front_valid) < self.obstacle_distance
        )

        self.latest_scan_time = self.get_clock().now().nanoseconds

    def execute_navigation(self, goal_handle):
        target_pose = goal_handle.request.pose.pose

        target_x = float(target_pose.position.x)
        target_y = float(target_pose.position.y)

        self.get_logger().info(
            f'Navigazione richiesta: {self.robot_id} -> '
            f'x={target_x:.2f}, y={target_y:.2f}'
        )

        feedback_msg = NavigateToPose.Feedback()
        result = NavigateToPose.Result()

        start_time = time.time()
        period = 1.0 / self.control_frequency

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                self.stop_robot()
                goal_handle.canceled()
                self.get_logger().warn('Navigazione cancellata')
                return result

            elapsed = time.time() - start_time

            if elapsed > self.navigation_timeout_sec:
                self.stop_robot()
                goal_handle.abort()
                self.get_logger().warn('Navigazione abortita: timeout')
                return result

            dx = target_x - self.pose["x"]
            dy = target_y - self.pose["y"]
            distance = math.sqrt(dx * dx + dy * dy)

            feedback_msg.distance_remaining = float(distance)
            goal_handle.publish_feedback(feedback_msg)

            if distance <= self.goal_tolerance:
                self.stop_robot()
                goal_handle.succeed()
                self.get_logger().info('Navigazione completata')
                return result

            if self.front_obstacle:
                self.stop_robot()
                goal_handle.abort()
                self.get_logger().warn('Navigazione abortita: ostacolo frontale')
                return result

            desired_theta = math.atan2(dy, dx)
            angle_error = self.normalize_angle(
                desired_theta - self.pose["theta"]
            )

            cmd = Twist()

            if abs(angle_error) > self.angle_tolerance:
                cmd.linear.x = 0.0
                cmd.angular.z = (
                    self.angular_speed
                    if angle_error > 0.0
                    else -self.angular_speed
                )
            else:
                cmd.linear.x = self.linear_speed
                cmd.angular.z = max(
                    -self.angular_speed,
                    min(self.angular_speed, angle_error)
                )

            self.cmd_vel_pub.publish(cmd)

            time.sleep(period)

        self.stop_robot()
        goal_handle.abort()
        return result

    def stop_robot(self):
        self.cmd_vel_pub.publish(Twist())

    def quaternion_to_yaw(self, x, y, z, w):
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi

        while angle < -math.pi:
            angle += 2.0 * math.pi

        return angle


def main(args=None):
    rclpy.init(args=args)

    node = NavigatorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.stop_robot()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()