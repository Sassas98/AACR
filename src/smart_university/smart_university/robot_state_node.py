import json
import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import String


class RobotStateNode(Node):

    def __init__(self):
        super().__init__('robot_state_node')

        self.declare_parameter('robot_id', 'robot_1')
        self.declare_parameter('publish_period', 0.5)

        self.robot_id = self.get_parameter('robot_id').value
        self.publish_period = float(self.get_parameter('publish_period').value)

        self.pose = {
            "x": 0.0,
            "y": 0.0,
            "theta": 0.0
        }

        self.battery = 100.0
        self.availability = "FREE"
        self.current_task_id = None
        self.current_cmd = None
        self.last_odom_timestamp = None
        self.last_battery_timestamp = None

        self.odom_sub = self.create_subscription(
            Odometry,
            f'/{self.robot_id}/odom',
            self.odom_callback,
            10
        )

        self.battery_sub = self.create_subscription(
            String,
            f'/{self.robot_id}/battery_state',
            self.battery_callback,
            10
        )

        self.local_status_sub = self.create_subscription(
            String,
            f'/{self.robot_id}/local_status',
            self.local_status_callback,
            10
        )

        self.status_pub = self.create_publisher(
            String,
            f'/{self.robot_id}/status',
            10
        )

        self.timer = self.create_timer(
            self.publish_period,
            self.publish_status
        )

        self.get_logger().info(
            f'Robot State Node avviato per {self.robot_id}'
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

        self.last_odom_timestamp = self.get_clock().now().nanoseconds

    def battery_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(
                f'Errore parsing /{self.robot_id}/battery_state: {e}'
            )
            return

        self.battery = float(data.get("battery", self.battery))
        self.last_battery_timestamp = self.get_clock().now().nanoseconds

    def local_status_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(
                f'Errore parsing /{self.robot_id}/local_status: {e}'
            )
            return

        self.availability = data.get("availability", self.availability)
        self.current_task_id = data.get("current_task_id", self.current_task_id)
        self.current_cmd = data.get("current_cmd", self.current_cmd)

    def publish_status(self):
        status = {
            "robot_id": self.robot_id,
            "pose": {
                "x": self.pose["x"],
                "y": self.pose["y"],
                "theta": self.pose["theta"]
            },
            "battery": self.battery,
            "availability": self.availability,
            "current_task_id": self.current_task_id,
            "current_cmd": self.current_cmd,
            "last_odom_timestamp": self.last_odom_timestamp,
            "last_battery_timestamp": self.last_battery_timestamp,
            "timestamp": self.get_clock().now().nanoseconds
        }

        msg = String()
        msg.data = json.dumps(status)

        self.status_pub.publish(msg)

    def quaternion_to_yaw(self, x, y, z, w):
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)


def main(args=None):
    rclpy.init(args=args)

    node = RobotStateNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()