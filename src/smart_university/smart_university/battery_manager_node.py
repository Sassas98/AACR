import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class BatteryManagerNode(Node):

    def __init__(self):
        super().__init__('battery_manager_node')

        self.declare_parameter('robot_id', 'robot_1')
        self.declare_parameter('initial_battery', 100.0)
        self.declare_parameter('low_threshold', 25.0)
        self.declare_parameter('critical_threshold', 10.0)
        self.declare_parameter('discharge_rate_idle', 0.02)
        self.declare_parameter('discharge_rate_working', 0.08)
        self.declare_parameter('charge_rate', 0.4)
        self.declare_parameter('publish_period', 1.0)

        self.robot_id = self.get_parameter('robot_id').value
        self.battery = float(self.get_parameter('initial_battery').value)
        self.low_threshold = float(self.get_parameter('low_threshold').value)
        self.critical_threshold = float(self.get_parameter('critical_threshold').value)
        self.discharge_rate_idle = float(self.get_parameter('discharge_rate_idle').value)
        self.discharge_rate_working = float(self.get_parameter('discharge_rate_working').value)
        self.charge_rate = float(self.get_parameter('charge_rate').value)
        self.publish_period = float(self.get_parameter('publish_period').value)

        self.availability = "FREE"
        self.current_cmd = None
        self.is_charging = False

        self.local_status_sub = self.create_subscription(
            String,
            f'/{self.robot_id}/local_status',
            self.local_status_callback,
            10
        )

        self.charging_state_sub = self.create_subscription(
            String,
            f'/{self.robot_id}/charging_state',
            self.charging_state_callback,
            10
        )

        self.battery_state_pub = self.create_publisher(
            String,
            f'/{self.robot_id}/battery_state',
            10
        )

        self.timer = self.create_timer(
            self.publish_period,
            self.update_and_publish_battery
        )

        self.get_logger().info(
            f'Battery Manager Node avviato per {self.robot_id}'
        )

    def local_status_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(
                f'Errore parsing /{self.robot_id}/local_status: {e}'
            )
            return

        self.availability = data.get("availability", self.availability)
        self.current_cmd = data.get("current_cmd", self.current_cmd)

    def charging_state_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(
                f'Errore parsing /{self.robot_id}/charging_state: {e}'
            )
            return

        self.is_charging = bool(data.get("is_charging", self.is_charging))

    def update_and_publish_battery(self):
        if self.is_charging:
            self.battery += self.charge_rate
        else:
            if self.availability == "BUSY":
                self.battery -= self.discharge_rate_working
            else:
                self.battery -= self.discharge_rate_idle

        self.battery = max(0.0, min(100.0, self.battery))

        if self.battery <= self.critical_threshold:
            level = "CRITICAL"
        elif self.battery <= self.low_threshold:
            level = "LOW"
        else:
            level = "OK"

        state = {
            "robot_id": self.robot_id,
            "battery": self.battery,
            "level": level,
            "is_charging": self.is_charging,
            "timestamp": self.get_clock().now().nanoseconds
        }

        msg = String()
        msg.data = json.dumps(state)

        self.battery_state_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    node = BatteryManagerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()