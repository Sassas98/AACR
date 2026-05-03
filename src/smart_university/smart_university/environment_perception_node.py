import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json


class EnvironmentPerceptionNode(Node):

    def __init__(self):
        super().__init__('environment_perception_node')

        # Subscriber da Gazebo / simulazione
        self.shelves_sub = self.create_subscription(
            String,
            '/gazebo/shelves',
            self.shelves_callback,
            10
        )

        self.chargers_sub = self.create_subscription(
            String,
            '/gazebo/chargers',
            self.chargers_callback,
            10
        )

        # Publisher osservazioni
        self.publisher = self.create_publisher(
            String,
            '/environment_observations',
            10
        )

        # Stato interno (ultima osservazione)
        self.shelves = []
        self.chargers = []

        # Timer per pubblicare periodicamente
        self.timer = self.create_timer(0.5, self.publish_observations)

        self.get_logger().info('Environment Perception Node avviato')

    def shelves_callback(self, msg: String):
        try:
            self.shelves = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f'Errore parsing shelves: {e}')

    def chargers_callback(self, msg: String):
        try:
            self.chargers = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f'Errore parsing chargers: {e}')

    def publish_observations(self):
        observation = {
            "shelves": self.shelves,
            "chargers": self.chargers,
            "timestamp": self.get_clock().now().nanoseconds
        }

        msg = String()
        msg.data = json.dumps(observation)

        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = EnvironmentPerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()