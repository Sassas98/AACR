import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class WorldStateAdapterNode(Node):

    def __init__(self):
        super().__init__('world_state_adapter_node')

        self.world_events_sub = self.create_subscription(
            String,
            '/world_events',
            self.world_events_callback,
            10
        )

        self.update_shelf_pub = self.create_publisher(
            String,
            '/gazebo/update_shelf',
            10
        )

        self.update_charger_pub = self.create_publisher(
            String,
            '/gazebo/update_charger',
            10
        )

        self.get_logger().info('World State Adapter Node avviato')

    def world_events_callback(self, msg: String):
        try:
            event = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f'Errore parsing /world_events: {e}')
            return

        event_type = event.get("event_type")

        if event_type == "SHELF_CLEANED":
            self.handle_shelf_cleaned(event)

        elif event_type == "CHARGER_OCCUPIED":
            self.handle_charger_occupied(event)

        elif event_type == "CHARGER_RELEASED":
            self.handle_charger_released(event)

        else:
            self.get_logger().warn(f'Evento sconosciuto: {event_type}')

    def handle_shelf_cleaned(self, event):
        shelf_id = event.get("target_id")

        if not shelf_id:
            self.get_logger().warn('Evento SHELF_CLEANED senza target_id')
            return

        command = {
            "id": shelf_id,
            "dirty": False,
            "visual_state": "clean"
        }

        self.publish_shelf_update(command)

        self.get_logger().info(f'Scaffale aggiornato in Gazebo: {shelf_id} clean')

    def handle_charger_occupied(self, event):
        charger_id = event.get("target_id")
        robot_id = event.get("robot_id")

        if not charger_id:
            self.get_logger().warn('Evento CHARGER_OCCUPIED senza target_id')
            return

        command = {
            "id": charger_id,
            "occupied": True,
            "occupied_by": robot_id
        }

        self.publish_charger_update(command)

        self.get_logger().info(f'Punto ricarica occupato in Gazebo: {charger_id}')

    def handle_charger_released(self, event):
        charger_id = event.get("target_id")

        if not charger_id:
            self.get_logger().warn('Evento CHARGER_RELEASED senza target_id')
            return

        command = {
            "id": charger_id,
            "occupied": False,
            "occupied_by": None
        }

        self.publish_charger_update(command)

        self.get_logger().info(f'Punto ricarica liberato in Gazebo: {charger_id}')

    def publish_shelf_update(self, command):
        msg = String()
        msg.data = json.dumps(command)
        self.update_shelf_pub.publish(msg)

    def publish_charger_update(self, command):
        msg = String()
        msg.data = json.dumps(command)
        self.update_charger_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    node = WorldStateAdapterNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()