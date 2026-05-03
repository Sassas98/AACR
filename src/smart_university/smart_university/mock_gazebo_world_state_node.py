import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MockGazeboWorldStateNode(Node):

    def __init__(self):
        super().__init__('mock_gazebo_world_state_node')

        self.shelves = {
            "shelf_1": {"id": "shelf_1", "x": 4.0, "y": 2.0, "theta": 0.0, "dirty": True},
            "shelf_2": {"id": "shelf_2", "x": 4.0, "y": -2.0, "theta": 0.0, "dirty": True},
            "shelf_3": {"id": "shelf_3", "x": -3.0, "y": 2.0, "theta": 0.0, "dirty": True},
        }

        self.chargers = {
            "charger_1": {"id": "charger_1", "x": -6.0, "y": -4.0, "theta": 0.0, "occupied": False},
            "charger_2": {"id": "charger_2", "x": 0.0, "y": -4.0, "theta": 0.0, "occupied": False},
            "charger_3": {"id": "charger_3", "x": 6.0, "y": -4.0, "theta": 0.0, "occupied": False},
        }

        self.shelves_pub = self.create_publisher(
            String,
            '/gazebo/shelves',
            10
        )

        self.chargers_pub = self.create_publisher(
            String,
            '/gazebo/chargers',
            10
        )

        self.update_shelf_sub = self.create_subscription(
            String,
            '/gazebo/update_shelf',
            self.update_shelf_callback,
            10
        )

        self.update_charger_sub = self.create_subscription(
            String,
            '/gazebo/update_charger',
            self.update_charger_callback,
            10
        )

        self.timer = self.create_timer(0.5, self.publish_world_state)

        self.get_logger().info('Mock Gazebo World State Node avviato')

    def update_shelf_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f'Errore parsing /gazebo/update_shelf: {e}')
            return

        shelf_id = data.get("id")

        if shelf_id not in self.shelves:
            self.get_logger().warn(f'Shelf sconosciuto: {shelf_id}')
            return

        if "dirty" in data:
            self.shelves[shelf_id]["dirty"] = bool(data["dirty"])

        self.get_logger().info(
            f'Shelf aggiornato: {shelf_id} dirty={self.shelves[shelf_id]["dirty"]}'
        )

    def update_charger_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f'Errore parsing /gazebo/update_charger: {e}')
            return

        charger_id = data.get("id")

        if charger_id not in self.chargers:
            self.get_logger().warn(f'Charger sconosciuto: {charger_id}')
            return

        if "occupied" in data:
            self.chargers[charger_id]["occupied"] = bool(data["occupied"])

        if "occupied_by" in data:
            self.chargers[charger_id]["occupied_by"] = data["occupied_by"]

        self.get_logger().info(
            f'Charger aggiornato: {charger_id} occupied={self.chargers[charger_id]["occupied"]}'
        )

    def publish_world_state(self):
        shelves_msg = String()
        shelves_msg.data = json.dumps(list(self.shelves.values()))
        self.shelves_pub.publish(shelves_msg)

        chargers_msg = String()
        chargers_msg.data = json.dumps(list(self.chargers.values()))
        self.chargers_pub.publish(chargers_msg)


def main(args=None):
    rclpy.init(args=args)

    node = MockGazeboWorldStateNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()