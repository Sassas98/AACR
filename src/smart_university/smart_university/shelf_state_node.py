import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class ShelfStateNode(Node):

    def __init__(self):
        super().__init__('shelf_state_node')

        self.environment_observations_sub = self.create_subscription(
            String,
            '/environment_observations',
            self.environment_observations_callback,
            10
        )

        self.reservation_sub = self.create_subscription(
            String,
            '/target_reservations',
            self.target_reservation_callback,
            10
        )

        self.environment_state_pub = self.create_publisher(
            String,
            '/environment_state',
            10
        )

        self.shelves = {}
        self.chargers = {}

        self.timer = self.create_timer(0.5, self.publish_environment_state)

        self.get_logger().info('Shelf State Node avviato')

    def environment_observations_callback(self, msg: String):
        try:
            data = json.loads(msg.data)

            observed_shelves = data.get("shelves", [])
            observed_chargers = data.get("chargers", [])

            self.update_shelves(observed_shelves)
            self.update_chargers(observed_chargers)

        except Exception as e:
            self.get_logger().error(f'Errore parsing /environment_observations: {e}')

    def target_reservation_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f'Errore parsing /target_reservations: {e}')
            return

        target_id = data.get("target_id")
        target_type = data.get("target_type")
        reserved = bool(data.get("reserved", False))
        reserved_by = data.get("reserved_by")
        task_id = data.get("task_id")

        if not target_id:
            self.get_logger().warn('Reservation senza target_id')
            return

        if target_type == "SHELF":
            target = self.shelves.get(target_id)
        elif target_type == "CHARGER":
            target = self.chargers.get(target_id)
        else:
            self.get_logger().warn(f'target_type sconosciuto: {target_type}')
            return

        if target is None:
            self.get_logger().warn(f'Target non trovato: {target_id}')
            return

        target["reserved"] = reserved
        target["reserved_by"] = reserved_by if reserved else None
        target["reserved_task_id"] = task_id if reserved else None

        self.get_logger().info(
            f'Reservation aggiornata: {target_type} {target_id} '
            f'reserved={reserved} by={reserved_by} task={task_id}'
        )

    def update_shelves(self, observed_shelves):
        for shelf in observed_shelves:
            shelf_id = shelf.get("id")

            if not shelf_id:
                continue

            dirty = bool(shelf.get("dirty", False))
            x = float(shelf.get("x", 0.0))
            y = float(shelf.get("y", 0.0))
            theta = float(shelf.get("theta", 0.0))

            already_known = self.shelves.get(shelf_id)

            if already_known is None:
                self.shelves[shelf_id] = {
                    "id": shelf_id,
                    "x": x,
                    "y": y,
                    "theta": theta,
                    "dirty": dirty,
                    "reserved": False,
                    "reserved_by": None,
                    "reserved_task_id": None
                }
            else:
                already_known["x"] = x
                already_known["y"] = y
                already_known["theta"] = theta
                already_known["dirty"] = dirty

                if not dirty:
                    already_known["reserved"] = False
                    already_known["reserved_by"] = None
                    already_known["reserved_task_id"] = None

    def update_chargers(self, observed_chargers):
        for charger in observed_chargers:
            charger_id = charger.get("id")

            if not charger_id:
                continue

            occupied = bool(charger.get("occupied", False))
            x = float(charger.get("x", 0.0))
            y = float(charger.get("y", 0.0))
            theta = float(charger.get("theta", 0.0))

            already_known = self.chargers.get(charger_id)

            if already_known is None:
                self.chargers[charger_id] = {
                    "id": charger_id,
                    "x": x,
                    "y": y,
                    "theta": theta,
                    "occupied": occupied,
                    "reserved": False,
                    "reserved_by": None,
                    "reserved_task_id": None
                }
            else:
                already_known["x"] = x
                already_known["y"] = y
                already_known["theta"] = theta
                already_known["occupied"] = occupied

                if not occupied:
                    already_known["reserved"] = False
                    already_known["reserved_by"] = None
                    already_known["reserved_task_id"] = None

    def publish_environment_state(self):
        state = {
            "shelves": list(self.shelves.values()),
            "chargers": list(self.chargers.values()),
            "timestamp": self.get_clock().now().nanoseconds
        }

        msg = String()
        msg.data = json.dumps(state)

        self.environment_state_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    node = ShelfStateNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()