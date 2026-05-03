import json
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class TaskAllocatorNode(Node):

    def __init__(self):
        super().__init__('task_allocator_node')

        self.declare_parameter('robot_ids', ['robot_1'])
        self.declare_parameter('battery_low_threshold', 25.0)

        self.robot_ids = list(self.get_parameter('robot_ids').value)
        self.battery_low_threshold = float(
            self.get_parameter('battery_low_threshold').value
        )

        self.environment_state = {
            "shelves": [],
            "chargers": []
        }

        self.robot_statuses = {}
        self.command_publishers = {}

        self.assigned_targets = set()
        self.active_tasks = {}
        self.task_counter = 0

        self.environment_sub = self.create_subscription(
            String,
            '/environment_state',
            self.environment_state_callback,
            10
        )

        self.reservation_pub = self.create_publisher(
            String,
            '/target_reservations',
            10
        )

        for robot_id in self.robot_ids:
            self.robot_statuses[robot_id] = None

            self.create_subscription(
                String,
                f'/{robot_id}/status',
                lambda msg, rid=robot_id: self.robot_status_callback(msg, rid),
                10
            )

            self.create_subscription(
                String,
                f'/{robot_id}/task_result',
                lambda msg, rid=robot_id: self.task_result_callback(msg, rid),
                10
            )

            self.command_publishers[robot_id] = self.create_publisher(
                String,
                f'/{robot_id}/command',
                10
            )

        self.timer = self.create_timer(1.0, self.allocate_tasks)

        self.get_logger().info(
            f'Task Allocator Node avviato per robot: {self.robot_ids}'
        )

    def environment_state_callback(self, msg: String):
        try:
            self.environment_state = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f'Errore parsing /environment_state: {e}')

    def robot_status_callback(self, msg: String, robot_id: str):
        try:
            self.robot_statuses[robot_id] = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f'Errore parsing /{robot_id}/status: {e}')

    def task_result_callback(self, msg: String, robot_id: str):
        try:
            result = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f'Errore parsing /{robot_id}/task_result: {e}')
            return

        task_id = result.get("task_id")
        success = bool(result.get("success", False))

        if not task_id:
            self.get_logger().warn(f'Risultato senza task_id da {robot_id}')
            return

        task = self.active_tasks.pop(task_id, None)

        if task is None:
            self.get_logger().warn(f'Risultato task sconosciuto: {task_id}')
            return

        target_id = task["target_id"]
        target_type = task["target_type"]

        self.release_target(
            target_id=target_id,
            target_type=target_type,
            task_id=task_id
        )

        if success:
            self.get_logger().info(
                f'Task completato: {task_id} da {robot_id}'
            )
        else:
            self.get_logger().warn(
                f'Task fallito: {task_id} da {robot_id}'
            )

    def allocate_tasks(self):
        available_robots = self.get_available_robots()

        if not available_robots:
            return

        reserved_targets_this_cycle = set()

        for robot_id, status in list(available_robots):
            if self.robot_needs_charge(status):
                charger = self.find_best_available_charger(
                    robot_status=status,
                    reserved_targets_this_cycle=reserved_targets_this_cycle
                )

                if charger is not None:
                    self.send_command(
                        robot_id=robot_id,
                        command_type="CHARGE",
                        target=charger
                    )

                    reserved_targets_this_cycle.add(charger["id"])

                    available_robots = [
                        item for item in available_robots
                        if item[0] != robot_id
                    ]

        dirty_shelves = [
            shelf for shelf in self.environment_state.get("shelves", [])
            if shelf.get("dirty") is True
            and shelf.get("reserved") is not True
            and shelf.get("id") not in self.assigned_targets
        ]

        if not dirty_shelves:
            return

        for shelf in dirty_shelves:
            best_robot = self.find_nearest_robot(
                shelf=shelf,
                available_robots=available_robots
            )

            if best_robot is None:
                continue

            robot_id, _ = best_robot

            self.send_command(
                robot_id=robot_id,
                command_type="CLEAN_SHELF",
                target=shelf
            )

            available_robots = [
                item for item in available_robots
                if item[0] != robot_id
            ]

            if not available_robots:
                break

    def get_available_robots(self):
        result = []

        for robot_id, status in self.robot_statuses.items():
            if status is None:
                continue

            if status.get("availability") != "FREE":
                continue

            result.append((robot_id, status))

        return result

    def robot_needs_charge(self, status):
        battery = float(status.get("battery", 100.0))
        return battery < self.battery_low_threshold

    def find_best_available_charger(self, robot_status, reserved_targets_this_cycle):
        chargers = [
            charger for charger in self.environment_state.get("chargers", [])
            if charger.get("occupied") is not True
            and charger.get("reserved") is not True
            and charger.get("id") not in reserved_targets_this_cycle
            and charger.get("id") not in self.assigned_targets
        ]

        if not chargers:
            return None

        return min(
            chargers,
            key=lambda charger: self.distance(robot_status, charger)
        )

    def find_nearest_robot(self, shelf, available_robots):
        candidates = []

        for robot_id, status in available_robots:
            if self.robot_needs_charge(status):
                continue

            candidates.append((robot_id, status))

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda item: self.distance(item[1], shelf)
        )

    def send_command(self, robot_id, command_type, target):
        self.task_counter += 1
        task_id = f"task_{self.task_counter}"

        target_type = "CHARGER" if command_type == "CHARGE" else "SHELF"

        command = {
            "task_id": task_id,
            "robot_id": robot_id,
            "cmd": command_type,
            "target_id": target["id"],
            "target_type": target_type,
            "target_pose": {
                "x": target["x"],
                "y": target["y"],
                "theta": target.get("theta", 0.0)
            }
        }

        msg = String()
        msg.data = json.dumps(command)
        self.command_publishers[robot_id].publish(msg)

        self.reserve_target(
            target_id=target["id"],
            target_type=target_type,
            robot_id=robot_id,
            task_id=task_id
        )

        self.assigned_targets.add(target["id"])

        self.active_tasks[task_id] = {
            "robot_id": robot_id,
            "target_id": target["id"],
            "target_type": target_type,
            "cmd": command_type
        }

        self.get_logger().info(
            f'Assegnato a {robot_id}: {command_type} -> {target["id"]}'
        )

    def reserve_target(self, target_id, target_type, robot_id, task_id):
        reservation = {
            "target_id": target_id,
            "target_type": target_type,
            "reserved": True,
            "reserved_by": robot_id,
            "task_id": task_id
        }

        msg = String()
        msg.data = json.dumps(reservation)
        self.reservation_pub.publish(msg)

    def release_target(self, target_id, target_type, task_id):
        reservation = {
            "target_id": target_id,
            "target_type": target_type,
            "reserved": False,
            "reserved_by": None,
            "task_id": task_id
        }

        msg = String()
        msg.data = json.dumps(reservation)
        self.reservation_pub.publish(msg)

        self.assigned_targets.discard(target_id)

    def distance(self, robot_status, target):
        robot_pose = robot_status.get("pose", {})

        rx = float(robot_pose.get("x", 0.0))
        ry = float(robot_pose.get("y", 0.0))

        tx = float(target.get("x", 0.0))
        ty = float(target.get("y", 0.0))

        return math.sqrt((rx - tx) ** 2 + (ry - ty) ** 2)


def main(args=None):
    rclpy.init(args=args)

    node = TaskAllocatorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()