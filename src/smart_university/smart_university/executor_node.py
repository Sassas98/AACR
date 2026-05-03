import json
import math
import rclpy

from rclpy.node import Node
from rclpy.action import ActionClient

from std_msgs.msg import String
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus

from smart_university_interfaces.action import CleanShelf, ChargeRobot


class ExecutorNode(Node):

    def __init__(self):
        super().__init__('executor_node')

        self.declare_parameter('robot_id', 'robot_1')
        self.declare_parameter('navigation_timeout_sec', 60.0)
        self.declare_parameter('task_action_timeout_sec', 60.0)

        self.robot_id = self.get_parameter('robot_id').value
        self.navigation_timeout_sec = float(
            self.get_parameter('navigation_timeout_sec').value
        )
        self.task_action_timeout_sec = float(
            self.get_parameter('task_action_timeout_sec').value
        )

        self.current_task = None
        self.busy = False

        self.command_sub = self.create_subscription(
            String,
            f'/{self.robot_id}/command',
            self.command_callback,
            10
        )

        self.local_status_pub = self.create_publisher(
            String,
            f'/{self.robot_id}/local_status',
            10
        )

        self.task_result_pub = self.create_publisher(
            String,
            f'/{self.robot_id}/task_result',
            10
        )

        self.nav_client = ActionClient(
            self,
            NavigateToPose,
            f'/{self.robot_id}/navigate_to_pose'
        )

        self.clean_client = ActionClient(
            self,
            CleanShelf,
            f'/{self.robot_id}/clean_shelf'
        )

        self.charge_client = ActionClient(
            self,
            ChargeRobot,
            f'/{self.robot_id}/charge'
        )

        self.get_logger().info(f'Executor Node avviato per {self.robot_id}')

    def command_callback(self, msg: String):
        if self.busy:
            self.get_logger().warn(
                f'Comando ignorato: {self.robot_id} è già occupato'
            )
            return

        try:
            command = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f'Errore parsing command: {e}')
            return

        validation_error = self.validate_command(command)

        if validation_error is not None:
            self.get_logger().warn(f'Comando non valido: {validation_error}')
            return

        self.current_task = command
        self.busy = True

        self.publish_local_status(
            availability="BUSY",
            task_id=command["task_id"],
            cmd=command["cmd"]
        )

        self.start_navigation(command)

    def validate_command(self, command):
        required_fields = [
            "task_id",
            "robot_id",
            "cmd",
            "target_id",
            "target_type",
            "target_pose"
        ]

        for field in required_fields:
            if field not in command:
                return f'campo mancante: {field}'

        if command["robot_id"] != self.robot_id:
            return f'robot_id errato: {command["robot_id"]}'

        if command["cmd"] not in ["CLEAN_SHELF", "CHARGE"]:
            return f'cmd sconosciuto: {command["cmd"]}'

        target_pose = command["target_pose"]

        for field in ["x", "y"]:
            if field not in target_pose:
                return f'target_pose senza {field}'

        return None

    def start_navigation(self, command):
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.fail_task("NAV_SERVER_UNAVAILABLE")
            return

        goal = NavigateToPose.Goal()

        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self.get_clock().now().to_msg()

        goal.pose.pose.position.x = float(command["target_pose"]["x"])
        goal.pose.pose.position.y = float(command["target_pose"]["y"])
        goal.pose.pose.position.z = 0.0

        theta = float(command["target_pose"].get("theta", 0.0))
        qz, qw = self.yaw_to_quaternion(theta)

        goal.pose.pose.orientation.z = qz
        goal.pose.pose.orientation.w = qw

        future = self.nav_client.send_goal_async(
            goal,
            feedback_callback=self.navigation_feedback_callback
        )

        future.add_done_callback(self.navigation_goal_response_callback)

    def navigation_goal_response_callback(self, future):
        try:
            goal_handle = future.result()
        except Exception as e:
            self.fail_task(f"NAV_GOAL_ERROR: {e}")
            return

        if not goal_handle.accepted:
            self.fail_task("NAV_REJECTED")
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.navigation_result_callback)

    def navigation_feedback_callback(self, feedback_msg):
        pass

    def navigation_result_callback(self, future):
        try:
            wrapped_result = future.result()
        except Exception as e:
            self.fail_task(f"NAV_RESULT_ERROR: {e}")
            return

        if wrapped_result.status == GoalStatus.STATUS_SUCCEEDED:
            self.start_task_action()
        elif wrapped_result.status == GoalStatus.STATUS_CANCELED:
            self.fail_task("NAV_CANCELED")
        elif wrapped_result.status == GoalStatus.STATUS_ABORTED:
            self.fail_task("NAV_ABORTED")
        else:
            self.fail_task(f"NAV_FAILED_STATUS_{wrapped_result.status}")

    def start_task_action(self):
        cmd = self.current_task["cmd"]

        if cmd == "CLEAN_SHELF":
            self.start_cleaning_action()
        elif cmd == "CHARGE":
            self.start_charging_action()
        else:
            self.fail_task("UNKNOWN_CMD")

    def start_cleaning_action(self):
        if not self.clean_client.wait_for_server(timeout_sec=5.0):
            self.fail_task("CLEAN_SERVER_UNAVAILABLE")
            return

        goal = CleanShelf.Goal()
        goal.task_id = self.current_task["task_id"]
        goal.robot_id = self.robot_id
        goal.target_id = self.current_task["target_id"]

        future = self.clean_client.send_goal_async(
            goal,
            feedback_callback=self.cleaning_feedback_callback
        )

        future.add_done_callback(self.cleaning_goal_response_callback)

    def cleaning_goal_response_callback(self, future):
        try:
            goal_handle = future.result()
        except Exception as e:
            self.fail_task(f"CLEAN_GOAL_ERROR: {e}")
            return

        if not goal_handle.accepted:
            self.fail_task("CLEAN_REJECTED")
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.cleaning_result_callback)

    def cleaning_feedback_callback(self, feedback_msg):
        pass

    def cleaning_result_callback(self, future):
        try:
            wrapped_result = future.result()
        except Exception as e:
            self.fail_task(f"CLEAN_RESULT_ERROR: {e}")
            return

        result = wrapped_result.result

        if wrapped_result.status == GoalStatus.STATUS_SUCCEEDED and result.success:
            self.complete_task()
        else:
            reason = getattr(result, "reason", None) or "CLEAN_FAILED"
            self.fail_task(reason)

    def start_charging_action(self):
        if not self.charge_client.wait_for_server(timeout_sec=5.0):
            self.fail_task("CHARGE_SERVER_UNAVAILABLE")
            return

        goal = ChargeRobot.Goal()
        goal.task_id = self.current_task["task_id"]
        goal.robot_id = self.robot_id
        goal.target_id = self.current_task["target_id"]

        future = self.charge_client.send_goal_async(
            goal,
            feedback_callback=self.charging_feedback_callback
        )

        future.add_done_callback(self.charging_goal_response_callback)

    def charging_goal_response_callback(self, future):
        try:
            goal_handle = future.result()
        except Exception as e:
            self.fail_task(f"CHARGE_GOAL_ERROR: {e}")
            return

        if not goal_handle.accepted:
            self.fail_task("CHARGE_REJECTED")
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.charging_result_callback)

    def charging_feedback_callback(self, feedback_msg):
        pass

    def charging_result_callback(self, future):
        try:
            wrapped_result = future.result()
        except Exception as e:
            self.fail_task(f"CHARGE_RESULT_ERROR: {e}")
            return

        result = wrapped_result.result

        if wrapped_result.status == GoalStatus.STATUS_SUCCEEDED and result.success:
            self.complete_task()
        else:
            reason = getattr(result, "reason", None) or "CHARGE_FAILED"
            self.fail_task(reason)

    def complete_task(self):
        self.publish_task_result(
            success=True,
            reason=None
        )

        self.reset_state()

    def fail_task(self, reason):
        self.get_logger().warn(f'Task fallito: {reason}')

        self.publish_task_result(
            success=False,
            reason=reason
        )

        self.reset_state()

    def publish_task_result(self, success, reason=None):
        if self.current_task is None:
            return

        result = {
            "task_id": self.current_task["task_id"],
            "robot_id": self.robot_id,
            "cmd": self.current_task["cmd"],
            "target_id": self.current_task["target_id"],
            "target_type": self.current_task["target_type"],
            "success": success,
            "reason": reason,
            "timestamp": self.get_clock().now().nanoseconds
        }

        msg = String()
        msg.data = json.dumps(result)

        self.task_result_pub.publish(msg)

    def publish_local_status(self, availability, task_id=None, cmd=None):
        status = {
            "robot_id": self.robot_id,
            "availability": availability,
            "current_task_id": task_id,
            "current_cmd": cmd,
            "timestamp": self.get_clock().now().nanoseconds
        }

        msg = String()
        msg.data = json.dumps(status)

        self.local_status_pub.publish(msg)

    def reset_state(self):
        self.current_task = None
        self.busy = False

        self.publish_local_status(
            availability="FREE",
            task_id=None,
            cmd=None
        )

    def yaw_to_quaternion(self, yaw):
        return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def main(args=None):
    rclpy.init(args=args)

    node = ExecutorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()