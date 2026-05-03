import json
import time
import rclpy

from rclpy.action import ActionServer
from rclpy.node import Node

from std_msgs.msg import String
from smart_university_interfaces.action import ChargeRobot


class ChargingActionServer(Node):

    def __init__(self):
        super().__init__('charging_action_server')

        self.declare_parameter('robot_id', 'robot_1')
        self.declare_parameter('charging_duration', 8.0)
        self.declare_parameter('feedback_period', 0.2)

        self.robot_id = self.get_parameter('robot_id').value
        self.charging_duration = float(
            self.get_parameter('charging_duration').value
        )
        self.feedback_period = float(
            self.get_parameter('feedback_period').value
        )

        self.charging_state_pub = self.create_publisher(
            String,
            f'/{self.robot_id}/charging_state',
            10
        )

        self.world_events_pub = self.create_publisher(
            String,
            '/world_events',
            10
        )

        self.action_server = ActionServer(
            self,
            ChargeRobot,
            f'/{self.robot_id}/charge',
            self.execute_charging
        )

        self.get_logger().info(
            f'Charging Action Server avviato per {self.robot_id}'
        )

    def execute_charging(self, goal_handle):
        task_id = goal_handle.request.task_id
        request_robot_id = goal_handle.request.robot_id
        target_id = goal_handle.request.target_id

        result = ChargeRobot.Result()

        if request_robot_id != self.robot_id:
            goal_handle.abort()
            result.success = False
            result.reason = f'Robot errato: {request_robot_id}'
            return result

        if not task_id or not target_id:
            goal_handle.abort()
            result.success = False
            result.reason = 'task_id o target_id mancante'
            return result

        self.get_logger().info(
            f'Inizio ricarica: robot={self.robot_id}, '
            f'charger={target_id}, task={task_id}'
        )

        self.publish_charging_state(True)
        self.publish_world_event(
            event_type='CHARGER_OCCUPIED',
            target_id=target_id,
            task_id=task_id
        )

        start_time = time.time()

        while time.time() - start_time < self.charging_duration:
            if goal_handle.is_cancel_requested:
                self.publish_charging_state(False)
                self.publish_world_event(
                    event_type='CHARGER_RELEASED',
                    target_id=target_id,
                    task_id=task_id
                )

                goal_handle.canceled()

                result.success = False
                result.reason = 'Ricarica cancellata'
                return result

            elapsed = time.time() - start_time
            progress = min(1.0, elapsed / self.charging_duration)

            feedback = ChargeRobot.Feedback()
            feedback.progress = float(progress)
            feedback.status = 'CHARGING'
            goal_handle.publish_feedback(feedback)

            time.sleep(self.feedback_period)

        self.publish_charging_state(False)
        self.publish_world_event(
            event_type='CHARGER_RELEASED',
            target_id=target_id,
            task_id=task_id
        )

        goal_handle.succeed()

        result.success = True
        result.reason = ''

        self.get_logger().info(
            f'Ricarica completata: charger={target_id}, task={task_id}'
        )

        return result

    def publish_charging_state(self, is_charging: bool):
        state = {
            "robot_id": self.robot_id,
            "is_charging": is_charging,
            "timestamp": self.get_clock().now().nanoseconds
        }

        msg = String()
        msg.data = json.dumps(state)

        self.charging_state_pub.publish(msg)

    def publish_world_event(self, event_type, target_id, task_id):
        event = {
            "event_type": event_type,
            "robot_id": self.robot_id,
            "target_id": target_id,
            "task_id": task_id,
            "timestamp": self.get_clock().now().nanoseconds
        }

        msg = String()
        msg.data = json.dumps(event)

        self.world_events_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    node = ChargingActionServer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.publish_charging_state(False)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()