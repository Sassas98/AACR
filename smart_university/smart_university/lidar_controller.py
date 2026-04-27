#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


class LidarController(Node):
    def __init__(self):
        super().__init__('lidar_controller')

        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        self.subscription = self.create_subscription(
            LaserScan,
            '/lidar',
            self.lidar_callback,
            10
        )

    def lidar_callback(self, msg: LaserScan):
        cmd = Twist()

        obstacle_detected = False

        for distance in msg.ranges:
            if distance < 1.0:
                obstacle_detected = True
                break

        if not obstacle_detected:
            cmd.linear.x = 0.5
            cmd.angular.z = 0.0
        else:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.5

        self.publisher.publish(cmd)


def main(args=None):
    rclpy.init(args=args)

    node = LidarController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()