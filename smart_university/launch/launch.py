from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('smart_university')
    ros_gz_sim_share = get_package_share_directory('ros_gz_sim')

    world_path = os.path.join(
        pkg_share,
        'simulation',
        'world.sdf'
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_share, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': world_path
        }.items()
    )

    bridge = TimerAction(
        period=2.0,
        actions=[
            Node(
                package='ros_gz_bridge',
                executable='parameter_bridge',
                arguments=[
                    '/lidar@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
                    '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
                ],
                output='screen'
            )
        ]
    )

    test_node = TimerAction(
        period=4.0,
        actions=[
            Node(
                package='smart_university',
                executable='lidar_controller',
                output='screen'
            )
        ]
    )

    return LaunchDescription([
        gazebo,
        bridge,
        test_node
    ])