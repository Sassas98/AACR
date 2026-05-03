from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def robot_nodes(robot_id, delay):
    return TimerAction(
        period=delay,
        actions=[
            Node(package='smart_university', executable='robot_state_node',
                 parameters=[{'robot_id': robot_id}], output='screen'),
            Node(package='smart_university', executable='battery_manager_node',
                 parameters=[{'robot_id': robot_id}], output='screen'),
            Node(package='smart_university', executable='navigator_node',
                 parameters=[{'robot_id': robot_id}], output='screen'),
            Node(package='smart_university', executable='executor_node',
                 parameters=[{'robot_id': robot_id}], output='screen'),
            Node(package='smart_university', executable='cleaning_action_server',
                 parameters=[{'robot_id': robot_id}], output='screen'),
            Node(package='smart_university', executable='charging_action_server',
                 parameters=[{'robot_id': robot_id}], output='screen'),
        ]
    )


def generate_launch_description():
    pkg_share = get_package_share_directory('smart_university')
    ros_gz_sim_share = get_package_share_directory('ros_gz_sim')

    world_path = os.path.join(pkg_share, 'simulation', 'world.sdf')
    model_path = os.path.join(pkg_share, 'model')

    gazebo_model_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=model_path
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
                    '/robot_1/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
                    '/robot_1/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
                    '/robot_1/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry',

                    '/robot_2/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
                    '/robot_2/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
                    '/robot_2/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry',

                    '/robot_3/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
                    '/robot_3/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
                    '/robot_3/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry',
                ],
                output='screen'
            )
        ]
    )

    robot_ids = ['robot_1', 'robot_2', 'robot_3']

    global_nodes = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='smart_university',
                executable='mock_gazebo_world_state_node',
                output='screen'
            ),
            Node(
                package='smart_university',
                executable='environment_perception_node',
                output='screen'
            ),
            Node(
                package='smart_university',
                executable='shelf_state_node',
                output='screen'
            ),
            Node(
                package='smart_university',
                executable='task_allocator_node',
                parameters=[{
                    'robot_ids': robot_ids,
                    'battery_low_threshold': 25.0
                }],
                output='screen'
            ),
            Node(
                package='smart_university',
                executable='world_state_adapter_node',
                output='screen'
            ),
        ]
    )

    return LaunchDescription([
        gazebo_model_path,
        gazebo,
        bridge,
        global_nodes,
        robot_nodes('robot_1', 4.0),
        robot_nodes('robot_2', 5.0),
        robot_nodes('robot_3', 6.0),
    ])