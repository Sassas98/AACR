from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os
import json


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def vehicle_bridge_topics(vehicle_id):
    return [
        f"/{vehicle_id}/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan",
        f"/{vehicle_id}/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist",
        f"/{vehicle_id}/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry"
    ]


def navigation_executor_node(vehicle_id, delay, city_map_file):
    return TimerAction(
        period=delay,
        actions=[
            Node(
                package="smart_city",
                executable="navigation_executor",
                name=f"{vehicle_id}_navigation_executor",
                namespace=vehicle_id,
                parameters=[{
                    "vehicle_id": vehicle_id,
                    "map_config_file": city_map_file
                }],
                remappings=[
                    ("/cmd_vel", f"/{vehicle_id}/cmd_vel"),
                    ("/odom", f"/{vehicle_id}/odom"),
                    ("/scan", f"/{vehicle_id}/scan"),
                    (
                        "/navigation_executor/navigate_to_pose",
                        f"/{vehicle_id}/navigation_executor/navigate_to_pose"
                    )
                ],
                output="screen"
            )
        ]
    )


def bus_nodes(vehicle, delay, city_map_file, bus_paths_file, parkings_file):
    bus_id = vehicle["id"]
    initial_path_id = vehicle.get("initial_path_id", "path_A")

    return TimerAction(
        period=delay,
        actions=[
            Node(
                package="smart_city",
                executable="navigation_executor",
                name=f"{bus_id}_navigation_executor",
                namespace=bus_id,
                parameters=[{
                    "vehicle_id": bus_id,
                    "map_config_file": city_map_file
                }],
                remappings=[
                    ("/cmd_vel", f"/{bus_id}/cmd_vel"),
                    ("/odom", f"/{bus_id}/odom"),
                    ("/scan", f"/{bus_id}/scan"),
                    (
                        "/navigation_executor/navigate_to_pose",
                        f"/{bus_id}/navigation_executor/navigate_to_pose"
                    )
                ],
                output="screen"
            ),
            Node(
                package="smart_city",
                executable="bus_path_manager",
                name=f"{bus_id}_bus_path_manager",
                namespace=bus_id,
                parameters=[{
                    "bus_id": bus_id,
                    "initial_path_id": initial_path_id,
                    "paths_config_file": bus_paths_file,
                    "parkings_config_file": parkings_file,
                    "laps": 3
                }],
                remappings=[
                    (
                        "/navigation_executor/navigate_to_pose",
                        f"/{bus_id}/navigation_executor/navigate_to_pose"
                    )
                ],
                output="screen"
            )
        ]
    )


def taxi_nodes(vehicle, delay, city_map_file, parkings_file):
    taxi_id = vehicle["id"]

    return TimerAction(
        period=delay,
        actions=[
            Node(
                package="smart_city",
                executable="navigation_executor",
                name=f"{taxi_id}_navigation_executor",
                namespace=taxi_id,
                parameters=[{
                    "vehicle_id": taxi_id,
                    "map_config_file": city_map_file
                }],
                remappings=[
                    ("/cmd_vel", f"/{taxi_id}/cmd_vel"),
                    ("/odom", f"/{taxi_id}/odom"),
                    ("/scan", f"/{taxi_id}/scan"),
                    (
                        "/navigation_executor/navigate_to_pose",
                        f"/{taxi_id}/navigation_executor/navigate_to_pose"
                    )
                ],
                output="screen"
            ),
            Node(
                package="smart_city",
                executable="taxi_request_manager",
                name=f"{taxi_id}_taxi_request_manager",
                namespace=taxi_id,
                parameters=[{
                    "taxi_id": taxi_id,
                    "parkings_config_file": parkings_file
                }],
                remappings=[
                    ("/taxi_status", f"/{taxi_id}/taxi_status"),
                    (
                        "/navigation_executor/navigate_to_pose",
                        f"/{taxi_id}/navigation_executor/navigate_to_pose"
                    )
                ],
                output="screen"
            ),
            Node(
                package="smart_city",
                executable="taxi_coordinator",
                name=f"{taxi_id}_taxi_coordinator",
                namespace=taxi_id,
                parameters=[{
                    "taxi_id": taxi_id
                }],
                remappings=[
                    ("/taxi_status", f"/{taxi_id}/taxi_status")
                ],
                output="screen"
            )
        ]
    )


def traffic_light_node(light, delay, city_map_file):
    node_id = light["node_id"]

    return TimerAction(
        period=delay,
        actions=[
            Node(
                package="smart_city",
                executable="traffic_light_manager",
                name=f"traffic_light_{node_id}",
                parameters=[{
                    "node_id": node_id,
                    "map_config_file": city_map_file
                }],
                output="screen"
            )
        ]
    )


def generate_launch_description():
    pkg_share = get_package_share_directory("smart_city")
    ros_gz_sim_share = get_package_share_directory("ros_gz_sim")

    world_path = os.path.join(pkg_share, "simulation", "world.sdf")
    model_path = os.path.join(pkg_share, "model")

    config_dir = os.path.join(pkg_share, "config")

    city_map_file = os.path.join(config_dir, "city_map.json")
    bus_paths_file = os.path.join(config_dir, "bus_paths.json")
    parkings_file = os.path.join(config_dir, "parkings.json")
    vehicles_file = os.path.join(config_dir, "vehicles.json")
    traffic_lights_file = os.path.join(config_dir, "traffic_lights.json")
    bus_stops_file = os.path.join(config_dir, "bus_stops.json")
    taxi_request_zones_file = os.path.join(config_dir, "taxi_request_zones.json")

    vehicles_data = load_json(vehicles_file)
    traffic_lights_data = load_json(traffic_lights_file)

    vehicles = vehicles_data["vehicles"]
    traffic_lights = traffic_lights_data["traffic_lights"]

    bus_vehicles = [v for v in vehicles if v["type"] == "BUS"]
    taxi_vehicles = [v for v in vehicles if v["type"] == "TAXI"]
    bridge_vehicles = [
        v for v in vehicles
        if v["type"] in ["BUS", "TAXI", "PRIVATE_CAR"]
    ]

    gazebo_model_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=model_path
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_share, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={
            "gz_args": world_path
        }.items()
    )

    bridge_arguments = []

    for vehicle in bridge_vehicles:
        bridge_arguments.extend(vehicle_bridge_topics(vehicle["id"]))

    bridge = TimerAction(
        period=2.0,
        actions=[
            Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                arguments=bridge_arguments,
                output="screen"
            )
        ]
    )

    simulation_event_nodes = TimerAction(
        period=3.0,
        actions=[
            Node(
                package="smart_city",
                executable="bus_booking_generator",
                parameters=[{
                    "bus_stops_config_file": bus_stops_file
                }],
                output="screen"
            ),
            Node(
                package="smart_city",
                executable="taxi_request_generator",
                parameters=[{
                    "taxi_request_zones_config_file": taxi_request_zones_file
                }],
                output="screen"
            ),
            Node(
                package="smart_city",
                executable="private_car_simulator_node",
                parameters=[{
                    "vehicles_config_file": vehicles_file,
                    "map_config_file": city_map_file
                }],
                output="screen"
            )
        ]
    )

    launch_items = [
        gazebo_model_path,
        gazebo,
        bridge,
        simulation_event_nodes
    ]

    delay = 3.5

    for light in traffic_lights:
        launch_items.append(
            traffic_light_node(
                light=light,
                delay=delay,
                city_map_file=city_map_file
            )
        )
        delay += 0.2

    delay = 5.0

    for bus in bus_vehicles:
        launch_items.append(
            bus_nodes(
                vehicle=bus,
                delay=delay,
                city_map_file=city_map_file,
                bus_paths_file=bus_paths_file,
                parkings_file=parkings_file
            )
        )
        delay += 0.5

    for taxi in taxi_vehicles:
        launch_items.append(
            taxi_nodes(
                vehicle=taxi,
                delay=delay,
                city_map_file=city_map_file,
                parkings_file=parkings_file
            )
        )
        delay += 0.5

    return LaunchDescription(launch_items)