from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    TimerAction,
    SetEnvironmentVariable
)
from launch.launch_description_sources import (
    PythonLaunchDescriptionSource
)
from launch_ros.actions import Node
from ament_index_python.packages import (
    get_package_share_directory
)

import os
import json

from launch.actions import (
    IncludeLaunchDescription,
    TimerAction,
    SetEnvironmentVariable,
    ExecuteProcess
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_node_degrees(city_map):
    degrees = {
        node["id"]: 0
        for node in city_map["nodes"]
    }

    for edge in city_map["edges"]:
        degrees[edge["from"]] += 1
        degrees[edge["to"]] += 1

    return degrees


def bridge_for_vehicle(vehicle_id):
    return [
        # LiDAR: solo Gazebo -> ROS
        f"/{vehicle_id}/scan@gz.msgs.LaserScan[sensor_msgs/msg/LaserScan",

        # Comandi: solo ROS -> Gazebo
        f"/{vehicle_id}/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
    ]


def navigation_executor_parameters(
    vehicle_id,
    city_map_file,
    vehicles_file,
    spawn_x=0.0,
    spawn_y=0.0,
    spawn_yaw=0.0
):
    return {
        "vehicle_id": vehicle_id,
        "map_config_file": city_map_file,
        "vehicles_config_file": vehicles_file,

        # Posizione di spawn nel mondo Gazebo.
        # Necessaria per trasformare l'odometria locale
        # in coordinate mondo corrette.
        "initial_x": spawn_x,
        "initial_y": spawn_y,
        "initial_yaw": spawn_yaw,

        "default_max_speed": 2.8,

        "linear_k": 2.0,
        "angular_k": 0.7,

        "max_angular_speed": 0.25,

        "waypoint_tolerance": 0.25,
        "target_tolerance": 0.40,

        "lane_offset_ratio": 0.5
    }


def navigation_executor_remappings(vehicle_id):
    return [
        ("/cmd_vel", f"/{vehicle_id}/cmd_vel"),
        ("/scan", f"/{vehicle_id}/scan"),

        (
            "/navigation_executor/navigate_to_pose",
            f"/{vehicle_id}/navigation_executor/navigate_to_pose"
        )
    ]


def navigation_executor_node(
    vehicle_id,
    delay,
    city_map_file,
    vehicles_file,
    spawn_x=0.0,
    spawn_y=0.0,
    spawn_yaw=0.0
):
    return TimerAction(
        period=delay,
        actions=[
            Node(
                package="smart_city",
                executable="navigation_executor",

                namespace=vehicle_id,
                name=f"{vehicle_id}_navigation_executor",

                parameters=[navigation_executor_parameters(
                    vehicle_id,
                    city_map_file,
                    vehicles_file,
                    spawn_x=spawn_x,
                    spawn_y=spawn_y,
                    spawn_yaw=spawn_yaw
                )],

                remappings=navigation_executor_remappings(
                    vehicle_id
                ),

                output="screen"
            )
        ]
    )


def bus_nodes(
    bus_id,
    delay,
    city_map_file,
    vehicles_file,
    bus_paths_file,
    parkings_file,
    spawn_x=0.0,
    spawn_y=0.0,
    spawn_yaw=0.0
):
    return TimerAction(
        period=delay,
        actions=[
            Node(
                package="smart_city",
                executable="navigation_executor",

                namespace=bus_id,
                name=f"{bus_id}_navigation_executor",

                parameters=[navigation_executor_parameters(
                    bus_id,
                    city_map_file,
                    vehicles_file,
                    spawn_x=spawn_x,
                    spawn_y=spawn_y,
                    spawn_yaw=spawn_yaw
                )],

                remappings=navigation_executor_remappings(
                    bus_id
                ),

                output="screen"
            ),

            Node(
                package="smart_city",
                executable="bus_path_manager",

                namespace=bus_id,
                name=f"{bus_id}_bus_path_manager",

                parameters=[{
                    "bus_id": bus_id,
                    "paths_config_file": bus_paths_file,
                    "parkings_config_file": parkings_file
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


def taxi_nodes(
    taxi_id,
    delay,
    city_map_file,
    vehicles_file,
    parkings_file,
    spawn_x=0.0,
    spawn_y=0.0,
    spawn_yaw=0.0
):
    return TimerAction(
        period=delay,
        actions=[
            Node(
                package="smart_city",
                executable="navigation_executor",

                namespace=taxi_id,
                name=f"{taxi_id}_navigation_executor",

                parameters=[navigation_executor_parameters(
                    taxi_id,
                    city_map_file,
                    vehicles_file,
                    spawn_x=spawn_x,
                    spawn_y=spawn_y,
                    spawn_yaw=spawn_yaw
                )],

                remappings=navigation_executor_remappings(
                    taxi_id
                ),

                output="screen"
            ),

            Node(
                package="smart_city",
                executable="taxi_request_manager",

                namespace=taxi_id,
                name=f"{taxi_id}_taxi_request_manager",

                parameters=[{
                    "taxi_id": taxi_id,
                    "parkings_config_file": parkings_file
                }],

                remappings=[
                    (
                        "/taxi_status",
                        f"/{taxi_id}/taxi_status"
                    ),

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

                namespace=taxi_id,
                name=f"{taxi_id}_taxi_coordinator",

                parameters=[{
                    "taxi_id": taxi_id
                }],

                remappings=[
                    (
                        "/taxi_status",
                        f"/{taxi_id}/taxi_status"
                    )
                ],

                output="screen"
            )
        ]
    )


def traffic_light_node(
    light,
    delay,
    city_map_file
):
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
                    "map_config_file": city_map_file,

                    "green_duration":
                        light.get("green_duration", 8.0),

                    "yellow_duration":
                        light.get("yellow_duration", 2.0),

                    "min_green_duration":
                        light.get("min_green_duration", 4.0),

                    "max_green_duration":
                        light.get("max_green_duration", 16.0)
                }],

                output="screen"
            )
        ]
    )


def generate_launch_description():
    pkg_share = get_package_share_directory(
        "smart_city"
    )

    ros_gz_sim_share = get_package_share_directory(
        "ros_gz_sim"
    )

    world_path = os.path.join(
        pkg_share,
        "simulation",
        "world_with_privates.sdf"
    )

    model_path = os.path.join(
        pkg_share,
        "model"
    )

    config_dir = os.path.join(
        pkg_share,
        "config"
    )

    vehicles_file = os.path.join(
        config_dir,
        "vehicles.json"
    )

    city_map_file = os.path.join(
        config_dir,
        "city_map.json"
    )

    bus_paths_file = os.path.join(
        config_dir,
        "bus_paths.json"
    )

    parkings_file = os.path.join(
        config_dir,
        "parkings.json"
    )

    traffic_lights_file = os.path.join(
        config_dir,
        "traffic_lights.json"
    )

    bus_stops_file = os.path.join(
        config_dir,
        "bus_stops.json"
    )

    taxi_request_zones_file = os.path.join(
        config_dir,
        "taxi_request_zones.json"
    )

    vehicles_data = load_json(vehicles_file)
    city_map = load_json(city_map_file)
    traffic_lights_data = load_json(traffic_lights_file)

    vehicles = vehicles_data["vehicles"]

    # Lookup rapido spawn per vehicle_id
    spawn_by_id = {
        v["id"]: v.get("spawn", {"x": 0.0, "y": 0.0, "yaw": 0.0})
        for v in vehicles
    }

    node_degrees = compute_node_degrees(city_map)

    bus_ids = [
        v["id"]
        for v in vehicles
        if v["type"] == "BUS"
    ]

    taxi_ids = [
        v["id"]
        for v in vehicles
        if v["type"] == "TAXI"
    ]

    private_car_ids = [
        v["id"]
        for v in vehicles
        if v["type"] == "PRIVATE_CAR"
    ]

    bridge_args = []

    for vehicle in vehicles:
        bridge_args.extend(
            bridge_for_vehicle(vehicle["id"])
        )

    gazebo_model_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=model_path
    )

    gazebo_software_rendering = SetEnvironmentVariable(
        name="LIBGL_ALWAYS_SOFTWARE",
        value="1"
    )

    gazebo_render_engine = SetEnvironmentVariable(
        name="GZ_RENDER_ENGINE",
        value="ogre"
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                ros_gz_sim_share,
                "launch",
                "gz_sim.launch.py"
            )
        ),

        launch_arguments={
            "gz_args": f"-r {world_path}"
        }.items()
    )

    bridge = TimerAction(
        period=2.0,
        actions=[
            Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",

                arguments=bridge_args,

                output="screen"
            )
        ]
    )

    simulation_event_nodes = TimerAction(
        period=3.0,
        actions=[
            Node(
                package="smart_city",
                executable="taxi_request_generator",

                parameters=[{
                    "taxi_request_zones_config_file":
                        taxi_request_zones_file
                }],

                output="screen"
            )
        ]
    )

    gazebo_visual_controller = TimerAction(
        period=3.5,
        actions=[
            Node(
                package="smart_city",
                executable="gazebo_visual_controller",

                parameters=[{
                    "world_name": "smart_city_world",
                    "city_map_file": city_map_file
                }],

                output="screen"
            )
        ]
    )

    private_car_controller = TimerAction(
        period=6.0,
        actions=[
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

    gz_plugin_path = SetEnvironmentVariable(
        name="GZ_SIM_SYSTEM_PLUGIN_PATH",
        value=os.path.join(
            os.path.expanduser("~"),
            "AACR",
            "install",
            "smart_city_gz_plugins",
            "lib"
        )
    )

    launch_items = [
        gazebo_model_path,
        gz_plugin_path,
        gazebo_software_rendering,
        gazebo_render_engine,

        gazebo,
        bridge,
        gazebo_visual_controller,
        simulation_event_nodes
    ]

    delay = 4.0

    for light in traffic_lights_data["traffic_lights"]:
        node_id = light["node_id"]

        degree = node_degrees.get(node_id, 0)

        if degree < 3 or degree > 4:
            continue

        launch_items.append(
            traffic_light_node(
                light=light,
                delay=delay,
                city_map_file=city_map_file
            )
        )

        delay += 0.15

    for bus_id in bus_ids:
        spawn = spawn_by_id.get(
            bus_id, {"x": 0.0, "y": 0.0, "yaw": 0.0}
        )

        launch_items.append(
            bus_nodes(
                bus_id=bus_id,
                delay=delay,

                city_map_file=city_map_file,
                vehicles_file=vehicles_file,

                bus_paths_file=bus_paths_file,
                parkings_file=parkings_file,

                spawn_x=float(spawn["x"]),
                spawn_y=float(spawn["y"]),
                spawn_yaw=float(spawn.get("yaw", 0.0))
            )
        )

        delay += 0.5

    for taxi_id in taxi_ids:
        spawn = spawn_by_id.get(
            taxi_id, {"x": 0.0, "y": 0.0, "yaw": 0.0}
        )

        launch_items.append(
            taxi_nodes(
                taxi_id=taxi_id,
                delay=delay,

                city_map_file=city_map_file,
                vehicles_file=vehicles_file,

                parkings_file=parkings_file,

                spawn_x=float(spawn["x"]),
                spawn_y=float(spawn["y"]),
                spawn_yaw=float(spawn.get("yaw", 0.0))
            )
        )

        delay += 0.5

    for car_id in private_car_ids:
        spawn = spawn_by_id.get(
            car_id, {"x": 0.0, "y": 0.0, "yaw": 0.0}
        )

        launch_items.append(
            navigation_executor_node(
                vehicle_id=car_id,
                delay=delay,

                city_map_file=city_map_file,
                vehicles_file=vehicles_file,

                spawn_x=float(spawn["x"]),
                spawn_y=float(spawn["y"]),
                spawn_yaw=float(spawn.get("yaw", 0.0))
            )
        )

        delay += 0.5

    launch_items.append(private_car_controller)

    return LaunchDescription(launch_items)