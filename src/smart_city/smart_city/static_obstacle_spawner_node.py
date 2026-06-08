import json
import os

import rclpy
from rclpy.node import Node

import subprocess
import textwrap


class StaticObstacleSpawnerNode(Node):

    def __init__(self):
        super().__init__("static_obstacle_spawner_node")

        self.declare_parameter("obstacle_id", "")
        self.declare_parameter("obstacles_config_file", "config/static_obstacles.json")
        self.declare_parameter("world_name", "smart_city_world")

        self.obstacle_id = self.get_parameter("obstacle_id").value
        self.obstacles_config_file = self.get_parameter("obstacles_config_file").value
        self.world_name = self.get_parameter("world_name").value

        if not self.obstacle_id:
            raise RuntimeError("Parametro obbligatorio mancante: obstacle_id")

        obstacle = self.load_obstacle_by_id(self.obstacle_id)

        self.spawn_obstacle(obstacle)

    # ------------------------------------------------------------
    # CONFIG
    # ------------------------------------------------------------

    def load_json(self, file_path):
        if not os.path.isabs(file_path):
            file_path = os.path.join(os.getcwd(), file_path)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File non trovato: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_obstacle_by_id(self, obstacle_id):
        data = self.load_json(self.obstacles_config_file)

        for obstacle in data.get("static_obstacles", []):
            if obstacle.get("id") == obstacle_id:
                return obstacle

        raise RuntimeError(
            f"Ostacolo '{obstacle_id}' non trovato in {self.obstacles_config_file}"
        )

    # ------------------------------------------------------------
    # SPAWN
    # ------------------------------------------------------------

    def spawn_obstacle(self, obstacle):
        obstacle_id = obstacle["id"]

        x = float(obstacle.get("x", 0.0))
        y = float(obstacle.get("y", 0.0))
        z = float(obstacle.get("z", 0.35))
        yaw = float(obstacle.get("yaw", 0.0))

        size = obstacle.get("size", {})
        sx = float(size.get("x", 1.0))
        sy = float(size.get("y", 1.0))
        sz = float(size.get("z", 0.7))

        sdf = self.build_obstacle_sdf(
            obstacle_id=obstacle_id,
            x=x,
            y=y,
            z=z,
            yaw=yaw,
            sx=sx,
            sy=sy,
            sz=sz
        )

        sdf = " ".join(textwrap.dedent(sdf).split())
        req = f'sdf: "{sdf}", name: "{obstacle_id}"'

        self.call_gz_service(
            service=f"/world/{self.world_name}/create",
            reqtype="gz.msgs.EntityFactory",
            reptype="gz.msgs.Boolean",
            req=req,
            obstacle_id=obstacle_id
        )
    
    def call_gz_service(self, service, reqtype, reptype, req, obstacle_id):
        cmd = [
            "gz", "service",
            "-s", service,
            "--reqtype", reqtype,
            "--reptype", reptype,
            "--timeout", "1000",
            "--req", req
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=2.0
            )

            if result.returncode == 0:
                self.get_logger().info(
                    f"{obstacle_id}: richiesta spawn inviata a Gazebo"
                )
            else:
                self.get_logger().error(
                    f"{obstacle_id}: spawn fallito | "
                    f"stdout={result.stdout} stderr={result.stderr}"
                )

        except Exception as ex:
            self.get_logger().error(
                f"{obstacle_id}: errore chiamata gz service: {ex}"
            )

    # ------------------------------------------------------------
    # SDF
    # ------------------------------------------------------------

    def build_obstacle_sdf(
        self,
        obstacle_id,
        x,
        y,
        z,
        yaw,
        sx,
        sy,
        sz
    ):
        ambient = "0.45 0.30 0.15 1"
        diffuse = "0.45 0.30 0.15 1"

        mass = max(1.0, sx * sy * sz * 20.0)

        return f"""
            <sdf version='1.9'>
                <model name='{obstacle_id}'>
                    <static>true</static>
                    <pose>{x} {y} {z} 0 0 {yaw}</pose>

                    <link name='base_link'>
                    <inertial>
                        <mass>{mass}</mass>
                        <inertia>
                        <ixx>1.0</ixx>
                        <iyy>1.0</iyy>
                        <izz>1.0</izz>
                        </inertia>
                    </inertial>

                    <collision name='collision'>
                        <geometry>
                        <box>
                            <size>{sx} {sy} {sz}</size>
                        </box>
                        </geometry>
                    </collision>

                    <visual name='visual'>
                        <geometry>
                        <box>
                            <size>{sx} {sy} {sz}</size>
                        </box>
                        </geometry>
                        <material>
                        <ambient>{ambient}</ambient>
                        <diffuse>{diffuse}</diffuse>
                        </material>
                    </visual>
                    </link>
                </model>
            </sdf>
        """

def main(args=None):
    rclpy.init(args=args)

    node = StaticObstacleSpawnerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    if rclpy.ok():
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()