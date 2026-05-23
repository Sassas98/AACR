import json
import os

import rclpy
from rclpy.node import Node

from ros_gz_interfaces.srv import SpawnEntity


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

        self.spawn_client = self.create_client(
            SpawnEntity,
            f"/world/{self.world_name}/create"
        )

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
        obstacle_type = obstacle.get("type", "BOX").upper()

        x = float(obstacle.get("x", 0.0))
        y = float(obstacle.get("y", 0.0))
        z = float(obstacle.get("z", 0.35))
        yaw = float(obstacle.get("yaw", 0.0))

        size = obstacle.get("size", {})
        sx = float(size.get("x", 1.0))
        sy = float(size.get("y", 1.0))
        sz = float(size.get("z", 0.7))

        if not self.spawn_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(
                f"Servizio Gazebo non disponibile: /world/{self.world_name}/create"
            )

        req = SpawnEntity.Request()
        req.name = obstacle_id
        req.allow_renaming = False
        req.sdf = self.build_obstacle_sdf(
            obstacle_id=obstacle_id,
            obstacle_type=obstacle_type,
            x=x,
            y=y,
            z=z,
            yaw=yaw,
            sx=sx,
            sy=sy,
            sz=sz
        )

        future = self.spawn_client.call_async(req)
        future.add_done_callback(
            lambda fut: self.on_spawn_result(fut, obstacle_id)
        )

        self.get_logger().info(
            f"spawn richiesto per ostacolo '{obstacle_id}' "
            f"tipo={obstacle_type} pos=({x:.2f},{y:.2f},{z:.2f}) "
            f"size=({sx:.2f},{sy:.2f},{sz:.2f})"
        )

    def on_spawn_result(self, future, obstacle_id):
        try:
            result = future.result()
        except Exception as ex:
            self.get_logger().error(
                f"{obstacle_id}: errore durante spawn: {ex}"
            )
            rclpy.shutdown()
            return

        if result.success:
            self.get_logger().info(
                f"{obstacle_id}: ostacolo spawnato correttamente"
            )
        else:
            self.get_logger().error(
                f"{obstacle_id}: spawn fallito: {result.status_message}"
            )

        rclpy.shutdown()

    # ------------------------------------------------------------
    # SDF
    # ------------------------------------------------------------

    def build_obstacle_sdf(
        self,
        obstacle_id,
        obstacle_type,
        x,
        y,
        z,
        yaw,
        sx,
        sy,
        sz
    ):
        if obstacle_type == "BARRIER":
            ambient = "0.95 0.25 0.05 1"
            diffuse = "0.95 0.25 0.05 1"
        elif obstacle_type == "BOX":
            ambient = "0.45 0.30 0.15 1"
            diffuse = "0.45 0.30 0.15 1"
        else:
            ambient = "0.45 0.45 0.45 1"
            diffuse = "0.45 0.45 0.45 1"

        mass = max(1.0, sx * sy * sz * 20.0)

        return f"""
<sdf version="1.9">
  <model name="{obstacle_id}">
    <static>true</static>
    <pose>{x} {y} {z} 0 0 {yaw}</pose>

    <link name="base_link">
      <inertial>
        <mass>{mass}</mass>
        <inertia>
          <ixx>1.0</ixx>
          <iyy>1.0</iyy>
          <izz>1.0</izz>
        </inertia>
      </inertial>

      <collision name="collision">
        <geometry>
          <box>
            <size>{sx} {sy} {sz}</size>
          </box>
        </geometry>
      </collision>

      <visual name="visual">
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