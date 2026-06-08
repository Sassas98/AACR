import json
import math
import os
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import subprocess
import textwrap


class PedestrianInstance:
    def __init__(self, name, crossing, direction):
        self.name = name
        self.crossing = crossing
        self.direction = direction
        self.spawn_time = time.time()


class DynamicObstacleSpawnerNode(Node):

    def __init__(self):
        super().__init__("dynamic_obstacle_spawner_node")

        self.declare_parameter("crossing_id", "")
        self.declare_parameter(
            "pedestrians_config_file",
            "config/pedestrian_crossings.json"
        )
        self.declare_parameter("world_name", "smart_city_world")

        self.crossing_id = self.get_parameter("crossing_id").value
        self.config_file = self.get_parameter("pedestrians_config_file").value
        self.world_name = self.get_parameter("world_name").value

        if not self.crossing_id:
            raise RuntimeError("Parametro obbligatorio mancante: crossing_id")

        self.crossing = self.load_crossing_by_id(self.crossing_id)

        self.active = {}
        self.cmd_pubs = {}

        self.counter = 0
        self.last_spawn_time = 0.0

        self.timer = self.create_timer(0.1, self.loop)

        self.get_logger().info(
            f"dynamic_obstacle_spawner_node avviato per crossing '{self.crossing_id}'"
        )

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

    def load_crossing_by_id(self, crossing_id):
        data = self.load_json(self.config_file)

        for crossing in data.get("pedestrian_crossings", []):
            if crossing.get("id") == crossing_id:
                return crossing

        raise RuntimeError(
            f"Crossing '{crossing_id}' non trovato in {self.config_file}"
        )

    # ------------------------------------------------------------
    # LOOP
    # ------------------------------------------------------------

    def loop(self):
        now = time.time()

        self.cleanup_finished_pedestrians(now)

        spawn_every = float(self.crossing.get("spawn_every_sec", 5.0))
        max_active = int(self.crossing.get("max_active_pedestrians", 1))

        if len(self.active) >= max_active:
            return

        if now - self.last_spawn_time < spawn_every:
            return

        self.spawn_pedestrian()

    # ------------------------------------------------------------
    # SPAWN
    # ------------------------------------------------------------

    def spawn_pedestrian(self):
        self.counter += 1

        direction = "FORWARD" if self.counter % 2 == 1 else "BACKWARD"
        ped_name = f"{self.crossing_id}_ped_{self.counter}"

        x, y, yaw = self.compute_start_pose(direction)

        sdf = self.build_pedestrian_sdf(ped_name, x, y, yaw)
        sdf = " ".join(textwrap.dedent(sdf).split())

        req = f'sdf: "{sdf}", name: "{ped_name}"'

        ok = self.call_gz_service(
            service=f"/world/{self.world_name}/create",
            reqtype="gz.msgs.EntityFactory",
            reptype="gz.msgs.Boolean",
            req=req,
            entity_id=ped_name
        )

        self.last_spawn_time = time.time()

        if not ok:
            return

        self.active[ped_name] = PedestrianInstance(
            name=ped_name,
            crossing=self.crossing,
            direction=direction
        )

        self.get_logger().info(
            f"{ped_name}: pedone spawnato direction={direction}"
        )

    def call_gz_service(self, service, reqtype, reptype, req, entity_id):
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
                    f"{entity_id}: richiesta spawn inviata a Gazebo"
                )
                return True

            self.get_logger().error(
                f"{entity_id}: spawn fallito | "
                f"stdout={result.stdout} stderr={result.stderr}"
            )
            return False

        except Exception as ex:
            self.get_logger().error(
                f"{entity_id}: errore chiamata gz service: {ex}"
            )
            return False

    # ------------------------------------------------------------
    # MOVIMENTO
    # ------------------------------------------------------------

    def cleanup_finished_pedestrians(self, now):
        finished = []

        for ped_name, ped in list(self.active.items()):
            crossing_time = self.compute_crossing_time()
            elapsed = now - ped.spawn_time

            if elapsed >= crossing_time:
                self.stop_pedestrian(ped_name)
                finished.append(ped_name)
                continue

            self.publish_pedestrian_velocity(ped_name, ped.direction)

        for ped_name in finished:
            del self.active[ped_name]

    def get_or_create_cmd_pub(self, ped_name):
        if ped_name not in self.cmd_pubs:
            self.cmd_pubs[ped_name] = self.create_publisher(
                Twist,
                f"/{ped_name}/cmd_vel",
                10
            )

        return self.cmd_pubs[ped_name]

    def publish_pedestrian_velocity(self, ped_name, direction):
        pub = self.get_or_create_cmd_pub(ped_name)

        speed = float(self.crossing.get("pedestrian_speed_mps", 1.0))
        axis = self.crossing.get("crossing_axis", "Y").upper()

        sign = 1.0 if direction == "FORWARD" else -1.0

        cmd = Twist()

        if axis == "X":
            cmd.linear.x = speed * sign
        else:
            cmd.linear.y = speed * sign

        pub.publish(cmd)

    def stop_pedestrian(self, ped_name):
        pub = self.get_or_create_cmd_pub(ped_name)
        pub.publish(Twist())

    # ------------------------------------------------------------
    # GEOMETRIA
    # ------------------------------------------------------------

    def compute_crossing_time(self):
        start_offset = float(self.crossing.get("start_offset", -4.0))
        end_offset = float(self.crossing.get("end_offset", 4.0))
        speed = max(
            0.01,
            float(self.crossing.get("pedestrian_speed_mps", 1.0))
        )

        return abs(end_offset - start_offset) / speed

    def compute_start_pose(self, direction):
        x = float(self.crossing.get("x", 0.0))
        y = float(self.crossing.get("y", 0.0))

        start_offset = float(self.crossing.get("start_offset", -4.0))
        end_offset = float(self.crossing.get("end_offset", 4.0))

        axis = self.crossing.get("crossing_axis", "Y").upper()

        offset = start_offset if direction == "FORWARD" else end_offset

        if axis == "X":
            x += offset
            yaw = 0.0 if direction == "FORWARD" else math.pi
        else:
            y += offset
            yaw = math.pi / 2.0 if direction == "FORWARD" else -math.pi / 2.0

        return x, y, yaw

    # ------------------------------------------------------------
    # SDF
    # ------------------------------------------------------------

    def build_pedestrian_sdf(self, ped_name, x, y, yaw):
        return f"""
<sdf version="1.9">
  <model name="{ped_name}">
    <static>false</static>
    <self_collide>false</self_collide>
    <pose>{x} {y} 0.0 0 0 {yaw}</pose>

    <link name="base_link">
      <pose>0 0 0.55 0 0 0</pose>

      <inertial>
        <mass>70</mass>
        <inertia>
          <ixx>8</ixx>
          <iyy>8</iyy>
          <izz>2</izz>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyz>0</iyz>
        </inertia>
      </inertial>

      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>0.28</radius>
            <length>1.1</length>
          </cylinder>
        </geometry>
        <surface>
          <friction>
            <ode>
              <mu>1.0</mu>
              <mu2>1.0</mu2>
            </ode>
          </friction>
        </surface>
      </collision>

      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>0.28</radius>
            <length>1.1</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.1 0.8 0.1 1</ambient>
          <diffuse>0.1 0.8 0.1 1</diffuse>
        </material>
      </visual>
    </link>

    <plugin filename="gz-sim-velocity-control-system"
            name="gz::sim::systems::VelocityControl">
      <topic>/{ped_name}/cmd_vel</topic>
    </plugin>
  </model>
</sdf>
"""

    # ------------------------------------------------------------


def main(args=None):
    rclpy.init(args=args)

    node = DynamicObstacleSpawnerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()