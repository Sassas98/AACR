import json
import math
import os
import subprocess
import tempfile
import textwrap
import time
from dataclasses import dataclass
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


@dataclass
class PedestrianInstance:
    name: str
    direction: str

    start_x: float
    start_y: float
    end_x: float
    end_y: float

    dir_x: float
    dir_y: float
    yaw: float

    speed: float
    distance: float

    progress: float
    spawn_time: float
    last_update_time: float

    finished_since: Optional[float] = None
    removed: bool = False
    waiting_for_vehicle: Optional[str] = None


class DynamicObstacleSpawnerNode(Node):
    """
    Pedoni cinematici che attraversano la carreggiata.

    - niente cmd_vel;
    - niente bridge dinamici;
    - niente fisica instabile;
    - movimento graduale via set_pose;
    - lettura /vehicle_states per evitare di attraversare davanti ai veicoli;
    - collisione presente, quindi il LiDAR dei veicoli dovrebbe rilevarli.
    """

    def __init__(self):
        super().__init__("dynamic_obstacle_spawner_node")

        self.declare_parameter("crossing_id", "")
        self.declare_parameter("pedestrians_config_file", "config/pedestrian_crossings.json")
        self.declare_parameter("world_name", "smart_city_world")

        # Più è alto, più rallenta tutto: spawn, attraversamento, removal.
        self.declare_parameter("global_time_scale", 2.5)

        # Aggiornamenti più frequenti = movimento meno a scatti.
        self.declare_parameter("update_period_sec", 0.2)

        # Evita salti grossi se il servizio Gazebo lagga.
        self.declare_parameter("max_step_dt_sec", 0.06)

        self.declare_parameter("vehicle_states_topic", "/vehicle_states")
        self.declare_parameter("vehicle_state_stale_timeout_sec", 1.5)

        # Sicurezza pedone-veicolo.
        self.declare_parameter("vehicle_wait_lookahead", 4.2)
        self.declare_parameter("vehicle_wait_behind", 0.6)
        self.declare_parameter("vehicle_wait_corridor_width", 1.7)
        self.declare_parameter("vehicle_emergency_radius", 1.8)

        # Evita spawn dentro veicoli o altri pedoni.
        self.declare_parameter("spawn_clearance_radius", 2.2)
        self.declare_parameter("pedestrian_spacing", 1.2)

        # Default lento, se nel JSON manca.
        self.declare_parameter("default_pedestrian_speed_mps", 0.45)

        # Tempo extra dopo arrivo prima di rimuovere.
        self.declare_parameter("remove_margin_sec", 0.8)

        self.crossing_id = str(self.get_parameter("crossing_id").value)
        self.config_file = str(self.get_parameter("pedestrians_config_file").value)
        self.world_name = str(self.get_parameter("world_name").value)

        self.global_time_scale = max(0.1, float(self.get_parameter("global_time_scale").value))
        self.update_period_sec = float(self.get_parameter("update_period_sec").value)
        self.max_step_dt_sec = float(self.get_parameter("max_step_dt_sec").value)

        self.vehicle_states_topic = str(self.get_parameter("vehicle_states_topic").value)
        self.vehicle_state_stale_timeout_sec = float(
            self.get_parameter("vehicle_state_stale_timeout_sec").value
        )

        self.vehicle_wait_lookahead = float(self.get_parameter("vehicle_wait_lookahead").value)
        self.vehicle_wait_behind = float(self.get_parameter("vehicle_wait_behind").value)
        self.vehicle_wait_corridor_width = float(
            self.get_parameter("vehicle_wait_corridor_width").value
        )
        self.vehicle_emergency_radius = float(self.get_parameter("vehicle_emergency_radius").value)

        self.spawn_clearance_radius = float(self.get_parameter("spawn_clearance_radius").value)
        self.pedestrian_spacing = float(self.get_parameter("pedestrian_spacing").value)

        self.default_pedestrian_speed_mps = float(
            self.get_parameter("default_pedestrian_speed_mps").value
        )
        self.remove_margin_sec = float(self.get_parameter("remove_margin_sec").value)

        if not self.crossing_id:
            raise RuntimeError("Parametro obbligatorio mancante: crossing_id")

        self.crossing = self.load_crossing_by_id(self.crossing_id)

        self.active: dict[str, PedestrianInstance] = {}
        self.other_vehicles: dict[str, dict] = {}

        self.counter = 0
        self.last_spawn_time = 0.0
        self.last_spawn_block_log_time = 0.0
        self.last_set_pose_error_time = 0.0

        self.vehicle_state_sub = self.create_subscription(
            String,
            self.vehicle_states_topic,
            self.on_vehicle_state,
            100
        )

        self.timer = self.create_timer(self.update_period_sec, self.loop)

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
    # VEHICLE STATE
    # ------------------------------------------------------------

    def on_vehicle_state(self, msg):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        vehicle_id = data.get("vehicle_id")
        if not vehicle_id:
            return

        self.other_vehicles[str(vehicle_id)] = data

    def cleanup_stale_vehicles(self, now):
        expired = []

        for vehicle_id, vehicle in self.other_vehicles.items():
            stamp = float(vehicle.get("stamp", 0.0))
            if now - stamp > self.vehicle_state_stale_timeout_sec:
                expired.append(vehicle_id)

        for vehicle_id in expired:
            self.other_vehicles.pop(vehicle_id, None)

    # ------------------------------------------------------------
    # LOOP
    # ------------------------------------------------------------

    def loop(self):
        now = time.time()

        self.cleanup_stale_vehicles(now)
        self.update_and_cleanup_pedestrians(now)
        self.maybe_spawn_pedestrian(now)

    def maybe_spawn_pedestrian(self, now):
        spawn_every = float(self.crossing.get("spawn_every_sec", 8.0))
        spawn_every *= self.global_time_scale

        max_active = int(self.crossing.get("max_active_pedestrians", 1))

        if len(self.active) >= max_active:
            return

        if now - self.last_spawn_time < spawn_every:
            return

        direction = "FORWARD" if self.counter % 2 == 0 else "BACKWARD"
        start_x, start_y, _, _, _, _, _, _ = self.compute_crossing_geometry(direction)

        if not self.can_spawn_at(start_x, start_y, now):
            if now - self.last_spawn_block_log_time > 2.0:
                self.last_spawn_block_log_time = now
            return

        self.spawn_pedestrian(now, direction)

    # ------------------------------------------------------------
    # SPAWN
    # ------------------------------------------------------------

    def spawn_pedestrian(self, now, direction):
        self.counter += 1

        ped_name = f"{self.crossing_id}_ped_{self.counter}"

        (
            start_x,
            start_y,
            end_x,
            end_y,
            dir_x,
            dir_y,
            yaw,
            distance,
        ) = self.compute_crossing_geometry(direction)

        config_speed = float(
            self.crossing.get(
                "pedestrian_speed_mps",
                self.default_pedestrian_speed_mps
            )
        )

        # Rallento i pedoni rispetto al mondo.
        speed = max(0.05, config_speed / self.global_time_scale)

        sdf = textwrap.dedent(
            self.build_pedestrian_sdf(
                ped_name=ped_name,
                x=start_x,
                y=start_y,
                yaw=yaw,
            )
        )

        sdf_path = os.path.join(tempfile.gettempdir(), f"{ped_name}.sdf")

        with open(sdf_path, "w", encoding="utf-8") as f:
            f.write(sdf)

        req = (
            f'sdf_filename: "{sdf_path}" '
            f'name: "{ped_name}" '
            f'allow_renaming: false'
        )

        ok = self.call_gz_service(
            service=f"/world/{self.world_name}/create",
            reqtype="gz.msgs.EntityFactory",
            reptype="gz.msgs.Boolean",
            req=req,
            entity_id=ped_name,
            action_name="spawn",
            log_success=True,
            log_failure=True,
        )

        self.last_spawn_time = now

        if not ok:
            return

        ped = PedestrianInstance(
            name=ped_name,
            direction=direction,
            start_x=start_x,
            start_y=start_y,
            end_x=end_x,
            end_y=end_y,
            dir_x=dir_x,
            dir_y=dir_y,
            yaw=yaw,
            speed=speed,
            distance=distance,
            progress=0.0,
            spawn_time=now,
            last_update_time=now,
        )

        self.active[ped_name] = ped

    # ------------------------------------------------------------
    # UPDATE + CLEANUP
    # ------------------------------------------------------------

    def update_and_cleanup_pedestrians(self, now):
        finished = []

        for ped_name, ped in list(self.active.items()):
            if ped.removed:
                finished.append(ped_name)
                continue

            if ped.progress >= 1.0:
                if ped.finished_since is None:
                    ped.finished_since = now

                if now - ped.finished_since >= self.remove_margin_sec * self.global_time_scale:
                    self.remove_pedestrian_from_gazebo(ped)
                    finished.append(ped_name)

                continue

            raw_dt = now - ped.last_update_time
            dt = self.clamp(raw_dt, 0.0, self.max_step_dt_sec)
            ped.last_update_time = now

            x, y = self.interpolate_pedestrian_position(ped)

            blocking_vehicle = self.find_blocking_vehicle_for_pedestrian(ped, x, y, now)
            blocking_pedestrian = self.find_blocking_pedestrian_for_pedestrian(ped, x, y)

            if blocking_vehicle is not None:
                ped.waiting_for_vehicle = blocking_vehicle
                self.set_pedestrian_pose(ped, x, y)
                continue

            if blocking_pedestrian is not None:
                ped.waiting_for_vehicle = f"pedestrian:{blocking_pedestrian}"
                self.set_pedestrian_pose(ped, x, y)
                continue

            ped.waiting_for_vehicle = None

            step = (ped.speed * dt) / max(ped.distance, 0.001)
            ped.progress = self.clamp(ped.progress + step, 0.0, 1.0)

            x, y = self.interpolate_pedestrian_position(ped)
            self.set_pedestrian_pose(ped, x, y)

        for ped_name in finished:
            self.active.pop(ped_name, None)

    def interpolate_pedestrian_position(self, ped: PedestrianInstance):
        x = ped.start_x + (ped.end_x - ped.start_x) * ped.progress
        y = ped.start_y + (ped.end_y - ped.start_y) * ped.progress
        return x, y

    # ------------------------------------------------------------
    # VEHICLE / PEDESTRIAN AVOIDANCE
    # ------------------------------------------------------------

    def can_spawn_at(self, x, y, now):
        for _, vehicle in self.other_vehicles.items():
            vx = float(vehicle.get("x", 0.0))
            vy = float(vehicle.get("y", 0.0))

            if self.distance_xy(x, y, vx, vy) <= self.spawn_clearance_radius:
                return False

        for _, ped in self.active.items():
            px, py = self.interpolate_pedestrian_position(ped)
            if self.distance_xy(x, y, px, py) <= self.pedestrian_spacing:
                return False

        return True

    def find_blocking_vehicle_for_pedestrian(self, ped: PedestrianInstance, x, y, now):
        """
        Il pedone guarda davanti lungo la sua traiettoria.
        Se c'è un veicolo nella zona di attraversamento, aspetta.
        """

        for vehicle_id, vehicle in self.other_vehicles.items():
            vx = float(vehicle.get("x", 0.0))
            vy = float(vehicle.get("y", 0.0))

            dx = vx - x
            dy = vy - y

            along = dx * ped.dir_x + dy * ped.dir_y
            side = abs(dx * (-ped.dir_y) + dy * ped.dir_x)
            euclidean = math.sqrt(dx * dx + dy * dy)

            emergency_close = euclidean <= self.vehicle_emergency_radius

            in_front_corridor = (
                -self.vehicle_wait_behind <= along <= self.vehicle_wait_lookahead
                and side <= self.vehicle_wait_corridor_width
            )

            if emergency_close or in_front_corridor:
                return vehicle_id

        return None

    def find_blocking_pedestrian_for_pedestrian(self, ped: PedestrianInstance, x, y):
        for other_name, other in self.active.items():
            if other_name == ped.name:
                continue

            ox, oy = self.interpolate_pedestrian_position(other)

            if self.distance_xy(x, y, ox, oy) <= self.pedestrian_spacing:
                return other_name

        return None

    # ------------------------------------------------------------
    # GAZEBO POSE / REMOVE
    # ------------------------------------------------------------

    def set_pedestrian_pose(self, ped: PedestrianInstance, x: float, y: float):
        qz = math.sin(ped.yaw / 2.0)
        qw = math.cos(ped.yaw / 2.0)

        req = (
            f'name: "{ped.name}" '
            f'position {{ x: {x:.6f} y: {y:.6f} z: 0.0 }} '
            f'orientation {{ x: 0.0 y: 0.0 z: {qz:.8f} w: {qw:.8f} }}'
        )

        ok = self.call_gz_service(
            service=f"/world/{self.world_name}/set_pose",
            reqtype="gz.msgs.Pose",
            reptype="gz.msgs.Boolean",
            req=req,
            entity_id=ped.name,
            action_name="set_pose",
            log_success=False,
            log_failure=False,
        )

        if not ok:
            now = time.time()
            if now - self.last_set_pose_error_time > 2.0:
                self.last_set_pose_error_time = now

    def remove_pedestrian_from_gazebo(self, ped: PedestrianInstance):
        if ped.removed:
            return

        ped.removed = True

        # gz.msgs.Entity: MODEL = 2.
        req = f'name: "{ped.name}" type: 2'

        ok = self.call_gz_service(
            service=f"/world/{self.world_name}/remove",
            reqtype="gz.msgs.Entity",
            reptype="gz.msgs.Boolean",
            req=req,
            entity_id=ped.name,
            action_name="remove",
            log_success=True,
            log_failure=True,
        )

    # ------------------------------------------------------------
    # GEOMETRIA
    # ------------------------------------------------------------

    def compute_crossing_geometry(self, direction):
        cx = float(self.crossing.get("x", 0.0))
        cy = float(self.crossing.get("y", 0.0))

        start_offset = float(self.crossing.get("start_offset", -4.0))
        end_offset = float(self.crossing.get("end_offset", 4.0))

        axis = str(self.crossing.get("crossing_axis", "Y")).upper()

        if direction == "FORWARD":
            a = start_offset
            b = end_offset
        else:
            a = end_offset
            b = start_offset

        if axis == "X":
            start_x = cx + a
            start_y = cy
            end_x = cx + b
            end_y = cy

        elif axis == "Y":
            start_x = cx
            start_y = cy + a
            end_x = cx
            end_y = cy + b

        else:
            raise RuntimeError(f"crossing_axis non valido: {axis}")

        dx = end_x - start_x
        dy = end_y - start_y
        distance = math.sqrt(dx * dx + dy * dy)

        if distance <= 0.001:
            raise RuntimeError(f"crossing '{self.crossing_id}' ha distanza nulla")

        dir_x = dx / distance
        dir_y = dy / distance
        yaw = math.atan2(dir_y, dir_x)

        return start_x, start_y, end_x, end_y, dir_x, dir_y, yaw, distance

    @staticmethod
    def distance_xy(x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        return math.sqrt(dx * dx + dy * dy)

    @staticmethod
    def clamp(value, min_value, max_value):
        return max(min_value, min(max_value, value))

    # ------------------------------------------------------------
    # GAZEBO SERVICE
    # ------------------------------------------------------------

    def call_gz_service(
        self,
        service,
        reqtype,
        reptype,
        req,
        entity_id,
        action_name,
        log_success=True,
        log_failure=True,
    ):
        cmd = [
            "gz", "service",
            "-s", service,
            "--reqtype", reqtype,
            "--reptype", reptype,
            "--timeout", "3000",
            "--req", req,
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=4.0,
            )

            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            if result.returncode != 0:
                return False

            if "data: true" not in stdout:
                return False

            return True

        except subprocess.TimeoutExpired:
            return False

        except Exception as ex:
            if log_failure:
                self.get_logger().error(
                    f"{entity_id}: errore durante {action_name}: {ex}"
                )
            return False

    # ------------------------------------------------------------
    # SDF
    # ------------------------------------------------------------

    def build_pedestrian_sdf(self, ped_name, x, y, yaw):
        """
        Collisione presente:
        - i LiDAR dovrebbero vederlo;
        - i veicoli possono evitarlo come ostacolo;
        - static=true evita cadute/rotolamenti;
        - movimento gestito via set_pose.
        """
        return f"""
<sdf version="1.9">
  <model name="{ped_name}">
    <static>true</static>
    <pose>{x:.6f} {y:.6f} 0.0 0 0 {yaw:.6f}</pose>

    <link name="base_link">
      <pose>0 0 0.70 0 0 0</pose>

      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>0.28</radius>
            <length>1.40</length>
          </cylinder>
        </geometry>
      </collision>

      <visual name="body">
        <geometry>
          <cylinder>
            <radius>0.24</radius>
            <length>1.20</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.1 0.85 0.1 1</ambient>
          <diffuse>0.1 0.85 0.1 1</diffuse>
        </material>
      </visual>

      <visual name="head">
        <pose>0 0 0.78 0 0 0</pose>
        <geometry>
          <sphere>
            <radius>0.18</radius>
          </sphere>
        </geometry>
        <material>
          <ambient>0.15 1.0 0.15 1</ambient>
          <diffuse>0.15 1.0 0.15 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""


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