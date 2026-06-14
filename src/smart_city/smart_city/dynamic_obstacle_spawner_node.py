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
    side_x: float
    side_y: float
    yaw: float

    speed: float
    distance: float
    coverage_width: float
    barrier_thickness: float

    progress: float
    spawn_time: float
    last_update_time: float

    finished_since: Optional[float] = None
    removed: bool = False
    waiting_for_vehicle: Optional[str] = None

    moving: bool = False
    last_commanded_speed: Optional[float] = None
    last_motion_command_time: float = 0.0
    last_remove_attempt_time: float = 0.0

    road_hold_progress: float = 0.5
    road_hold_sec: float = 0.0
    road_hold_started_at: Optional[float] = None
    road_hold_done: bool = False


class DynamicObstacleSpawnerNode(Node):
    """
    Ostacoli pedonali leggeri per attraversamenti.

    Versione senza set_pose continuo:
    - spawn UNA volta tramite /world/<world>/create;
    - ogni modello ha VelocityControl;
    - il modello e' orientato verso la fine dell'attraversamento;
    - viene comandato solo linear.x nel frame locale del modello;
    - il punto finale viene allungato con travel_extra_distance per evitare rimozioni premature;
    - il pedone si ferma per qualche secondo in mezzo alla strada prima di completare l'attraversamento;
    - la collisione e' larga lateralmente, come una piccola fascia di pedoni,
      cosi' copre tutta la carreggiata invece di un solo puntino;
    - il modello e' sospeso leggermente, senza gravita' e con fisica minima.
    """

    def __init__(self):
        super().__init__("dynamic_obstacle_spawner_node")

        self.declare_parameter("crossing_id", "")
        self.declare_parameter("pedestrians_config_file", "config/pedestrian_crossings.json")
        self.declare_parameter("world_name", "smart_city_world")

        self.declare_parameter("global_time_scale", 2.5)
        self.declare_parameter("update_period_sec", 0.15)
        self.declare_parameter("max_step_dt_sec", 0.25)

        self.declare_parameter("vehicle_states_topic", "/vehicle_states")
        self.declare_parameter("vehicle_state_stale_timeout_sec", 1.5)

        self.declare_parameter("vehicle_wait_lookahead", 4.2)
        self.declare_parameter("vehicle_wait_behind", 0.6)
        self.declare_parameter("vehicle_wait_corridor_width", 1.7)
        self.declare_parameter("vehicle_emergency_radius", 1.8)

        self.declare_parameter("spawn_clearance_radius", 2.2)
        self.declare_parameter("pedestrian_spacing", 1.2)

        self.declare_parameter("default_pedestrian_speed_mps", 0.45)
        self.declare_parameter("remove_margin_sec", 0.8)
        self.declare_parameter("remove_retry_sec", 1.0)

        self.declare_parameter("default_crossing_coverage_width", 5.2)
        self.declare_parameter("default_barrier_thickness", 0.55)

        self.declare_parameter("default_travel_extra_distance", 28.0)

        self.declare_parameter("default_road_hold_sec", 5.0)
        self.declare_parameter("default_road_hold_offset_m", 0.0)

        self.declare_parameter("pedestrian_visual_count", 1)

        self.declare_parameter("initial_command_warmup_sec", 1.5)
        self.declare_parameter("initial_command_refresh_sec", 0.35)
        self.declare_parameter("motion_command_refresh_sec", 0.25)

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
        self.remove_retry_sec = float(self.get_parameter("remove_retry_sec").value)

        self.default_crossing_coverage_width = float(
            self.get_parameter("default_crossing_coverage_width").value
        )
        self.default_barrier_thickness = float(
            self.get_parameter("default_barrier_thickness").value
        )
        self.default_travel_extra_distance = float(
            self.get_parameter("default_travel_extra_distance").value
        )
        self.default_road_hold_sec = float(
            self.get_parameter("default_road_hold_sec").value
        )
        self.default_road_hold_offset_m = float(
            self.get_parameter("default_road_hold_offset_m").value
        )
        self.pedestrian_visual_count = int(self.get_parameter("pedestrian_visual_count").value)

        self.initial_command_warmup_sec = float(
            self.get_parameter("initial_command_warmup_sec").value
        )
        self.initial_command_refresh_sec = float(
            self.get_parameter("initial_command_refresh_sec").value
        )
        self.motion_command_refresh_sec = float(
            self.get_parameter("motion_command_refresh_sec").value
        )

        if not self.crossing_id:
            raise RuntimeError("Parametro obbligatorio mancante: crossing_id")

        self.crossing = self.load_crossing_by_id(self.crossing_id)

        self.active: dict[str, PedestrianInstance] = {}
        self.other_vehicles: dict[str, dict] = {}

        self.counter = 0
        self.last_spawn_time = 0.0

        self.vehicle_state_sub = self.create_subscription(
            String,
            self.vehicle_states_topic,
            self.on_vehicle_state,
            100,
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
        start_x, start_y, _, _, _, _, _, _, _, _ = self.compute_crossing_geometry(direction)

        if not self.can_spawn_at(start_x, start_y, now):
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
            side_x,
            side_y,
            yaw,
            distance,
        ) = self.compute_crossing_geometry(direction)

        config_speed = float(
            self.crossing.get(
                "pedestrian_speed_mps",
                self.default_pedestrian_speed_mps,
            )
        )
        speed = max(0.05, config_speed / self.global_time_scale)

        coverage_width = float(
            self.crossing.get(
                "coverage_width",
                self.crossing.get(
                    "crossing_coverage_width",
                    self.crossing.get(
                        "road_width",
                        self.default_crossing_coverage_width,
                    ),
                ),
            )
        )
        coverage_width = max(1.0, coverage_width)

        barrier_thickness = float(
            self.crossing.get(
                "barrier_thickness",
                self.default_barrier_thickness,
            )
        )
        barrier_thickness = max(0.25, barrier_thickness)

        road_hold_sec = float(
            self.crossing.get(
                "road_hold_sec",
                self.crossing.get(
                    "middle_road_wait_sec",
                    self.crossing.get("hold_in_road_sec", self.default_road_hold_sec),
                ),
            )
        )
        road_hold_sec = max(0.0, road_hold_sec) * self.global_time_scale

        road_hold_progress = self.compute_road_hold_progress(
            start_x=start_x,
            start_y=start_y,
            dir_x=dir_x,
            dir_y=dir_y,
            distance=distance,
        )

        sdf = textwrap.dedent(
            self.build_pedestrian_sdf(
                ped_name=ped_name,
                x=start_x,
                y=start_y,
                yaw=yaw,
                coverage_width=coverage_width,
                barrier_thickness=barrier_thickness,
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
            side_x=side_x,
            side_y=side_y,
            yaw=yaw,
            speed=speed,
            distance=distance,
            coverage_width=coverage_width,
            barrier_thickness=barrier_thickness,
            progress=0.0,
            spawn_time=now,
            last_update_time=now,
            road_hold_progress=road_hold_progress,
            road_hold_sec=road_hold_sec,
        )

        self.active[ped_name] = ped

        self.set_pedestrian_motion(ped, moving=True, now=now, force=True)

    # ------------------------------------------------------------
    # UPDATE + CLEANUP
    # ------------------------------------------------------------

    def update_and_cleanup_pedestrians(self, now):
        to_delete_from_local_state = []

        for ped_name, ped in list(self.active.items()):
            if ped.removed:
                to_delete_from_local_state.append(ped_name)
                continue

            raw_dt = now - ped.last_update_time
            dt = self.clamp(raw_dt, 0.0, self.max_step_dt_sec)
            ped.last_update_time = now

            x, y = self.interpolate_pedestrian_position(ped)

            if ped.progress >= 1.0:
                self.set_pedestrian_motion(ped, moving=False, now=now)

                if ped.finished_since is None:
                    ped.finished_since = now

                if now - ped.finished_since >= self.remove_margin_sec * self.global_time_scale:
                    removed = self.remove_pedestrian_from_gazebo(ped, now)
                    if removed:
                        to_delete_from_local_state.append(ped_name)

                continue

            if self.update_road_hold(ped, now):
                continue

            x, y = self.interpolate_pedestrian_position(ped)

            blocking_vehicle = self.find_blocking_vehicle_for_pedestrian(ped, x, y, now)
            blocking_pedestrian = self.find_blocking_pedestrian_for_pedestrian(ped, x, y)

            if blocking_vehicle is not None:
                ped.waiting_for_vehicle = blocking_vehicle
                self.set_pedestrian_motion(ped, moving=False, now=now)
                continue

            if blocking_pedestrian is not None:
                ped.waiting_for_vehicle = f"pedestrian:{blocking_pedestrian}"
                self.set_pedestrian_motion(ped, moving=False, now=now)
                continue

            ped.waiting_for_vehicle = None

            step = (ped.speed * dt) / max(ped.distance, 0.001)
            ped.progress = self.clamp(ped.progress + step, 0.0, 1.0)

            self.set_pedestrian_motion(ped, moving=True, now=now)

        for ped_name in to_delete_from_local_state:
            self.active.pop(ped_name, None)

    def interpolate_pedestrian_position(self, ped: PedestrianInstance):
        x = ped.start_x + (ped.end_x - ped.start_x) * ped.progress
        y = ped.start_y + (ped.end_y - ped.start_y) * ped.progress
        return x, y

    def update_road_hold(self, ped: PedestrianInstance, now: float):
        """
        Pausa controllata in mezzo alla carreggiata.

        La pausa viene fatta una sola volta, quando la progressione stimata
        raggiunge il punto centrale dell'attraversamento. Durante la pausa
        last_update_time viene comunque aggiornato dal loop principale, quindi
        quando il pedone riparte non recupera tutto il tempo fermo in un solo tick.
        """

        if ped.road_hold_done or ped.road_hold_sec <= 0.0:
            return False

        if ped.road_hold_started_at is not None:
            self.set_pedestrian_motion(ped, moving=False, now=now)

            if now - ped.road_hold_started_at >= ped.road_hold_sec:
                ped.road_hold_started_at = None
                ped.road_hold_done = True
                return False

            return True

        if ped.progress >= ped.road_hold_progress:
            ped.progress = ped.road_hold_progress
            ped.road_hold_started_at = now
            self.set_pedestrian_motion(ped, moving=False, now=now, force=True)
            return True

        return False

    # ------------------------------------------------------------
    # VEHICLE / PEDESTRIAN AVOIDANCE
    # ------------------------------------------------------------

    def can_spawn_at(self, x, y, now):
        # Uso la geometria del prossimo pedone per considerare la fascia larga.
        direction = "FORWARD" if self.counter % 2 == 0 else "BACKWARD"
        (
            start_x,
            start_y,
            _,
            _,
            dir_x,
            dir_y,
            side_x,
            side_y,
            _,
            _,
        ) = self.compute_crossing_geometry(direction)

        coverage_width = float(
            self.crossing.get(
                "coverage_width",
                self.crossing.get(
                    "crossing_coverage_width",
                    self.crossing.get("road_width", self.default_crossing_coverage_width),
                ),
            )
        )
        barrier_thickness = float(self.crossing.get("barrier_thickness", self.default_barrier_thickness))

        for _, vehicle in self.other_vehicles.items():
            vx = float(vehicle.get("x", 0.0))
            vy = float(vehicle.get("y", 0.0))

            along = (vx - start_x) * dir_x + (vy - start_y) * dir_y
            side = abs((vx - start_x) * side_x + (vy - start_y) * side_y)

            if (
                abs(along) <= self.spawn_clearance_radius + barrier_thickness * 0.5
                and side <= self.spawn_clearance_radius + coverage_width * 0.5
            ):
                return False

        for _, ped in self.active.items():
            px, py = self.interpolate_pedestrian_position(ped)
            if self.distance_xy(start_x, start_y, px, py) <= self.pedestrian_spacing:
                return False

        return True

    def find_blocking_vehicle_for_pedestrian(self, ped: PedestrianInstance, x, y, now):
        """
        Il pedone/fascia guarda davanti lungo la traiettoria.
        La larghezza laterale tiene conto della fascia che copre la carreggiata.
        """

        for vehicle_id, vehicle in self.other_vehicles.items():
            vx = float(vehicle.get("x", 0.0))
            vy = float(vehicle.get("y", 0.0))

            dx = vx - x
            dy = vy - y

            along = dx * ped.dir_x + dy * ped.dir_y
            side = abs(dx * ped.side_x + dy * ped.side_y)
            euclidean = math.sqrt(dx * dx + dy * dy)

            emergency_close = euclidean <= self.vehicle_emergency_radius

            # La fascia e' larga: se il veicolo e' dentro la carreggiata coperta,
            # il pedone/gruppo aspetta invece di attraversare davanti.
            lateral_limit = max(
                self.vehicle_wait_corridor_width,
                ped.coverage_width * 0.5 + 0.35,
            )

            in_front_corridor = (
                -self.vehicle_wait_behind <= along <= self.vehicle_wait_lookahead
                and side <= lateral_limit
            )

            if emergency_close or in_front_corridor:
                return vehicle_id

        return None

    def find_blocking_pedestrian_for_pedestrian(self, ped: PedestrianInstance, x, y):
        for other_name, other in self.active.items():
            if other_name == ped.name:
                continue

            ox, oy = self.interpolate_pedestrian_position(other)

            along = (ox - x) * ped.dir_x + (oy - y) * ped.dir_y
            side = abs((ox - x) * ped.side_x + (oy - y) * ped.side_y)

            same_band = side <= (ped.coverage_width + other.coverage_width) * 0.5
            too_close_along = abs(along) <= self.pedestrian_spacing + ped.barrier_thickness

            if same_band and too_close_along:
                return other_name

        return None

    # ------------------------------------------------------------
    # GAZEBO CMD_VEL / REMOVE
    # ------------------------------------------------------------

    def set_pedestrian_motion(
        self,
        ped: PedestrianInstance,
        moving: bool,
        now: float,
        force: bool = False,
    ):
        # IMPORTANTISSIMO:
        # VelocityControl usa la velocita' nel frame del modello.
        # Il modello e' gia' ruotato verso end_x/end_y, quindi si comanda solo linear.x.
        speed_cmd = ped.speed if moving else 0.0

        same_command = (
            ped.last_commanded_speed is not None
            and abs(speed_cmd - ped.last_commanded_speed) < 0.001
        )

        warmup_active = now - ped.spawn_time <= self.initial_command_warmup_sec
        refresh_sec = (
            self.initial_command_refresh_sec
            if warmup_active
            else self.motion_command_refresh_sec
        )
        refresh_due = now - ped.last_motion_command_time >= refresh_sec

        if same_command and not force and not refresh_due:
            ped.moving = moving
            return

        ped.last_commanded_speed = speed_cmd
        ped.last_motion_command_time = now
        ped.moving = moving

        self.command_pedestrian_velocity(ped, speed_cmd)

    def command_pedestrian_velocity(self, ped: PedestrianInstance, speed: float):
        topic = f"/model/{ped.name}/cmd_vel"

        msg = (
            f"linear {{ x: {speed:.6f} y: 0.0 z: 0.0 }} "
            f"angular {{ x: 0.0 y: 0.0 z: 0.0 }}"
        )

        cmd = [
            "gz", "topic",
            "-t", topic,
            "-m", "gz.msgs.Twist",
            "-p", msg,
        ]

        try:
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass

    def remove_pedestrian_from_gazebo(self, ped: PedestrianInstance, now: float):
        if ped.removed:
            return True

        if now - ped.last_remove_attempt_time < self.remove_retry_sec:
            return False

        ped.last_remove_attempt_time = now
        self.set_pedestrian_motion(ped, moving=False, now=now, force=True)

        req = f'name: "{ped.name}" type: 2'

        ok = self.call_gz_service(
            service=f"/world/{self.world_name}/remove",
            reqtype="gz.msgs.Entity",
            reptype="gz.msgs.Boolean",
            req=req,
        )

        if ok:
            ped.removed = True
            return True

        return False

    # ------------------------------------------------------------
    # GEOMETRIA
    # ------------------------------------------------------------

    def compute_road_hold_progress(self, start_x, start_y, dir_x, dir_y, distance):
        # Default: fermati nel centro geometrico del crossing, cioe'
        # normalmente nel mezzo della carreggiata.
        cx = float(self.crossing.get("x", 0.0))
        cy = float(self.crossing.get("y", 0.0))

        offset = float(
            self.crossing.get(
                "road_hold_offset_m",
                self.crossing.get("middle_road_wait_offset_m", self.default_road_hold_offset_m),
            )
        )

        along_to_center = (cx - start_x) * dir_x + (cy - start_y) * dir_y
        hold_distance = along_to_center + offset

        explicit_progress = self.crossing.get("road_hold_progress", None)
        if explicit_progress is not None:
            try:
                return self.clamp(float(explicit_progress), 0.02, 0.95)
            except (TypeError, ValueError):
                pass

        return self.clamp(hold_distance / max(distance, 0.001), 0.05, 0.90)

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

        # Il punto di spawn resta quello del JSON, ma il punto finale viene
        # allungato nella stessa direzione di attraversamento.
        # In questo modo il pedone non viene rimosso appena la stima interna
        # pensa che abbia finito: continua abbastanza da entrare e attraversare
        # davvero la carreggiata.
        travel_extra = float(
            self.crossing.get(
                "travel_extra_distance",
                self.crossing.get("lifecycle_extra_distance", self.default_travel_extra_distance),
            )
        )
        travel_extra = max(0.0, travel_extra)

        if b >= a:
            b = b + travel_extra
        else:
            b = b - travel_extra

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

        # Vettore laterale della fascia, perpendicolare al movimento.
        side_x = -dir_y
        side_y = dir_x

        yaw = math.atan2(dir_y, dir_x)

        return start_x, start_y, end_x, end_y, dir_x, dir_y, side_x, side_y, yaw, distance

    @staticmethod
    def distance_xy(x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        return math.sqrt(dx * dx + dy * dy)

    @staticmethod
    def clamp(value, min_value, max_value):
        return max(min_value, min(max_value, value))

    # ------------------------------------------------------------
    # GAZEBO SERVICE: solo create/remove, mai nel movimento normale
    # ------------------------------------------------------------

    def call_gz_service(self, service, reqtype, reptype, req):
        cmd = [
            "gz", "service",
            "-s", service,
            "--reqtype", reqtype,
            "--reptype", reptype,
            "--timeout", "5000",
            "--req", req,
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=6.0,
            )

            if result.returncode != 0:
                return False

            return "data: true" in result.stdout

        except Exception:
            return False

    # ------------------------------------------------------------
    # SDF
    # ------------------------------------------------------------

    def build_pedestrian_sdf(
        self,
        ped_name,
        x,
        y,
        yaw,
        coverage_width,
        barrier_thickness,
    ):
        """
        Pedone singolo visibile + collisione invisibile larga.

        Nota importante:
        - il visual resta UNO, quindi non sembra che il nodo generi 5 pedoni;
        - la collisione larga serve solo al LiDAR / obstacle detection;
        - collide_bitmask=0 evita che il pedone fisico venga spinto, incastrato
          o fatto sparire quando entra nella carreggiata;
        - gravity=false e posa leggermente alta lo tengono sospeso.
        """
        return f"""
<sdf version="1.9">
  <model name="{ped_name}">
    <static>false</static>
    <pose>{x:.6f} {y:.6f} 0.22 0 0 {yaw:.6f}</pose>

    <plugin
      filename="gz-sim-velocity-control-system"
      name="gz::sim::systems::VelocityControl">
      <topic>/model/{ped_name}/cmd_vel</topic>
    </plugin>

    <link name="base_link">
      <gravity>false</gravity>
      <kinematic>true</kinematic>
      <self_collide>false</self_collide>

      <inertial>
        <mass>0.05</mass>
        <inertia>
          <ixx>0.001</ixx>
          <iyy>0.001</iyy>
          <izz>0.001</izz>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyz>0.0</iyz>
        </inertia>
      </inertial>

      <!--
        Collisione larga ma invisibile.
        X locale = direzione attraversamento.
        Y locale = larghezza coperta sulla carreggiata.
        collide_bitmask=0 disattiva i contatti fisici, ma la geometria resta
        presente per i ray/LiDAR nella maggior parte delle configurazioni Gazebo.
      -->
      <collision name="lidar_crosswalk_band_collision">
        <pose>0 0 0.48 0 0 0</pose>
        <geometry>
          <box>
            <size>{barrier_thickness:.3f} {coverage_width:.3f} 0.75</size>
          </box>
        </geometry>
        <surface>
          <contact>
            <collide_bitmask>0x00</collide_bitmask>
          </contact>
        </surface>
      </collision>

      <!-- Singolo pedone visibile, non una fila da 5. -->
      <visual name="body">
        <pose>0 0 0.58 0 0 0</pose>
        <geometry>
          <cylinder>
            <radius>0.18</radius>
            <length>0.86</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.1 0.85 0.1 1</ambient>
          <diffuse>0.1 0.85 0.1 1</diffuse>
        </material>
      </visual>

      <visual name="head">
        <pose>0 0 1.12 0 0 0</pose>
        <geometry>
          <sphere>
            <radius>0.15</radius>
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
