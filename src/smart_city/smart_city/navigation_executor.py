import json
import math
import os
import heapq
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional
from std_srvs.srv import Trigger

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from smart_city_interfaces.action import NavigateToPose


class ExecutorState(Enum):
    IDLE = "IDLE"
    NAVIGATING = "NAVIGATING"
    WAITING_TRAFFIC_LIGHT = "WAITING_TRAFFIC_LIGHT"
    OBSTACLE_STOP = "OBSTACLE_STOP"
    RECALCULATING = "RECALCULATING"
    STUCK_RECOVERY = "STUCK_RECOVERY"


class DecisionType(Enum):
    """Azioni mutuamente esclusive: una sola policy comanda per ogni tick."""

    FOLLOW_LANE = "FOLLOW_LANE"
    SLOW_FOLLOW = "SLOW_FOLLOW"
    TEMPORARY_TARGET = "TEMPORARY_TARGET"
    STOP = "STOP"
    RESET_WAYPOINT_TIMER = "RESET_WAYPOINT_TIMER"
    VEHICLE_DEADLOCK_RECOVERY = "VEHICLE_DEADLOCK_RECOVERY"
    GENERIC_STUCK_RECOVERY = "GENERIC_STUCK_RECOVERY"
    LIDAR_OBSTACLE_RECOVERY = "LIDAR_OBSTACLE_RECOVERY"


@dataclass
class NavigationContext:
    goal: Any
    waypoint: Dict[str, Any]
    distance_to_wp: float
    elapsed_on_wp: float
    waypoint_timeout_sec: float


@dataclass
class NavigationDecision:
    type: DecisionType
    reason: str = ""
    speed_factor: float = 1.0
    target_x: Optional[float] = None
    target_y: Optional[float] = None
    payload: Dict[str, Any] = field(default_factory=dict)


class DeadlockPolicy:
    """
    Deadlock fisico vero.

    Non deve risolvere la precedenza: quella viene già decisa da VehicleSafetyPolicy.
    Qui interviene solo se il modello è fisicamente fermo e c'è un conflitto reale.
    """

    def __init__(self, node):
        self.node = node

    def evaluate(self, ctx: NavigationContext) -> Optional[NavigationDecision]:
        if self._looks_like_traffic_light_wait(ctx):
            return None

        self.node.update_motion_watchdog()
        conflict = self.node.detect_vehicle_deadlock()

        if conflict is None:
            return None

        other_id = conflict.get("vehicle_id")

        # Regola fondamentale:
        # se io devo cedere, NON faccio manovre strane.
        # Mi fermo e lascio l'altro liberare.
        if other_id and self.node.should_yield_to_vehicle(other_id):
            return NavigationDecision(
                DecisionType.STOP,
                reason=f"deadlock_yield_wait_{other_id}",
                payload={"conflict": conflict},
            )

        return NavigationDecision(
            DecisionType.VEHICLE_DEADLOCK_RECOVERY,
            reason="vehicle_deadlock_priority_recovery",
            payload={"conflict": conflict},
        )

    def _looks_like_traffic_light_wait(self, ctx: NavigationContext) -> bool:
        wp = ctx.waypoint
        node = self.node

        if wp.get("kind") != "approach_intersection":
            return False

        if ctx.distance_to_wp > node.traffic_light_stop_distance * 1.15:
            return False

        intersection_node_id = wp.get("node_id")
        from_node_id = wp.get("from_node_id")
        to_node_id = wp.get("to_node_id")

        if not intersection_node_id or not from_node_id:
            return False

        # Se questo nodo non è semaforizzato, non può essere una vera attesa semaforo.
        if intersection_node_id not in node.traffic_light_node_ids:
            return False

        # Se dovrebbe esserci un semaforo ma non ho ancora lo status,
        # lascio che sia TrafficLightPolicy a gestire l'attesa/timeout.
        # DeadlockPolicy deve solo evitare di scambiare questa attesa per un deadlock.
        if intersection_node_id not in node.traffic_light_statuses:
            return True

        return node.get_signal_color_for_branch(
            intersection_node_id,
            from_node_id,
            to_node_id
        ) != "green"


class LidarObstaclePolicy:
    """
    Ostacolo rilevato da LiDAR.

    Nuova logica:
    - se il LiDAR vede qualcosa davanti, mi fermo;
    - per 5 secondi osservo se la lettura cambia;
    - se sembra mobile, aspetto;
    - se resta fermo, aspetto comunque fino a obstacle_static_replan_delay_sec;
    - dopo obstacle_max_wait_sec forzo il replan.
    """

    def __init__(self, node):
        self.node = node

    def _vehicle_conflict_near_path(self, distance_limit: float) -> bool:
        node = self.node

        if not node.has_odom:
            return False

        now = node.get_clock().now().nanoseconds / 1e9

        forward_x = math.cos(node.current_yaw)
        forward_y = math.sin(node.current_yaw)
        right_x = math.sin(node.current_yaw)
        right_y = -math.cos(node.current_yaw)

        for _, other in list(node.other_vehicles.items()):
            stamp = float(other.get("stamp", 0.0))
            if now - stamp > node.vehicle_state_stale_timeout_sec:
                continue

            dx = float(other.get("x", 0.0)) - node.current_x
            dy = float(other.get("y", 0.0)) - node.current_y

            forward_dist = dx * forward_x + dy * forward_y
            side_dist = dx * right_x + dy * right_y
            euclidean_dist = math.sqrt(dx * dx + dy * dy)

            in_front_corridor = (
                -1.0 <= forward_dist <= distance_limit
                and abs(side_dist) <= node.lane_width * 1.35
            )

            close_conflict = euclidean_dist <= distance_limit * 0.85

            if in_front_corridor or close_conflict:
                return True

        return False

    def _lidar_obstacle_ahead(self, distance_limit: float) -> bool:
        node = self.node

        if node.obstacle_min_distance > distance_limit:
            return False

        if not math.isfinite(node.obstacle_min_distance):
            return False

        # Se è un veicolo noto, non lo tratto come ostacolo fisso.
        if self._vehicle_conflict_near_path(distance_limit + 2.0):
            return False

        return True

    def evaluate(self, ctx: NavigationContext) -> Optional[NavigationDecision]:
        node = self.node

        obstacle_close = self._lidar_obstacle_ahead(node.obstacle_stop_distance)
        obstacle_slow = self._lidar_obstacle_ahead(node.obstacle_slow_distance)

        if not obstacle_close:
            if node.state == ExecutorState.OBSTACLE_STOP:
                node.state = ExecutorState.NAVIGATING

            node.reset_lidar_obstacle_watch()
            return None

        now = node.get_clock().now()

        if node.obstacle_stop_start_time is None:
            node.obstacle_stop_start_time = now
            node.obstruction_attempt_count = 0
            node.lidar_obstacle_watch_started_at = now.nanoseconds / 1e9
            node.lidar_obstacle_samples = []

            node.alert(
                "OBSTACLE_WATCH_START",
                f"ostacolo LiDAR a {node.obstacle_min_distance:.2f}m: mi fermo e osservo",
                throttle=False
            )
            node.log_lidar_obstacle_wait_start()

        node.record_lidar_obstacle_sample()

        stopped_for = (now - node.obstacle_stop_start_time).nanoseconds / 1e9
        movement = node.lidar_obstacle_movement_info()

        node.state = ExecutorState.OBSTACLE_STOP

        # 1. Prima finestra: osservo e basta.
        if stopped_for < node.obstacle_observation_window_sec:
            return NavigationDecision(
                DecisionType.STOP,
                reason="lidar_obstacle_observing",
                payload={
                    "stopped_for": stopped_for,
                    "obstacle_distance": node.obstacle_min_distance,
                    "movement": movement,
                },
            )

        # 2. Se sembra mobile, aspetto: magari pedone / ostacolo dinamico.
        if movement["moving"] and stopped_for < node.obstacle_max_wait_sec:
            return NavigationDecision(
                DecisionType.STOP,
                reason="lidar_obstacle_moving_wait",
                payload={
                    "stopped_for": stopped_for,
                    "obstacle_distance": node.obstacle_min_distance,
                    "movement": movement,
                },
            )

        # 3. Se non sembra mobile, non parto comunque subito:
        # aspetto fino alla soglia "statica".
        if stopped_for < node.obstacle_static_replan_delay_sec:
            return NavigationDecision(
                DecisionType.STOP,
                reason="lidar_obstacle_static_wait",
                payload={
                    "stopped_for": stopped_for,
                    "obstacle_distance": node.obstacle_min_distance,
                    "movement": movement,
                },
            )

        # 4. Timeout duro: 180s massimo, poi basta.
        if stopped_for >= node.obstacle_max_wait_sec:
            return NavigationDecision(
                DecisionType.LIDAR_OBSTACLE_RECOVERY,
                reason="lidar_obstacle_max_wait_replan",
                payload={
                    "stopped_for": stopped_for,
                    "obstacle_distance": node.obstacle_min_distance,
                    "movement": movement,
                },
            )

        # 5. Dopo 60s, se è ancora lì e non sembra mobile, lo tratto come fisso.
        return NavigationDecision(
            DecisionType.LIDAR_OBSTACLE_RECOVERY,
            reason="lidar_obstacle_static_confirmed_replan",
            payload={
                "stopped_for": stopped_for,
                "obstacle_distance": node.obstacle_min_distance,
                "movement": movement,
            },
        )

class VehicleSafetyPolicy:
    """
    Coordinamento distribuito tra veicoli.

    Filosofia:
    - chi cede si ferma solo per pochissimo;
    - chi ha priorità non deve "sorpassare", deve solo disallinearsi;
    - appena la collisione non è più probabile, entrambi ripartono.
    """

    def __init__(self, node):
        self.node = node
        self.yield_started_by_vehicle = {}

        # Stop massimo del veicolo che cede.
        # Dopo questo tempo, se l'altro ha avuto modo di orientarsi,
        # il veicolo torna a muoversi piano.
        self.max_yield_stop_sec = 1.10

        # Quando considero la situazione abbastanza libera da ripartire.
        self.release_side_distance = 2.20
        self.release_euclidean_distance = 3.60

        # Micro-manovra del veicolo con priorità.
        # Non è un sorpasso: è solo "mettiti storto/non in traiettoria".
        self.clearance_forward = 2.80
        self.clearance_lateral = 2.45

    def evaluate(self, ctx: NavigationContext) -> Optional[NavigationDecision]:
        node = self.node

        if not node.has_odom:
            return None

        best = None
        best_priority = -1

        now = node.get_clock().now().nanoseconds / 1e9

        forward_x = math.cos(node.current_yaw)
        forward_y = math.sin(node.current_yaw)
        right_x = math.sin(node.current_yaw)
        right_y = -math.cos(node.current_yaw)

        corridor = getattr(
            node,
            "vehicle_corridor_width",
            node.lane_width * 0.75
        )

        for vehicle_id, other in list(node.other_vehicles.items()):
            stamp = float(other.get("stamp", 0.0))
            if now - stamp > node.vehicle_state_stale_timeout_sec:
                continue

            other_x = float(other.get("x", 0.0))
            other_y = float(other.get("y", 0.0))
            other_yaw = float(other.get("yaw", 0.0))

            dx = other_x - node.current_x
            dy = other_y - node.current_y

            euclidean_dist = math.sqrt(dx * dx + dy * dy)
            forward_dist = dx * forward_x + dy * forward_y
            side_dist = dx * right_x + dy * right_y
            abs_side = abs(side_dist)

            heading_diff = abs(node.normalize_angle(other_yaw - node.current_yaw))

            same_direction = heading_diff < math.radians(45)
            opposite_direction = heading_diff > math.radians(135)
            crossing_direction = not same_direction and not opposite_direction

            i_must_yield = node.should_yield_to_vehicle(vehicle_id)

            # ====================================================
            # 1. Stessa direzione: semplice coda
            # ====================================================
            if same_direction:
                queue_risk = (
                    0.0 < forward_dist < node.vehicle_slow_distance
                    and abs_side < corridor
                )

                if queue_risk:
                    if forward_dist < node.vehicle_stop_distance:
                        candidate = NavigationDecision(
                            DecisionType.STOP,
                            reason=f"vehicle_queue_stop_{vehicle_id}",
                        )
                        best, best_priority = self._choose(best, best_priority, candidate, 80)
                    else:
                        t = (forward_dist - node.vehicle_stop_distance) / max(
                            node.vehicle_slow_distance - node.vehicle_stop_distance,
                            0.001
                        )
                        factor = 0.5 * (1.0 - math.cos(node.clamp(t, 0.0, 1.0) * math.pi))
                        factor = node.clamp(factor, 0.20, 0.85)

                        candidate = NavigationDecision(
                            DecisionType.SLOW_FOLLOW,
                            reason=f"vehicle_queue_slow_{vehicle_id}",
                            speed_factor=factor,
                        )
                        best, best_priority = self._choose(best, best_priority, candidate, 40)

                continue

            # ====================================================
            # 2. Conflitto molto vicino
            # ====================================================
            very_close = euclidean_dist < node.assumed_vehicle_width + 1.05

            if very_close:
                if i_must_yield:
                    candidate = self._yield_briefly_or_release(
                        vehicle_id=vehicle_id,
                        now=now,
                        euclidean_dist=euclidean_dist,
                        forward_dist=forward_dist,
                        side_dist=side_dist,
                        reason_prefix="vehicle_close",
                    )
                    best, best_priority = self._choose(best, best_priority, candidate, 100)
                else:
                    candidate = self._clearance_nudge_decision(
                        vehicle_id=vehicle_id,
                        forward_x=forward_x,
                        forward_y=forward_y,
                        right_x=right_x,
                        right_y=right_y,
                        forward_dist=forward_dist,
                        side_dist=side_dist,
                        reason="vehicle_close_clearance_nudge",
                        speed_factor=0.72,
                    )
                    best, best_priority = self._choose(best, best_priority, candidate, 95)

                continue

            # ====================================================
            # 3. Frontale / lati opposti
            # ====================================================
            if opposite_direction:
                headon_risk = (
                    -1.0 < forward_dist < node.vehicle_headon_warn_distance
                    and abs_side < max(corridor, node.vehicle_headon_side_corridor)
                    and euclidean_dist < node.vehicle_headon_warn_distance + 2.0
                )

                if not headon_risk:
                    self.yield_started_by_vehicle.pop(vehicle_id, None)
                    continue

                if i_must_yield:
                    candidate = self._yield_briefly_or_release(
                        vehicle_id=vehicle_id,
                        now=now,
                        euclidean_dist=euclidean_dist,
                        forward_dist=forward_dist,
                        side_dist=side_dist,
                        reason_prefix="vehicle_headon",
                    )
                    best, best_priority = self._choose(best, best_priority, candidate, 90)
                else:
                    candidate = self._clearance_nudge_decision(
                        vehicle_id=vehicle_id,
                        forward_x=forward_x,
                        forward_y=forward_y,
                        right_x=right_x,
                        right_y=right_y,
                        forward_dist=forward_dist,
                        side_dist=side_dist,
                        reason="vehicle_headon_clearance_nudge",
                        speed_factor=0.78,
                    )
                    best, best_priority = self._choose(best, best_priority, candidate, 88)

                continue

            # ====================================================
            # 4. Incrocio / perpendicolari
            # ====================================================
            if crossing_direction:
                crossing_risk = (
                    -1.2 < forward_dist < 8.0
                    and abs_side < 4.2
                    and euclidean_dist < 8.5
                )

                if not crossing_risk:
                    self.yield_started_by_vehicle.pop(vehicle_id, None)
                    continue

                if i_must_yield:
                    candidate = self._yield_briefly_or_release(
                        vehicle_id=vehicle_id,
                        now=now,
                        euclidean_dist=euclidean_dist,
                        forward_dist=forward_dist,
                        side_dist=side_dist,
                        reason_prefix="vehicle_crossing",
                    )
                    best, best_priority = self._choose(best, best_priority, candidate, 75)
                else:
                    # Chi ha priorità non deve strisciare per 10 metri.
                    # Deve solo orientarsi quel tanto che basta.
                    candidate = self._clearance_nudge_decision(
                        vehicle_id=vehicle_id,
                        forward_x=forward_x,
                        forward_y=forward_y,
                        right_x=right_x,
                        right_y=right_y,
                        forward_dist=forward_dist,
                        side_dist=side_dist,
                        reason="vehicle_crossing_clearance_nudge",
                        speed_factor=0.75,
                    )
                    best, best_priority = self._choose(best, best_priority, candidate, 70)

                continue

        return best

    def _yield_briefly_or_release(
        self,
        vehicle_id,
        now,
        euclidean_dist,
        forward_dist,
        side_dist,
        reason_prefix,
    ):
        """
        Chi cede non deve rimanere bloccato per sempre.
        Si ferma un attimo, poi riparte piano se:
        - l'altro si è spostato lateralmente;
        - oppure il blocco sta durando troppo.
        """

        safe_to_release = self._safe_to_release(
            euclidean_dist=euclidean_dist,
            forward_dist=forward_dist,
            side_dist=side_dist,
        )

        if safe_to_release:
            self.yield_started_by_vehicle.pop(vehicle_id, None)
            return NavigationDecision(
                DecisionType.SLOW_FOLLOW,
                reason=f"{reason_prefix}_release_{vehicle_id}",
                speed_factor=0.55,
            )

        started = self.yield_started_by_vehicle.get(vehicle_id)

        if started is None:
            self.yield_started_by_vehicle[vehicle_id] = now
            started = now

        stopped_for = now - started

        if stopped_for > self.max_yield_stop_sec:
            # Non è perfetto, ma evita la cosa peggiore:
            # veicolo fermo + veicolo lento + terzo veicolo che arriva.
            return NavigationDecision(
                DecisionType.SLOW_FOLLOW,
                reason=f"{reason_prefix}_timeout_release_{vehicle_id}",
                speed_factor=0.32,
                payload={
                    "vehicle_id": vehicle_id,
                    "stopped_for": stopped_for,
                    "forward_dist": forward_dist,
                    "side_dist": side_dist,
                    "euclidean_dist": euclidean_dist,
                },
            )

        return NavigationDecision(
            DecisionType.STOP,
            reason=f"{reason_prefix}_brief_yield_{vehicle_id}",
            payload={
                "vehicle_id": vehicle_id,
                "stopped_for": stopped_for,
                "forward_dist": forward_dist,
                "side_dist": side_dist,
                "euclidean_dist": euclidean_dist,
            },
        )

    def _safe_to_release(self, euclidean_dist, forward_dist, side_dist):
        """
        Condizione semplice:
        se non siamo più quasi sovrapposti o non siamo più nella stessa traiettoria,
        non vale la pena tenere uno dei due fermo.
        """

        if euclidean_dist > self.release_euclidean_distance:
            return True

        if abs(side_dist) > self.release_side_distance:
            return True

        # L'altro è già leggermente dietro e lateralmente separato:
        # lasciami ripartire piano.
        if forward_dist < -0.6 and abs(side_dist) > 1.2:
            return True

        return False

    def _clearance_nudge_decision(
        self,
        vehicle_id,
        forward_x,
        forward_y,
        right_x,
        right_y,
        forward_dist,
        side_dist,
        reason,
        speed_factor,
    ):
        """
        Micro-manovra: orientati/scarta a destra e riparti.
        Non deve diventare una lunga manovra di sorpasso.
        """

        node = self.node

        forward = self.clearance_forward
        lateral = self.clearance_lateral

        # Se sono praticamente allineato al centro dell'altro veicolo,
        # aumento leggermente lo scarto laterale.
        if abs(side_dist) < node.assumed_vehicle_width * 0.70:
            lateral *= 1.15

        lateral = node.clamp(lateral, 2.15, 2.95)

        return NavigationDecision(
            DecisionType.TEMPORARY_TARGET,
            reason=f"{reason}_{vehicle_id}",
            speed_factor=speed_factor,
            target_x=node.current_x + forward_x * forward + right_x * lateral,
            target_y=node.current_y + forward_y * forward + right_y * lateral,
            payload={
                "vehicle_id": vehicle_id,
                "forward_dist": forward_dist,
                "side_dist": side_dist,
                "clearance_forward": forward,
                "clearance_lateral": lateral,
            },
        )

    @staticmethod
    def _choose(current, current_priority, candidate, priority):
        if current is None or priority > current_priority:
            return candidate, priority

        if priority == current_priority and candidate.speed_factor < current.speed_factor:
            return candidate, priority

        return current, current_priority

class TrafficLightPolicy:
    def __init__(self, node):
        self.node = node

    def evaluate(self, ctx: NavigationContext) -> Optional[NavigationDecision]:
        if self.node.must_wait_at_traffic_light(ctx.waypoint, ctx.goal, ctx.distance_to_wp):
            return NavigationDecision(DecisionType.STOP, reason="traffic_light_wait")
        return None


class GenericStuckPolicy:
    def __init__(self, node):
        self.node = node

    def evaluate(self, ctx: NavigationContext) -> Optional[NavigationDecision]:
        if self.node.is_stuck_without_reason(ctx.distance_to_wp):
            return NavigationDecision(DecisionType.GENERIC_STUCK_RECOVERY, reason="generic_stuck")
        if ctx.elapsed_on_wp > ctx.waypoint_timeout_sec:
            return NavigationDecision(DecisionType.RESET_WAYPOINT_TIMER, reason="waypoint_timeout_no_hard_reason")
        return None


class SafetySupervisor:
    """Orchestratore delle policy. L'ordine qui è la gerarchia vera."""

    def __init__(self, node):
        self.policies = [
            # Regola dura: se il semaforo dice stop, stop.
            TrafficLightPolicy(node),

            # Prima di fare target temporanei, controllo se sono fisicamente bloccato.
            # Altrimenti un veicolo incastrato continua per sempre a dire TEMPORARY_TARGET.
            DeadlockPolicy(node),

            # Coordinamento normale tra veicoli.
            VehicleSafetyPolicy(node),

            # Ostacoli fissi non-veicolo.
            LidarObstaclePolicy(node),

            GenericStuckPolicy(node),
        ]

    def decide(self, ctx: NavigationContext) -> NavigationDecision:
        for policy in self.policies:
            decision = policy.evaluate(ctx)
            if decision is not None:
                return decision

        return NavigationDecision(
            DecisionType.FOLLOW_LANE,
            reason="normal_follow"
        )


class RecoveryController:
    def __init__(self, node):
        self.node = node

    def run_vehicle_deadlock(self, conflict):
        self.node.perform_vehicle_unjam_maneuver(conflict)

    def run_lidar_obstacle_escape(self, goal, current_wp):
        node = self.node
        node.stop_vehicle()
        node.perform_obstruction_escape_maneuver(current_wp)
        node.mark_obstructed_road_ahead(current_wp)
        node.replan_after_obstacle(goal)
        node.obstacle_stop_start_time = None
        node.reset_lidar_obstacle_watch()

    def run_generic_stuck_recovery(self, current_wp, goal):
        node = self.node
        node.stop_vehicle()
        if node.enable_recovery_maneuver:
            node.run_recovery_maneuver()
        if (
            node.obstacle_min_distance <= node.obstacle_slow_distance
            and not node.has_known_vehicle_in_front(node.obstacle_slow_distance + 2.0)
        ):
            node.mark_obstructed_road_ahead(current_wp)
            node.replan_after_obstacle(goal)


class ConsoleReporter:
    """Console pulita: nessun log di start/arrival dal navigation executor."""

    def __init__(self, node):
        self.node = node

    def mission_start(self, goal):
        return

    def mission_arrival(self, goal):
        return


class NavigationExecutor(Node):

    # ============================================================
    # INIT
    # ============================================================

    def __init__(self):
        super().__init__("navigation_executor")

        self.declare_parameters(
            namespace="",
            parameters=[
                ("vehicle_id", "vehicle"),
                ("map_config_file", "config/city_map.json"),
                ("pose_entity_name", ""),

                ("default_max_speed", 2.4),
                ("linear_k", 1.6),
                ("angular_k", 3.6),
                ("max_angular_speed", 4.0),
                ("lookahead_distance", 4.0),
                ("lane_recovery_threshold", 0.35),

                ("waypoint_tolerance", 0.30),
                ("target_tolerance", 0.45),

                ("lane_width", 2.4),
                ("lane_offset_ratio", 1.0),

                ("intersection_clearance", 2.2),
                ("traffic_light_stop_distance", 7.0),

                ("obstacle_stop_distance", 3.0),
                ("obstacle_slow_distance", 5.0),
                ("obstacle_fov_deg", 60.0),
                ("obstacle_replan_timeout_sec", 15.0),
                ("obstacle_escape_delay_sec", 1.0),
                ("obstruction_reverse_sec", 1.75),
                ("obstruction_turn_sec", 1.10),
                ("obstruction_reverse_speed", -0.45),
                ("obstruction_turn_speed", 0.90),
                ("obstruction_block_next_edge", True),
                ("soft_avoidance_enabled", True),
                ("obstacle_observation_window_sec", 5.0),
                ("obstacle_static_replan_delay_sec", 60.0),
                ("obstacle_max_wait_sec", 180.0),
                ("obstacle_movement_distance_epsilon", 0.35),
                ("obstacle_movement_bearing_epsilon_deg", 6.0),
                ("obstruction_prefer_left", True),

                ("vehicle_headon_warn_distance", 14.0),
                ("vehicle_headon_stop_distance", 3.0),
                ("vehicle_headon_side_corridor", 2.6),
                ("vehicle_headon_extra_right_ratio", 0.38),
                ("vehicle_state_stale_timeout_sec", 1.2),
                ("vehicle_deadlock_detection_sec", 1.25),
                ("vehicle_deadlock_min_displacement", 0.08),
                ("vehicle_unjam_reverse_sec", 1.80),
                ("vehicle_unjam_turn_sec", 1.60),
                ("vehicle_unjam_reverse_speed", -0.65),
                ("vehicle_unjam_turn_speed", 1.45),

                ("diagnostic_log_enabled", False),
                ("diagnostic_log_period_sec", 5.0),
                ("path_log_enabled", False),

                ("traffic_light_wait_timeout_sec", 180.0),
                ("traffic_light_commit_ttl_sec", 4.0),
                ("stuck_timeout_sec", 8.0),
                ("stuck_progress_epsilon", 0.12),
                ("alert_log_period_sec", 2.0),
                ("enable_recovery_maneuver", True),
                ("traffic_light_node_ids", ["n7", "n9", "n14", "n17", "n19", "n22", "n23", "n24"]),
            ]
        )

        self.last_decision_type = ""
        self.last_decision_reason = ""

        self.vehicle_id = self.get_parameter("vehicle_id").value
        self.map_config_file = self.get_parameter("map_config_file").value
        self.pose_entity_name = self.get_parameter("pose_entity_name").value or self.vehicle_id

        self.default_max_speed = float(self.get_parameter("default_max_speed").value)
        self.linear_k = float(self.get_parameter("linear_k").value)
        self.angular_k = float(self.get_parameter("angular_k").value)
        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)

        self.waypoint_tolerance = float(self.get_parameter("waypoint_tolerance").value)
        self.target_tolerance = float(self.get_parameter("target_tolerance").value)

        self.lane_width = float(self.get_parameter("lane_width").value)
        self.lane_offset_ratio = float(self.get_parameter("lane_offset_ratio").value)
        self.lane_recovery_threshold = float(self.get_parameter("lane_recovery_threshold").value)
        self.lookahead_distance = float(self.get_parameter("lookahead_distance").value)

        self.intersection_clearance = max(3.0, float(self.get_parameter("intersection_clearance").value))
        self.traffic_light_stop_distance = max(5.0, float(self.get_parameter("traffic_light_stop_distance").value))

        self.obstacle_stop_distance = float(self.get_parameter("obstacle_stop_distance").value)
        self.obstacle_slow_distance = float(self.get_parameter("obstacle_slow_distance").value)
        self.obstacle_fov_deg = float(self.get_parameter("obstacle_fov_deg").value)
        self.obstacle_replan_timeout_sec = float(self.get_parameter("obstacle_replan_timeout_sec").value)
        self.obstacle_escape_delay_sec = float(self.get_parameter("obstacle_escape_delay_sec").value)
        self.obstruction_reverse_sec = float(self.get_parameter("obstruction_reverse_sec").value)
        self.obstruction_turn_sec = float(self.get_parameter("obstruction_turn_sec").value)
        self.obstruction_reverse_speed = float(self.get_parameter("obstruction_reverse_speed").value)
        self.obstruction_turn_speed = float(self.get_parameter("obstruction_turn_speed").value)
        self.obstruction_block_next_edge = bool(self.get_parameter("obstruction_block_next_edge").value)
        self.soft_avoidance_enabled = bool(self.get_parameter("soft_avoidance_enabled").value)
        self.obstacle_observation_window_sec = float(self.get_parameter("obstacle_observation_window_sec").value)
        self.obstacle_static_replan_delay_sec = float(self.get_parameter("obstacle_static_replan_delay_sec").value)
        self.obstacle_max_wait_sec = float(self.get_parameter("obstacle_max_wait_sec").value)
        self.obstacle_movement_distance_epsilon = float(self.get_parameter("obstacle_movement_distance_epsilon").value)
        self.obstacle_movement_bearing_epsilon = math.radians(float(self.get_parameter("obstacle_movement_bearing_epsilon_deg").value))
        self.obstruction_prefer_left = bool(self.get_parameter("obstruction_prefer_left").value)

        self.vehicle_headon_warn_distance = float(self.get_parameter("vehicle_headon_warn_distance").value)
        self.vehicle_headon_stop_distance = float(self.get_parameter("vehicle_headon_stop_distance").value)
        self.vehicle_headon_side_corridor = float(self.get_parameter("vehicle_headon_side_corridor").value)
        self.vehicle_headon_extra_right_ratio = float(self.get_parameter("vehicle_headon_extra_right_ratio").value)
        self.vehicle_state_stale_timeout_sec = float(self.get_parameter("vehicle_state_stale_timeout_sec").value)
        self.vehicle_deadlock_detection_sec = float(self.get_parameter("vehicle_deadlock_detection_sec").value)
        self.vehicle_deadlock_min_displacement = float(self.get_parameter("vehicle_deadlock_min_displacement").value)
        self.vehicle_unjam_reverse_sec = float(self.get_parameter("vehicle_unjam_reverse_sec").value)
        self.vehicle_unjam_turn_sec = float(self.get_parameter("vehicle_unjam_turn_sec").value)
        self.vehicle_unjam_reverse_speed = float(self.get_parameter("vehicle_unjam_reverse_speed").value)
        self.vehicle_unjam_turn_speed = float(self.get_parameter("vehicle_unjam_turn_speed").value)

        self.diagnostic_log_enabled = bool(self.get_parameter("diagnostic_log_enabled").value)
        self.diagnostic_log_period_sec = float(self.get_parameter("diagnostic_log_period_sec").value)
        self.path_log_enabled = bool(self.get_parameter("path_log_enabled").value)

        self.traffic_light_wait_timeout_sec = float(self.get_parameter("traffic_light_wait_timeout_sec").value)
        self.traffic_light_commit_ttl_sec = float(self.get_parameter("traffic_light_commit_ttl_sec").value)
        self.stuck_timeout_sec = float(self.get_parameter("stuck_timeout_sec").value)
        self.stuck_progress_epsilon = float(self.get_parameter("stuck_progress_epsilon").value)
        self.alert_log_period_sec = float(self.get_parameter("alert_log_period_sec").value)
        self.enable_recovery_maneuver = bool(self.get_parameter("enable_recovery_maneuver").value)
        self.traffic_light_node_ids = set(str(x) for x in self.get_parameter("traffic_light_node_ids").value)


        # Stato veicolo
        self.state = ExecutorState.IDLE
        self.has_odom = False
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        # Sensori
        self.obstacle_min_distance = float("inf")
        self.obstacle_bearing = 0.0
        self.obstacle_left_min_distance = float("inf")
        self.obstacle_right_min_distance = float("inf")
        self.last_scan_stamp = None
        self.obstacle_stop_start_time = None
        self.lidar_obstacle_samples = []
        self.lidar_obstacle_watch_started_at = None
        self.obstruction_attempt_count = 0
        self.last_obstruction_replan_time = 0.0

        # FIX: contatore per evitare blocchi infiniti di edge con replan a catena
        self._replan_failure_count = 0
        self._max_replan_failures = 4

        # Veicoli
        self.other_vehicles = {}
        self.vehicle_state_publish_period_sec = 0.2
        # Distanze più prudenti: il veicolo deve iniziare a ragionare prima,
        # non quando ormai il frontale è già inevitabile.
        self.vehicle_stop_distance = 4.0
        self.vehicle_slow_distance = 9.0

        # Assunzione semplice: tutti i veicoli hanno circa la stessa larghezza.
        # Non serve più provare a dedurre troppo: usiamo un corridoio stabile.
        self.assumed_vehicle_width = 1.35
        self.vehicle_lateral_clearance = 0.55

        # Corridoio entro cui considero un veicolo davvero sulla mia traiettoria.
        self.vehicle_corridor_width = self.assumed_vehicle_width + self.vehicle_lateral_clearance

        # Manovra molto più audace:
        # prima era circa 0.9m; ora punta quasi a una corsia laterale piena.
        self.vehicle_pass_lateral_offset = max(
            2.20,
            self.assumed_vehicle_width + self.vehicle_lateral_clearance
        )

        self.vehicle_pass_lookahead = 6.0
        self.vehicle_priority_pass_speed_factor = 0.62
        self.vehicle_yield_release_distance = 7.0
        self.vehicle_state_pub = self.create_publisher(String, "/vehicle_states", 100)
        self.vehicle_state_sub = self.create_subscription(
            String, "/vehicle_states", self.on_vehicle_state, 100
        )
        self.vehicle_state_timer = self.create_timer(
            self.vehicle_state_publish_period_sec, self.publish_vehicle_state
        )

        # Semafori
        self.traffic_light_statuses = {}
        self.last_priority_request_time = {}
        self.traffic_light_wait_started_at = {}
        self.committed_traffic_lights = {}
        self.traffic_light_commit_ttl_sec = max(0.5, self.traffic_light_commit_ttl_sec)

        # Mappa
        self.nodes = {}
        self.edges = []
        self.edge_by_id = {}
        self.adj = {}
        self.default_map_speed_limit = 1.4
        self.blocked_edges = {}
        self.blocked_edge_ttl_sec = 45.0

        # Navigazione
        self.current_path = []
        self.node_path = []
        self.current_waypoint_index = 0
        self.current_mission_id = ""
        self.last_decision_type = ""
        self.last_decision_reason = ""
        self.last_decision_payload = {}
        self.last_cmd_linear_x = 0.0
        self.last_cmd_angular_z = 0.0

        self.last_diag_time = self.get_clock().now()
        self.last_alert_time_by_key = {}
        self.last_console_event_time_by_key = {}
        self.last_progress_distance = None
        self.last_progress_time = self.get_clock().now()
        self.last_recovery_time = 0.0

        # Watchdog fisico: non guarda se il waypoint si avvicina, ma se il
        # modello si sta proprio spostando nel mondo. Serve per i casi in cui
        # due veicoli si incastrano, continuano a comandare velocità, ma la posa
        # resta quasi ferma.
        self.last_motion_x = None
        self.last_motion_y = None
        self.last_motion_time = self.get_clock().now()
        self.last_vehicle_deadlock_recovery_time = 0.0

        self.load_map()
        self.validate_runtime_parameters()

        # Componenti logici interni: non sono nodi ROS separati.
        self.safety_supervisor = SafetySupervisor(self)
        self.recovery_controller = RecoveryController(self)
        self.console_reporter = ConsoleReporter(self)

        # ROS
        self.callback_group = ReentrantCallbackGroup()

        self.cmd_vel_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.alert_pub = self.create_publisher(String, "navigation_alerts", 20)

        self.pose_sub = self.create_subscription(
            PoseStamped,
            f"/gazebo/model_pose/{self.vehicle_id}",
            self.on_world_pose,
            10,
            callback_group=self.callback_group
        )

        self.scan_sub = self.create_subscription(
            LaserScan, "scan", self.on_scan, 10,
            callback_group=self.callback_group
        )

        self.traffic_light_sub = self.create_subscription(
            String, "/traffic_light/status", self.on_traffic_light_status, 10,
            callback_group=self.callback_group
        )

        self.priority_pub = self.create_publisher(String, "/traffic_light/priority_request", 10)

        self.action_server = ActionServer(
            self,
            NavigateToPose,
            "navigation_executor/navigate_to_pose",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group
        )

        self.debug_state_srv = self.create_service(
            Trigger,
            "navigation_executor/debug_state",
            self.on_debug_state_request,
            callback_group=self.callback_group
        )

    # ============================================================
    # MAPPA
    # ============================================================

    def load_map(self):
        path = self.map_config_file
        if not os.path.isabs(path):
            path = os.path.join(os.getcwd(), path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File mappa non trovato: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.lane_width = float(data.get("lane_width", self.lane_width))
        self.default_map_speed_limit = float(data.get("default_speed_limit", 1.4))

        for node in data["nodes"]:
            self.nodes[node["id"]] = {
                "id": node["id"],
                "x": float(node["x"]),
                "y": float(node["y"]),
            }

        for edge in data["edges"]:
            edge_id = edge["id"]
            from_id = edge["from"]
            to_id = edge["to"]

            if from_id not in self.nodes or to_id not in self.nodes:
                raise RuntimeError(f"Edge {edge_id} usa nodi non esistenti: {from_id}->{to_id}")

            a = self.nodes[from_id]
            b = self.nodes[to_id]
            length = self.distance_xy(a["x"], a["y"], b["x"], b["y"])

            e = {
                "id": edge_id,
                "from": from_id,
                "to": to_id,
                "speed_limit": float(edge.get("speed_limit", self.default_map_speed_limit)),
                "length": length,
            }

            self.edges.append(e)
            self.edge_by_id[edge_id] = e
            self.adj.setdefault(from_id, [])
            self.adj.setdefault(to_id, [])
            self.adj[from_id].append((to_id, edge_id, length))
            self.adj[to_id].append((from_id, edge_id, length))

    # ============================================================
    # SENSORI
    # ============================================================

    def on_world_pose(self, msg):
        p = msg.pose.position
        q = msg.pose.orientation

        self.current_x = float(p.x)
        self.current_y = float(p.y)

        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        raw_yaw = math.atan2(siny_cosp, cosy_cosp)
        # Compensazione modello: il frontale reale del bus è ruotato di 180°.
        self.current_yaw = self.normalize_angle(raw_yaw + math.pi)
        self.has_odom = True

    def on_scan(self, msg):
        self.last_scan_stamp = msg.header.stamp

        fov_rad = math.radians(self.obstacle_fov_deg / 2.0)
        min_dist = float("inf")
        min_bearing = 0.0
        left_min = float("inf")
        right_min = float("inf")

        # Nel modello il verso di movimento "avanti" corrisponde ad angolo ±pi nel LaserScan.
        scan_forward_angle = math.pi
        angle = msg.angle_min

        for r in msg.ranges:
            if msg.range_min <= r <= msg.range_max and math.isfinite(r):
                normalized = self.normalize_angle(angle)
                delta = self.normalize_angle(normalized - scan_forward_angle)

                if abs(delta) <= fov_rad:
                    if r < min_dist:
                        min_dist = r
                        min_bearing = delta
                    if delta >= 0.0:
                        left_min = min(left_min, r)
                    else:
                        right_min = min(right_min, r)

            angle += msg.angle_increment

        self.obstacle_min_distance = min_dist
        self.obstacle_bearing = min_bearing
        self.obstacle_left_min_distance = left_min
        self.obstacle_right_min_distance = right_min

    def publish_vehicle_state(self):
        if not self.has_odom:
            return

        payload = {
            "vehicle_id": self.vehicle_id,
            "x": self.current_x,
            "y": self.current_y,
            "yaw": self.current_yaw,
            "stamp": self.get_clock().now().nanoseconds / 1e9,
        }

        msg = String()
        msg.data = json.dumps(payload)
        self.vehicle_state_pub.publish(msg)

    def on_vehicle_state(self, msg):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        vehicle_id = data.get("vehicle_id")
        if not vehicle_id or vehicle_id == self.vehicle_id:
            return

        self.other_vehicles[vehicle_id] = data

    def on_traffic_light_status(self, msg):
        try:
            status = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        node_id = status.get("node_id")
        if node_id:
            self.traffic_light_statuses[node_id] = status

    # ============================================================
    # ACTION SERVER
    # ============================================================

    def goal_callback(self, goal_request):
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        goal = goal_handle.request
        self.current_mission_id = goal.mission_id

        if not self.has_odom:
            self.stop_vehicle()
            goal_handle.abort()
            result = NavigateToPose.Result()
            result.success = False
            result.message = "Posa reale Gazebo non ancora disponibile"
            return result

        try:
            self.state = ExecutorState.NAVIGATING
            self._replan_failure_count = 0
            self.plan_path_to_goal(goal)
            self.reset_motion_watchdog()
        except Exception as ex:
            self.stop_vehicle()
            self.state = ExecutorState.IDLE
            goal_handle.abort()
            result = NavigateToPose.Result()
            result.success = False
            result.message = f"Errore calcolo path: {ex}"
            return result

        rate = self.create_rate(20)
        waypoint_start_time = self.get_clock().now()
        self.last_progress_time = waypoint_start_time
        self.last_progress_distance = None
        waypoint_timeout_sec = self.compute_waypoint_timeout(self.current_path[0])

        while rclpy.ok():

            if goal_handle.is_cancel_requested:
                self.stop_vehicle()
                self.state = ExecutorState.IDLE
                goal_handle.canceled()
                result = NavigateToPose.Result()
                result.success = False
                result.message = "Navigazione cancellata"
                return result

            if self.current_waypoint_index >= len(self.current_path):
                final_wp = self.current_path[-1]
                final_distance = self.distance_xy(
                    self.current_x, self.current_y,
                    final_wp["x"], final_wp["y"]
                )

                if final_distance > self.target_tolerance:
                    self.stop_vehicle()
                    self.state = ExecutorState.IDLE
                    goal_handle.abort()
                    result = NavigateToPose.Result()
                    result.success = False
                    result.message = (
                        f"Path terminato ma target non raggiunto: distanza={final_distance:.2f} m"
                    )
                    return result

                self.stop_vehicle()
                self.state = ExecutorState.IDLE
                self.log_mission_arrival(goal)
                goal_handle.succeed()
                result = NavigateToPose.Result()
                result.success = True
                result.message = "Target raggiunto"
                return result

            current_wp = self.current_path[self.current_waypoint_index]
            distance_to_wp = self.distance_xy(
                self.current_x, self.current_y,
                current_wp["x"], current_wp["y"]
            )

            is_last = self.current_waypoint_index == len(self.current_path) - 1
            tolerance = self.compute_waypoint_reach_tolerance(current_wp, is_last)

            if distance_to_wp <= tolerance:
                self.current_waypoint_index += 1
                self.traffic_light_wait_started_at.clear()
                waypoint_start_time = self.get_clock().now()
                self.last_progress_time = waypoint_start_time
                self.last_progress_distance = None
                self.reset_motion_watchdog()

                if self.current_waypoint_index < len(self.current_path):
                    waypoint_timeout_sec = self.compute_waypoint_timeout(
                        self.current_path[self.current_waypoint_index]
                    )

                rate.sleep()
                continue

            self.update_progress_watchdog(distance_to_wp)
            elapsed_on_wp = (self.get_clock().now() - waypoint_start_time).nanoseconds / 1e9
            ctx = NavigationContext(
                goal=goal,
                waypoint=current_wp,
                distance_to_wp=distance_to_wp,
                elapsed_on_wp=elapsed_on_wp,
                waypoint_timeout_sec=waypoint_timeout_sec,
            )

            decision = self.safety_supervisor.decide(ctx)
            reset_timer = self.apply_navigation_decision(decision, goal, current_wp)

            if reset_timer:
                waypoint_start_time = self.get_clock().now()
                self.last_progress_time = waypoint_start_time
                self.last_progress_distance = None

                # Non resettare il watchdog fisico quando stiamo solo facendo STOP:
                # altrimenti un frontale già bloccato non maturerebbe mai come deadlock.
                if decision.type != DecisionType.STOP:
                    self.reset_motion_watchdog()

                if self.current_waypoint_index < len(self.current_path):
                    waypoint_timeout_sec = self.compute_waypoint_timeout(
                        self.current_path[self.current_waypoint_index]
                    )

            goal_handle.publish_feedback(self.make_feedback(goal))
            self.maybe_log_diagnostics(current_wp, distance_to_wp)
            rate.sleep()

        self.stop_vehicle()
        self.state = ExecutorState.IDLE
        goal_handle.abort()
        result = NavigateToPose.Result()
        result.success = False
        result.message = "Navigazione interrotta"
        return result

    def plan_path_to_goal(self, goal):
        self.log_mission_start(goal)

        self.current_path, self.node_path = self.build_navigation_path(
            self.current_x, self.current_y,
            goal.target_x, goal.target_y
        )

        self.current_waypoint_index = 0

        if not self.current_path:
            raise RuntimeError("path vuoto")

        self.log_path(self.current_path)

    def make_feedback(self, goal):
        feedback = NavigateToPose.Feedback()
        feedback.current_x = float(self.current_x)
        feedback.current_y = float(self.current_y)
        feedback.distance_remaining = float(self.compute_remaining_distance())
        feedback.status = (
            f"mission={goal.mission_id} "
            f"wp={self.current_waypoint_index + 1}/{len(self.current_path)} "
            f"state={self.state.value} "
            f"decision={self.last_decision_type} "
            f"reason={self.last_decision_reason}"
        )
        return feedback
    
    def compute_waypoint_reach_tolerance(self, waypoint, is_last):
        """
        I waypoint di incrocio sono guide geometriche, non bersagli millimetrici.
        Se provo a centrarli con tolleranza 0.25/0.45, i veicoli si piantano
        agli incroci e iniziano a dondolare.
        """
        base = self.target_tolerance if is_last else self.waypoint_tolerance
        kind = waypoint.get("kind")
        node_id = waypoint.get("node_id")

        if kind == "start_lane_projection":
            return max(base, 0.80)

        if kind == "approach_intersection":
            # Se non c'è semaforo reale, questo waypoint NON è una linea di stop:
            # è solo un punto guida prima dell'incrocio. Lo posso considerare raggiunto
            # appena sono abbastanza vicino.
            if node_id not in self.traffic_light_node_ids:
                return max(base, 3.20)

            # Se il semaforo è stato già preso/committato, posso attraversare senza
            # centrare perfettamente il punto di approccio.
            if node_id in self.committed_traffic_lights:
                return max(base, 2.20)

            # Semaforo vero non ancora autorizzato: qui tengo una tolleranza stretta.
            return max(base, 0.55)

        if kind == "intersection_corner":
            return max(base, 1.80)

        if kind == "exit_intersection":
            return max(base, 1.80)

        if kind == "target_lane_projection":
            return max(base, self.target_tolerance)

        return base

    def compute_waypoint_timeout(self, waypoint):
        speed_limit = max(0.5, float(waypoint.get("speed_limit", self.default_max_speed)))

        if self.current_waypoint_index == 0:
            previous_x = self.current_x
            previous_y = self.current_y
        else:
            previous_wp = self.current_path[self.current_waypoint_index - 1]
            previous_x = previous_wp["x"]
            previous_y = previous_wp["y"]

        segment_length = self.distance_xy(
            previous_x, previous_y,
            waypoint["x"], waypoint["y"]
        )

        expected_time = segment_length / speed_limit
        return max(8.0, expected_time * 4.0)

    # ============================================================
    # OSTACOLI – LIDAR
    # ============================================================

    def get_obstacle_speed_factor(self):
        d = self.obstacle_min_distance
        lidar_factor = 1.0

        if d <= self.obstacle_stop_distance:
            lidar_factor = 0.0
        elif d <= self.obstacle_slow_distance:
            lidar_factor = (d - self.obstacle_stop_distance) / (
                self.obstacle_slow_distance - self.obstacle_stop_distance
            )
            lidar_factor = self.clamp(lidar_factor, 0.0, 1.0)

        # Se il LiDAR sta vedendo un veicolo noto, NON trattarlo come
        # ostacolo fisso: lo gestisce la logica vehicle-aware sotto.
        if d <= self.obstacle_slow_distance and self.has_known_vehicle_in_front(
            self.obstacle_slow_distance + 2.0
        ):
            lidar_factor = 1.0

        avoidance = self.get_vehicle_avoidance_target()

        if avoidance is not None:
            # STOP veicolo: non deve finire in handle_obstacle_stop(), altrimenti
            # bloccherebbe edge e farebbe replan come se fosse un muro.
            if avoidance.get("type") == "STOP":
                vehicle_factor = 1.0
            else:
                vehicle_factor = float(avoidance.get("speed_factor", 0.65))
        else:
            vehicle_factor = self.get_vehicle_proximity_factor()

        return min(lidar_factor, vehicle_factor)

    def handle_obstacle_stop(self, goal, current_wp):
        """
        Ostacolo LiDAR non associato a un veicolo noto.

        Qui non provo più a "passare comunque": se resta davanti anche solo
        per poco, tratto la strada come ostruita, faccio retromarcia, blocco
        l'edge e ricalcolo. I veicoli mobili vengono filtrati prima con
        has_known_vehicle_in_front(), quindi non dovrebbero finire qui.
        """
        now = self.get_clock().now()
        now_sec = now.nanoseconds / 1e9

        if self.has_known_vehicle_in_front(self.obstacle_slow_distance + 2.0):
            self.stop_vehicle()
            return False

        if self.obstacle_stop_start_time is None:
            self.obstacle_stop_start_time = now
            self.obstruction_attempt_count = 0

        stopped_for = (now - self.obstacle_stop_start_time).nanoseconds / 1e9

        if self.state != ExecutorState.OBSTACLE_STOP:
            self.state = ExecutorState.OBSTACLE_STOP
            self.alert(
                "OBSTACLE_STOP",
                f"ostacolo fisso a {self.obstacle_min_distance:.2f} m: stop + preparo fuga",
                throttle=True
            )

        self.stop_vehicle()

        # Non aspetto 15 secondi davanti a un muro: basta un breve debounce
        # per evitare falsi positivi di uno scan sporco.
        escape_delay = max(0.2, min(self.obstacle_escape_delay_sec, self.obstacle_replan_timeout_sec))
        if stopped_for < escape_delay:
            return False

        if now_sec - self.last_obstruction_replan_time < 2.0:
            return False

        if self._replan_failure_count >= self._max_replan_failures:
            self.alert(
                "REPLAN_LIMIT",
                f"raggiunti {self._replan_failure_count} replan falliti: rilascio blocchi edge",
                throttle=False
            )
            self.blocked_edges.clear()
            self._replan_failure_count = 0

        self.last_obstruction_replan_time = now_sec
        self.obstruction_attempt_count += 1

        self.alert(
            "ROAD_OBSTRUCTED",
            f"ostacolo persistente da {stopped_for:.1f}s: retro + blocco edge + replan",
            throttle=False
        )

        self.perform_obstruction_escape_maneuver(current_wp)
        self.mark_obstructed_road_ahead(current_wp)
        self.replan_after_obstacle(goal)

        self.obstacle_stop_start_time = None
        return True

    def perform_obstruction_escape_maneuver(self, current_wp):
        """
        Retromarcia vera + arco di uscita.

        La vecchia fase finale avanzava di nuovo verso l'ostacolo; qui invece
        il veicolo arretra e sterza verso il lato più libero, poi si ferma: il
        replan successivo lo porta via da un altro edge.
        """
        if not self.enable_recovery_maneuver:
            return

        left_clear = self.normalize_clearance(self.obstacle_left_min_distance)
        right_clear = self.normalize_clearance(self.obstacle_right_min_distance)

        # Per ostacoli fissi preferisco liberarmi verso sinistra,
        # salvo sinistra chiaramente più chiusa.
        if self.obstruction_prefer_left:
            turn_sign = 1.0 if left_clear >= right_clear * 0.70 else -1.0
        else:
            turn_sign = -1.0 if right_clear >= left_clear * 0.70 else 1.0

        asymmetry = abs(right_clear - left_clear) / max(right_clear + left_clear, 0.001)
        turn_intensity = self.obstruction_turn_speed * (0.80 + 0.35 * asymmetry)
        turn_intensity = self.clamp(turn_intensity, 0.35, self.max_angular_speed)

        self.alert(
            "OBSTRUCTION_ESCAPE",
            f"retromarcia evasiva verso {'sx' if turn_sign > 0 else 'dx'} "
            f"(sx={left_clear:.1f}m dx={right_clear:.1f}m)",
            throttle=False
        )

        cmd = Twist()

        # Fase 1: arretra quasi dritto per creare spazio dal blocco.
        end_reverse = self.get_clock().now().nanoseconds / 1e9 + self.obstruction_reverse_sec
        while rclpy.ok() and self.get_clock().now().nanoseconds / 1e9 < end_reverse:
            cmd.linear.x = self.obstruction_reverse_speed
            cmd.angular.z = turn_sign * turn_intensity * 0.45
            self.cmd_vel_pub.publish(cmd)
            time.sleep(0.02)

        # Fase 2: continua ad arretrare, ma gira più deciso per puntare via.
        end_turn = self.get_clock().now().nanoseconds / 1e9 + self.obstruction_turn_sec
        while rclpy.ok() and self.get_clock().now().nanoseconds / 1e9 < end_turn:
            cmd.linear.x = self.obstruction_reverse_speed * 0.70
            cmd.angular.z = turn_sign * turn_intensity
            self.cmd_vel_pub.publish(cmd)
            time.sleep(0.02)

        self.stop_vehicle()

    def mark_obstructed_road_ahead(self, current_wp):
        """
        Blocca solo l'edge realmente davanti al veicolo.
        FIX: evita di bloccare edge già bloccati o edge su cui il veicolo
        non è realmente posizionato (proiezione ambigua agli incroci).
        """
        edge_ids = []

        wp_edge = current_wp.get("edge_id") if current_wp else None
        if wp_edge:
            edge_ids.append(wp_edge)

        if self.obstruction_block_next_edge and current_wp:
            from_node = current_wp.get("node_id")
            to_node = current_wp.get("to_node_id")
            if from_node and to_node and from_node != to_node:
                try:
                    next_edge = self.get_edge_between(from_node, to_node)
                    edge_ids.append(next_edge["id"])
                except Exception:
                    pass

        if not edge_ids:
            try:
                lane_projection = self.find_nearest_lane_projection(
                    self.current_x, self.current_y, allow_blocked=True
                )
                if lane_projection and lane_projection.get("distance", 999.0) < self.lane_width * 1.2:
                    edge_ids.append(lane_projection["edge"]["id"])
            except Exception as ex:
                self.log_navigation_event(f"impossibile localizzare edge ostruito: {ex}")

        for edge_id in sorted(set(edge_ids)):
            self.block_edge_temporarily(edge_id)

        self.log_navigation_event(
            "strade bloccate per ostruzione: " + ", ".join(sorted(self.blocked_edges))
        )

    def block_edge_temporarily(self, edge_id):
        if not edge_id:
            return
        expire_at = self.get_clock().now().nanoseconds / 1e9 + self.blocked_edge_ttl_sec
        self.blocked_edges[edge_id] = expire_at

    def cleanup_expired_blocked_edges(self):
        now = self.get_clock().now().nanoseconds / 1e9
        expired = [eid for eid, exp in self.blocked_edges.items() if exp <= now]
        for eid in expired:
            del self.blocked_edges[eid]

    def is_edge_blocked(self, edge_id):
        self.cleanup_expired_blocked_edges()
        return edge_id in self.blocked_edges

    def mark_current_road_blocked(self, current_wp):
        edge_id = current_wp.get("edge_id")
        if edge_id:
            self.block_edge_temporarily(edge_id)

        try:
            lane_projection = self.find_nearest_lane_projection(
                self.current_x, self.current_y,
                preferred_edge_id=edge_id, allow_blocked=True
            )
            if lane_projection and lane_projection["edge"]:
                self.block_edge_temporarily(lane_projection["edge"]["id"])
        except Exception as ex:
            self.log_navigation_event(f"impossibile localizzare corsia durante blocco: {ex}")

        self.log_navigation_event(
            "strade bloccate: " + ", ".join(sorted(self.blocked_edges))
        )

    def replan_after_obstacle(self, goal):
        """
        FIX: tiene traccia dei fallimenti consecutivi.
        Se il replan fallisce con tutti gli edge bloccati, libera progressivamente
        i blocchi più vecchi prima di arrendersi.
        """
        self.state = ExecutorState.RECALCULATING

        try:
            self.current_path, self.node_path = self.build_navigation_path(
                self.current_x, self.current_y,
                goal.target_x, goal.target_y
            )
            self.current_waypoint_index = 0
            self._replan_failure_count = 0
            self.state = ExecutorState.NAVIGATING
            self.log_path(self.current_path)

        except Exception as ex:
            self._replan_failure_count += 1
            self.stop_vehicle()
            self.state = ExecutorState.OBSTACLE_STOP
            self.log_navigation_event(
                f"ricalcolo fallito ({self._replan_failure_count}/{self._max_replan_failures}): {ex}"
            )
            self.alert(
                "REPLAN_FAILED",
                f"ricalcolo fallito ({self._replan_failure_count}): {ex}",
                throttle=True
            )

    def reset_lidar_obstacle_watch(self):
        self.obstacle_stop_start_time = None
        self.lidar_obstacle_watch_started_at = None
        self.lidar_obstacle_samples = []


    def record_lidar_obstacle_sample(self):
        if not math.isfinite(self.obstacle_min_distance):
            return

        now_sec = self.get_clock().now().nanoseconds / 1e9

        if self.lidar_obstacle_watch_started_at is None:
            self.lidar_obstacle_watch_started_at = now_sec

        # Non salvare 20 campioni identici al secondo.
        if self.lidar_obstacle_samples:
            last_t = self.lidar_obstacle_samples[-1]["t"]
            if now_sec - last_t < 0.20:
                return

        self.lidar_obstacle_samples.append({
            "t": now_sec,
            "distance": float(self.obstacle_min_distance),
            "bearing": float(self.obstacle_bearing),
            "left": float(self.normalize_clearance(self.obstacle_left_min_distance)),
            "right": float(self.normalize_clearance(self.obstacle_right_min_distance)),
        })

        # Tengo solo una finestra recente, più un po' di margine.
        keep_window = max(self.obstacle_observation_window_sec + 2.0, 8.0)
        self.lidar_obstacle_samples = [
            s for s in self.lidar_obstacle_samples
            if now_sec - s["t"] <= keep_window
        ]


    def lidar_obstacle_watch_age(self):
        if self.lidar_obstacle_watch_started_at is None:
            return 0.0

        now_sec = self.get_clock().now().nanoseconds / 1e9
        return now_sec - self.lidar_obstacle_watch_started_at


    def lidar_obstacle_movement_info(self):
        """
        Ritorna info sul movimento apparente dell'ostacolo.
        Funziona perché il veicolo è fermo: se distanza/bearing cambiano,
        probabilmente l'ostacolo davanti si sta muovendo.
        """
        now_sec = self.get_clock().now().nanoseconds / 1e9

        recent = [
            s for s in self.lidar_obstacle_samples
            if now_sec - s["t"] <= self.obstacle_observation_window_sec
        ]

        if len(recent) < 3:
            return {
                "enough_samples": False,
                "moving": False,
                "distance_delta": 0.0,
                "bearing_delta": 0.0,
                "sample_count": len(recent),
            }

        distances = [s["distance"] for s in recent]
        bearings = [s["bearing"] for s in recent]

        distance_delta = max(distances) - min(distances)
        bearing_delta = max(bearings) - min(bearings)

        moving = (
            distance_delta >= self.obstacle_movement_distance_epsilon
            or bearing_delta >= self.obstacle_movement_bearing_epsilon
        )

        return {
            "enough_samples": True,
            "moving": moving,
            "distance_delta": distance_delta,
            "bearing_delta": bearing_delta,
            "sample_count": len(recent),
        }

    # ============================================================
    # OSTACOLI – VEICOLI
    # ============================================================

    def normalize_clearance(self, value):
        if not math.isfinite(value):
            return self.obstacle_slow_distance * 3.0
        return float(value)

    def stable_vehicle_order_key(self, vehicle_id):
        # Non usare hash(): in Python è randomizzato per processo, quindi due
        # nodi ROS possono decidere priorità opposte. Questa chiave è stabile.
        text = str(vehicle_id)
        return sum((i + 1) * ord(ch) for i, ch in enumerate(text))

    def should_yield_to_vehicle(self, vehicle_id):
        my_key = self.stable_vehicle_order_key(self.vehicle_id)
        other_key = self.stable_vehicle_order_key(vehicle_id)
        if my_key == other_key:
            return str(self.vehicle_id) > str(vehicle_id)
        return my_key > other_key

    def has_known_vehicle_in_front(self, max_distance):
        if not self.has_odom:
            return False

        now = self.get_clock().now().nanoseconds / 1e9
        forward_x = math.cos(self.current_yaw)
        forward_y = math.sin(self.current_yaw)
        right_x = math.sin(self.current_yaw)
        right_y = -math.cos(self.current_yaw)

        for _, other in list(self.other_vehicles.items()):
            stamp = float(other.get("stamp", 0.0))
            if now - stamp > self.vehicle_state_stale_timeout_sec:
                continue

            dx = float(other.get("x", 0.0)) - self.current_x
            dy = float(other.get("y", 0.0)) - self.current_y

            forward_dist = dx * forward_x + dy * forward_y
            side_dist = dx * right_x + dy * right_y

            if -0.6 <= forward_dist <= max_distance and abs(side_dist) <= self.lane_width * 0.95:
                return True

        return False

    def get_current_lane_anchor_for_avoidance(self):
        waypoint = None
        if self.current_path and self.current_waypoint_index < len(self.current_path):
            waypoint = self.current_path[self.current_waypoint_index]

        preferred_edge_id = waypoint.get("edge_id") if waypoint else None
        destination_node_id = waypoint.get("node_id") if waypoint else None

        try:
            return self.find_nearest_lane_projection(
                self.current_x, self.current_y,
                preferred_edge_id=preferred_edge_id,
                destination_node_id=destination_node_id,
                allow_blocked=True
            )
        except Exception:
            return {"x": self.current_x, "y": self.current_y, "distance": 0.0}

    def get_vehicle_proximity_factor(self):
        """
        Fattore di velocità vehicle-aware.

        Caso chiave: frontale. Non basta "uno rallenta, l'altro schiva":
        entrambi iniziano presto a rallentare e a spostarsi verso la propria
        destra. Qui restituisco solo la velocità; il target laterale viene
        deciso in get_vehicle_avoidance_target().
        """
        if not self.has_odom:
            return 1.0

        now = self.get_clock().now().nanoseconds / 1e9

        forward_x = math.cos(self.current_yaw)
        forward_y = math.sin(self.current_yaw)
        right_x = math.sin(self.current_yaw)
        right_y = -math.cos(self.current_yaw)

        min_factor = 1.0

        for vehicle_id, other in list(self.other_vehicles.items()):
            stamp = float(other.get("stamp", 0.0))
            if now - stamp > self.vehicle_state_stale_timeout_sec:
                continue

            other_x = float(other.get("x", 0.0))
            other_y = float(other.get("y", 0.0))
            other_yaw = float(other.get("yaw", 0.0))

            dx = other_x - self.current_x
            dy = other_y - self.current_y

            euclidean_dist = math.sqrt(dx * dx + dy * dy)
            forward_dist = dx * forward_x + dy * forward_y
            side_dist = dx * right_x + dy * right_y
            abs_side_dist = abs(side_dist)

            heading_diff = abs(self.normalize_angle(other_yaw - self.current_yaw))
            same_direction = heading_diff < math.radians(45)
            opposite_direction = heading_diff > math.radians(145)
            crossing_direction = not same_direction and not opposite_direction

            i_must_yield = self.should_yield_to_vehicle(vehicle_id)

            # Emergenza vera: troppo vicini e non già separati lateralmente.
            if euclidean_dist < 2.4 and abs_side_dist < 1.35:
                if opposite_direction:
                    return 0.0
                return 0.0 if i_must_yield else 0.75

            # Accodamento: parte prima e resta più fluido.
            if same_direction:
                if forward_dist > 0.0 and abs_side_dist < self.vehicle_corridor_width:
                    stop_d = self.vehicle_stop_distance
                    slow_d = self.vehicle_slow_distance

                    if forward_dist < stop_d:
                        return 0.0

                    if forward_dist < slow_d:
                        t = (forward_dist - stop_d) / max(slow_d - stop_d, 0.001)
                        factor = 0.5 * (1.0 - math.cos(t * math.pi))
                        min_factor = min(min_factor, self.clamp(factor, 0.10, 1.0))
                continue

            # Frontale: anticipa molto prima. Se la separazione laterale è poca,
            # rallenta entrambi; se si sono già spostati ai lati, lascia scorrere.
            if opposite_direction:
                in_headon_zone = (
                    -0.5 < forward_dist < self.vehicle_headon_warn_distance
                    and abs_side_dist < self.vehicle_headon_side_corridor
                )
                if in_headon_zone:
                    if forward_dist < self.vehicle_headon_stop_distance and abs_side_dist < 1.25:
                        return 0.0

                    distance_t = (forward_dist - self.vehicle_headon_stop_distance) / max(
                        self.vehicle_headon_warn_distance - self.vehicle_headon_stop_distance,
                        0.001
                    )
                    distance_t = self.clamp(distance_t, 0.0, 1.0)
                    lateral_t = self.clamp(abs_side_dist / max(self.vehicle_headon_side_corridor, 0.001), 0.0, 1.0)

                    # Se è ancora quasi centrato: forte rallentamento. Se si è
                    # già spostato di lato: permette di passare piano.
                    factor = max(0.18 + 0.62 * distance_t, 0.30 + 0.45 * lateral_t)
                    min_factor = min(min_factor, self.clamp(factor, 0.18, 0.88))
                continue

            # Incrocio/perpendicolare: priorità stabile; quello che cede si ferma.
            if crossing_direction:
                crossing_conflict = (
                    -1.5 < forward_dist < 8.0
                    and abs_side_dist < 4.8
                    and euclidean_dist < 8.0
                )
                if crossing_conflict:
                    min_factor = min(min_factor, 0.0 if i_must_yield else 0.82)
                continue

            if euclidean_dist < 4.0:
                min_factor = min(min_factor, 0.12 if i_must_yield else 0.80)

        return min_factor

    def get_vehicle_avoidance_target(self):
        """
        Target laterale vehicle-aware.

        Per il frontale non faccio più una micro-schivata da 40 cm quando è
        troppo tardi: appena un veicolo opposto entra nella zona di rischio,
        punto a una posizione più estrema sulla mia destra. Siccome anche
        l'altro veicolo fa lo stesso rispetto al suo yaw, i due si separano.
        """
        if not self.has_odom or not self.soft_avoidance_enabled:
            return None

        now = self.get_clock().now().nanoseconds / 1e9

        forward_x = math.cos(self.current_yaw)
        forward_y = math.sin(self.current_yaw)
        right_x = math.sin(self.current_yaw)
        right_y = -math.cos(self.current_yaw)

        best_headon = None
        best_headon_score = float("inf")
        best_stop = None
        best_stop_score = float("inf")

        for vehicle_id, other in list(self.other_vehicles.items()):
            stamp = float(other.get("stamp", 0.0))
            if now - stamp > self.vehicle_state_stale_timeout_sec:
                continue

            other_x = float(other.get("x", 0.0))
            other_y = float(other.get("y", 0.0))
            other_yaw = float(other.get("yaw", 0.0))

            dx = other_x - self.current_x
            dy = other_y - self.current_y

            euclidean_dist = math.sqrt(dx * dx + dy * dy)
            forward_dist = dx * forward_x + dy * forward_y
            side_dist = dx * right_x + dy * right_y

            if forward_dist < -0.75 and euclidean_dist > 1.8:
                continue

            heading_diff = abs(self.normalize_angle(other_yaw - self.current_yaw))
            same_direction = heading_diff < math.radians(45)
            opposite_direction = heading_diff > math.radians(145)
            crossing_direction = not same_direction and not opposite_direction

            i_must_yield = self.should_yield_to_vehicle(vehicle_id)

            if same_direction:
                same_lane_following_risk = (
                    0.0 < forward_dist < self.vehicle_stop_distance
                    and abs(side_dist) < self.vehicle_corridor_width
                )
                if same_lane_following_risk:
                    score = forward_dist + abs(side_dist) * 0.25
                    if score < best_stop_score:
                        best_stop_score = score
                        best_stop = {
                            "type": "STOP",
                            "reason": "following_vehicle_too_close",
                            "vehicle_id": vehicle_id,
                            "forward_dist": forward_dist,
                            "side_dist": side_dist,
                            "euclidean_dist": euclidean_dist,
                            "heading_diff": heading_diff,
                        }
                continue

            if crossing_direction:
                crossing_risk = (
                    -0.5 < forward_dist < 7.5
                    and abs(side_dist) < 4.2
                    and euclidean_dist < 7.5
                )
                if crossing_risk and i_must_yield:
                    score = max(0.0, forward_dist) + abs(side_dist) * 0.35
                    if score < best_stop_score:
                        best_stop_score = score
                        best_stop = {
                            "type": "STOP",
                            "reason": "crossing_vehicle_yield",
                            "vehicle_id": vehicle_id,
                            "forward_dist": forward_dist,
                            "side_dist": side_dist,
                            "euclidean_dist": euclidean_dist,
                            "heading_diff": heading_diff,
                        }
                continue

            headon_risk = (
                opposite_direction
                and 0.4 < forward_dist < self.vehicle_headon_warn_distance
                and abs(side_dist) < self.vehicle_headon_side_corridor
                and euclidean_dist < self.vehicle_headon_warn_distance + 1.0
            )

            if headon_risk:
                score = forward_dist + abs(side_dist) * 0.25
                if score < best_headon_score:
                    best_headon_score = score
                    best_headon = {
                        "vehicle_id": vehicle_id,
                        "forward_dist": forward_dist,
                        "side_dist": side_dist,
                        "euclidean_dist": euclidean_dist,
                        "heading_diff": heading_diff,
                    }

        if best_stop is not None:
            return best_stop

        if best_headon is None:
            return None

        left_clear = self.normalize_clearance(self.obstacle_left_min_distance)
        right_clear = self.normalize_clearance(self.obstacle_right_min_distance)

        # Se la destra è chiaramente chiusa, non invado la corsia opposta:
        # meglio fermarsi che scartare a sinistra in un frontale.
        if right_clear < 1.0 and left_clear > right_clear + 0.7:
            return {
                "type": "STOP",
                "reason": "headon_right_blocked",
                **best_headon,
            }

        urgency = 1.0 - self.clamp(
            (best_headon["forward_dist"] - self.vehicle_headon_stop_distance) / max(
                self.vehicle_headon_warn_distance - self.vehicle_headon_stop_distance,
                0.001
            ),
            0.0,
            1.0
        )

        lane_anchor = self.get_current_lane_anchor_for_avoidance()
        extra_right = self.lane_width * self.vehicle_headon_extra_right_ratio

        # Se il veicolo è quasi al centro della nostra traiettoria, spingo di più.
        if abs(best_headon["side_dist"]) < 0.8:
            extra_right *= 1.25
        if right_clear > left_clear + 0.6:
            extra_right *= 1.15

        extra_right = self.clamp(extra_right, 0.55, self.lane_width * 0.65)
        evade_forward = self.clamp(best_headon["forward_dist"] * 0.45, 2.2, 5.0)

        speed_factor = self.clamp(0.28 + 0.50 * (1.0 - urgency), 0.24, 0.78)

        return {
            "type": "HEAD_ON_KEEP_RIGHT",
            "x": lane_anchor["x"] + forward_x * evade_forward + right_x * extra_right,
            "y": lane_anchor["y"] + forward_y * evade_forward + right_y * extra_right,
            "speed_factor": speed_factor,
            "reason": best_headon,
            "right_clear": right_clear,
            "left_clear": left_clear,
        }

    # ============================================================
    # SEMAFORI
    # ============================================================

    def cleanup_committed_traffic_lights(self):
        now = self.get_clock().now().nanoseconds / 1e9
        expired = [
            node_id
            for node_id, expire_at in self.committed_traffic_lights.items()
            if expire_at <= now
        ]

        for node_id in expired:
            del self.committed_traffic_lights[node_id]

    
    def commit_traffic_light(self, node_id):
        """
        Segna temporaneamente un semaforo come già 'preso'.

        Con TTL basso, tipo 4s, evita che il veicolo rimanga autorizzato
        troppo a lungo se ha visto verde ma poi rallenta prima dell'incrocio.
        """
        if not node_id:
            return

        now_sec = self.get_clock().now().nanoseconds / 1e9
        expire_at = now_sec + self.traffic_light_commit_ttl_sec
        self.committed_traffic_lights[node_id] = expire_at

    
    def must_wait_at_traffic_light(self, current_wp, goal, distance_to_wp):
        self.cleanup_committed_traffic_lights()

        # Il semaforo si controlla solo sui waypoint di approccio.
        if current_wp.get("kind") != "approach_intersection":
            return False

        intersection_node_id = current_wp.get("node_id")
        from_node_id = current_wp.get("from_node_id")
        to_node_id = current_wp.get("to_node_id")

        if not intersection_node_id:
            return False

        # FIX IMPORTANTE:
        # se questo incrocio non è tra quelli semaforizzati,
        # non devo aspettare nessun semaforo.
        if intersection_node_id not in self.traffic_light_node_ids:
            return False

        # Se ho già preso il verde da poco, lascio passare.
        if intersection_node_id in self.committed_traffic_lights:
            return False

        if not from_node_id:
            from_node_id, to_node_id = self.get_movement_for_intersection(intersection_node_id)

        if not from_node_id:
            return False

        # Chiedo priorità già quando sono abbastanza vicino.
        if distance_to_wp <= self.traffic_light_stop_distance * 5.0:
            self.maybe_publish_priority_request(
                from_node_id,
                to_node_id,
                intersection_node_id,
                goal.mission_id
            )

        # Se sono ancora lontano dalla linea di stop, continuo.
        if distance_to_wp > self.traffic_light_stop_distance:
            return False

        now_sec = self.get_clock().now().nanoseconds / 1e9

        # Se il semaforo dovrebbe esistere ma non ho ancora ricevuto lo stato,
        # mi fermo, ma non per sempre: faccio partire il timeout.
        if intersection_node_id not in self.traffic_light_statuses:
            started = self.traffic_light_wait_started_at.get(intersection_node_id)

            if started is None:
                self.traffic_light_wait_started_at[intersection_node_id] = now_sec
                started = now_sec

                self.alert(
                    "TRAFFIC_LIGHT_NO_STATUS",
                    f"nessuno status per semaforo {intersection_node_id}: mi fermo per sicurezza",
                    throttle=True
                )

            waited = now_sec - started

            if waited > self.traffic_light_wait_timeout_sec:
                self.alert(
                    "TRAFFIC_LIGHT_UNKNOWN_TIMEOUT",
                    f"nessuno status per semaforo {intersection_node_id} da {waited:.1f}s: "
                    f"forzo passaggio",
                    throttle=True
                )

                self.commit_traffic_light(intersection_node_id)
                self.traffic_light_wait_started_at.pop(intersection_node_id, None)
                self.state = ExecutorState.NAVIGATING
                return False

            self.state = ExecutorState.WAITING_TRAFFIC_LIGHT
            return True

        color = self.get_signal_color_for_branch(
            intersection_node_id,
            from_node_id,
            to_node_id
        )

        if color == "green":
            self.commit_traffic_light(intersection_node_id)
            self.traffic_light_wait_started_at.pop(intersection_node_id, None)

            if self.state == ExecutorState.WAITING_TRAFFIC_LIGHT:
                self.state = ExecutorState.NAVIGATING

            return False

        started = self.traffic_light_wait_started_at.get(intersection_node_id)

        if started is None:
            self.traffic_light_wait_started_at[intersection_node_id] = now_sec
            started = now_sec

            self.alert(
                "TRAFFIC_LIGHT_WAIT",
                f"stop al semaforo {intersection_node_id}: colore={color}, "
                f"movimento={from_node_id}->{intersection_node_id}->{to_node_id}",
                throttle=True
            )
            self.log_traffic_light_wait_start(intersection_node_id, color)

        waited = now_sec - started

        if waited > self.traffic_light_wait_timeout_sec:
            self.alert(
                "TRAFFIC_LIGHT_STUCK",
                f"attesa {intersection_node_id} da {waited:.1f}s: "
                f"forzo passaggio e richiedo priorità",
                throttle=True
            )

            self.maybe_publish_priority_request(
                from_node_id,
                to_node_id,
                intersection_node_id,
                goal.mission_id
            )

            self.commit_traffic_light(intersection_node_id)
            self.traffic_light_wait_started_at.pop(intersection_node_id, None)
            self.state = ExecutorState.NAVIGATING
            return False

        self.state = ExecutorState.WAITING_TRAFFIC_LIGHT
        return True

    def maybe_publish_priority_request(self, from_node_id, to_node_id, intersection_node_id, mission_id):
        now_sec = self.get_clock().now().nanoseconds / 1e9
        last_sent = self.last_priority_request_time.get(intersection_node_id, 0.0)

        if now_sec - last_sent < 2.0:
            return

        payload = {
            "vehicle_id": self.vehicle_id,
            "mission_id": mission_id,
            "node_id": intersection_node_id,
            "from_node_id": from_node_id,
            "to_node_id": to_node_id,
            "priority": 1,
        }

        msg = String()
        msg.data = json.dumps(payload)
        self.priority_pub.publish(msg)
        self.last_priority_request_time[intersection_node_id] = now_sec

        self.log_navigation_event(
            f"chiedo priorità al semaforo {intersection_node_id}: "
            f"movimento {from_node_id}->{intersection_node_id}->{to_node_id}"
        )

    def get_signal_color_for_branch(self, intersection_node_id, from_node_id, to_node_id=None):
        status = self.traffic_light_statuses.get(intersection_node_id)
        if status is None:
            return "green"

        for signal in status.get("signal_states", []):
            if signal.get("from_node_id") == from_node_id:
                return str(signal.get("color", "red")).lower()

        # Fallback: vecchia logica a movimenti consentiti
        allowed = status.get("allowed_movements", [])
        for movement in allowed:
            if movement.get("from") == from_node_id:
                if to_node_id is None or movement.get("to") == to_node_id:
                    return "green"

        return "red"

    def is_movement_allowed(self, from_node_id, to_node_id, intersection_node_id):
        return self.get_signal_color_for_branch(
            intersection_node_id, from_node_id, to_node_id
        ) == "green"

    def get_movement_for_intersection(self, intersection_node_id):
        if not self.node_path:
            return None, None
        try:
            idx = self.node_path.index(intersection_node_id)
        except ValueError:
            return None, None

        from_node_id = self.node_path[idx - 1] if idx > 0 else None
        to_node_id = self.node_path[idx + 1] if idx < len(self.node_path) - 1 else None
        return from_node_id, to_node_id

    # ============================================================
    # PATH PLANNING
    # ============================================================

    def path_uses_blocked_edge(self, node_path):
        if len(node_path) < 2:
            return False
        for i in range(len(node_path) - 1):
            edge = self.get_edge_between(node_path[i], node_path[i + 1])
            if edge["id"] in self.blocked_edges:
                return True
        return False

    def find_nearest_lane_projection(
        self, x, y,
        preferred_edge_id=None,
        destination_node_id=None,
        allow_blocked=False
    ):
        best = None
        best_distance = float("inf")

        candidate_edges = self.edges
        if preferred_edge_id in self.edge_by_id:
            candidate_edges = [self.edge_by_id[preferred_edge_id]]

        for edge in candidate_edges:
            if not allow_blocked and edge["id"] in self.blocked_edges:
                continue

            a = self.nodes[edge["from"]]
            b = self.nodes[edge["to"]]

            center_projection = self.project_point_on_segment(
                x, y, a["x"], a["y"], b["x"], b["y"]
            )

            if destination_node_id in (edge["from"], edge["to"]):
                possible_destinations = [destination_node_id]
            else:
                possible_destinations = [edge["from"], edge["to"]]

            for dest in possible_destinations:
                lane_projection = self.project_center_projection_to_right_lane(
                    {
                        "edge": edge,
                        "center_x": center_projection["x"],
                        "center_y": center_projection["y"],
                        "x": center_projection["x"],
                        "y": center_projection["y"],
                        "t": center_projection["t"],
                    },
                    dest
                )

                distance = self.distance_xy(x, y, lane_projection["x"], lane_projection["y"])

                if distance < best_distance:
                    best_distance = distance
                    best = {
                        "edge": edge,
                        "center_x": center_projection["x"],
                        "center_y": center_projection["y"],
                        "x": lane_projection["x"],
                        "y": lane_projection["y"],
                        "t": center_projection["t"],
                        "distance": distance,
                        "destination_node_id": dest,
                    }

        if best is None:
            if allow_blocked:
                raise RuntimeError("impossibile proiettare sulla corsia")
            return self.find_nearest_lane_projection(
                x, y, preferred_edge_id=preferred_edge_id,
                destination_node_id=destination_node_id,
                allow_blocked=True
            )

        return best

    def build_navigation_path(self, start_x, start_y, target_x, target_y):
        # 1. Proiezioni grezze per identificare gli edge candidati
        start_projection_raw = self.find_nearest_lane_projection(
            start_x, start_y, allow_blocked=True
        )
        target_projection_raw = self.find_nearest_lane_projection(
            target_x, target_y, allow_blocked=False
        )

        start_edge_raw = start_projection_raw["edge"]
        target_edge_raw = target_projection_raw["edge"]

        start_candidates = [start_edge_raw["from"], start_edge_raw["to"]]
        target_candidates = [target_edge_raw["from"], target_edge_raw["to"]]

        # 2. Scelta del miglior percorso tra candidati
        best_node_path = None
        best_cost = float("inf")

        for s in start_candidates:
            for t in target_candidates:
                node_path, graph_cost = self.shortest_path(s, t)
                if node_path is None:
                    continue
                if self.path_uses_blocked_edge(node_path):
                    continue

                total_cost = (
                    graph_cost
                    + self.distance_xy(start_x, start_y, self.nodes[s]["x"], self.nodes[s]["y"])
                    + self.distance_xy(target_x, target_y, self.nodes[t]["x"], self.nodes[t]["y"])
                )

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_node_path = node_path

        if not best_node_path:
            raise RuntimeError("nessun path trovato sul grafo")

        # 3. Proiezioni corsia con verso corretto
        first_destination = (
            best_node_path[1] if len(best_node_path) > 1 else best_node_path[0]
        )
        final_destination = best_node_path[-1]

        start_projection = self.find_nearest_lane_projection(
            start_x, start_y,
            allow_blocked=True,
            destination_node_id=first_destination
        )
        target_projection = self.find_nearest_lane_projection(
            target_x, target_y,
            allow_blocked=False,
            destination_node_id=final_destination
        )

        start_edge = start_projection["edge"]
        target_edge = target_projection["edge"]

        # 4. Costruzione waypoint con approach/corner/exit per gli incroci
        waypoints = []

        waypoints.append({
            "x": start_projection["x"],
            "y": start_projection["y"],
            "edge_id": start_edge["id"],
            "node_id": first_destination,
            "from_node_id": (
                start_edge["from"] if first_destination == start_edge["to"] else start_edge["to"]
            ),
            "to_node_id": first_destination,
            "kind": "start_lane_projection",
            "speed_limit": start_edge["speed_limit"],
        })

        for i in range(len(best_node_path) - 1):
            current_node_id = best_node_path[i]
            next_node_id = best_node_path[i + 1]
            following_node_id = (
                best_node_path[i + 2] if i + 2 < len(best_node_path) else None
            )

            incoming_edge = self.get_edge_between(current_node_id, next_node_id)

            approach = self.node_to_right_lane_point(
                node_id=next_node_id, other_node_id=current_node_id, mode="approach"
            )

            waypoints.append({
                "x": approach["x"],
                "y": approach["y"],
                "edge_id": incoming_edge["id"],
                "node_id": next_node_id,
                "from_node_id": current_node_id,
                "to_node_id": following_node_id,
                "kind": "approach_intersection",
                "speed_limit": incoming_edge["speed_limit"],
            })

            if following_node_id is not None:
                outgoing_edge = self.get_edge_between(next_node_id, following_node_id)

                exit_point = self.node_to_right_lane_point(
                    node_id=next_node_id, other_node_id=following_node_id, mode="exit"
                )

                corner = self.compute_intersection_corner_point(
                    intersection_node_id=next_node_id,
                    approach_point=approach,
                    exit_point=exit_point
                )

                waypoints.append({
                    "x": corner["x"],
                    "y": corner["y"],
                    "edge_id": incoming_edge["id"],
                    "node_id": next_node_id,
                    "from_node_id": current_node_id,
                    "to_node_id": following_node_id,
                    "kind": "intersection_corner",
                    "speed_limit": min(incoming_edge["speed_limit"], outgoing_edge["speed_limit"], 0.8),
                })

                waypoints.append({
                    "x": exit_point["x"],
                    "y": exit_point["y"],
                    "edge_id": outgoing_edge["id"],
                    "node_id": following_node_id,
                    "from_node_id": next_node_id,
                    "to_node_id": following_node_id,
                    "kind": "exit_intersection",
                    "speed_limit": outgoing_edge["speed_limit"],
                })

        waypoints.append({
            "x": target_projection["x"],
            "y": target_projection["y"],
            "edge_id": target_edge["id"],
            "node_id": final_destination,
            "from_node_id": (
                target_edge["from"] if final_destination == target_edge["to"] else target_edge["to"]
            ),
            "to_node_id": final_destination,
            "kind": "target_lane_projection",
            "speed_limit": target_edge["speed_limit"],
        })

        # 5. Pulizia e log
        waypoints = self.simplify_waypoints(waypoints)
        self.log_built_path(start_x, start_y, target_x, target_y, best_node_path, waypoints)

        return waypoints, best_node_path

    def shortest_path(self, start_node_id, target_node_id):
        queue = [(0.0, start_node_id, [])]
        visited = set()

        while queue:
            cost, node_id, path = heapq.heappop(queue)

            if node_id in visited:
                continue

            visited.add(node_id)
            new_path = path + [node_id]

            if node_id == target_node_id:
                return new_path, cost

            for neighbor_id, edge_id, length in self.adj.get(node_id, []):
                if self.is_edge_blocked(edge_id):
                    continue
                if neighbor_id not in visited:
                    heapq.heappush(queue, (cost + length, neighbor_id, new_path))

        return None, float("inf")

    # ============================================================
    # CORSIE / GEOMETRIA STRADALE
    # ============================================================

    def project_center_projection_to_right_lane(self, projection, destination_node_id):
        edge = projection["edge"]
        a = self.nodes[edge["from"]]
        b = self.nodes[edge["to"]]

        if destination_node_id == edge["to"]:
            from_node, to_node = a, b
        else:
            from_node, to_node = b, a

        base_x = projection.get("center_x", projection["x"])
        base_y = projection.get("center_y", projection["y"])

        lane_x, lane_y = self.apply_right_lane_offset(
            base_x, base_y,
            from_node["x"], from_node["y"],
            to_node["x"], to_node["y"]
        )

        return {
            "x": lane_x,
            "y": lane_y,
            "edge_id": edge["id"],
            "destination_node_id": destination_node_id,
        }

    def compute_intersection_corner_point(self, intersection_node_id, approach_point, exit_point):
        """
        Punto dentro l'incrocio spostato verso il bordo esterno.
        Evita che i veicoli taglino al centro dell'incrocio.
        """
        node = self.nodes[intersection_node_id]
        cx, cy = float(node["x"]), float(node["y"])

        mx = (approach_point["x"] + exit_point["x"]) * 0.5
        my = (approach_point["y"] + exit_point["y"]) * 0.5

        vx = mx - cx
        vy = my - cy
        length = math.sqrt(vx * vx + vy * vy)

        if length < 0.000001:
            return {"x": mx, "y": my}

        vx /= length
        vy /= length
        push = self.lane_width * 0.45

        return {
            "x": mx + vx * push,
            "y": my + vy * push,
        }

    def node_to_right_lane_point(self, node_id, other_node_id, mode):
        node = self.nodes[node_id]
        other = self.nodes[other_node_id]

        # Vettore dal nodo/incrocio verso l'altro nodo.
        dx = other["x"] - node["x"]
        dy = other["y"] - node["y"]
        length = math.sqrt(dx * dx + dy * dy)

        if length <= 0.000001:
            return {"x": node["x"], "y": node["y"]}

        ux, uy = dx / length, dy / length
        clearance = self.intersection_clearance

        if mode == "approach":
            # Se arrivo da "other" verso "node", il punto di approach deve stare
            # dalla parte di "other", quindi node + ux * clearance.
            base_x = node["x"] + ux * clearance
            base_y = node["y"] + uy * clearance

            lane_from_x, lane_from_y = other["x"], other["y"]
            lane_to_x, lane_to_y = node["x"], node["y"]

        elif mode == "exit":
            # Se esco da "node" verso "other", il punto di exit deve stare
            # dalla parte di "other", quindi anche qui node + ux * clearance.
            base_x = node["x"] + ux * clearance
            base_y = node["y"] + uy * clearance

            lane_from_x, lane_from_y = node["x"], node["y"]
            lane_to_x, lane_to_y = other["x"], other["y"]

        else:
            raise RuntimeError(f"mode non valido: {mode}")

        lane_x, lane_y = self.apply_right_lane_offset(
            base_x, base_y,
            lane_from_x, lane_from_y,
            lane_to_x, lane_to_y
        )

        return {"x": lane_x, "y": lane_y}

    def apply_right_lane_offset(self, x, y, from_x, from_y, to_x, to_y):
        dx = to_x - from_x
        dy = to_y - from_y
        length = math.sqrt(dx * dx + dy * dy)

        if length < 1e-6:
            return x, y

        dx /= length
        dy /= length

        # Normale destra rispetto al verso reale
        right_x = dy
        right_y = -dx

        forced_edge_ratio = max(float(self.lane_offset_ratio), 1.45)
        offset = self.lane_width * 0.5 * forced_edge_ratio

        return (x + right_x * offset, y + right_y * offset)

    def get_lane_follow_target(self, waypoint):
        """
        Lane follower puro: non decide più precedenze, stop o schivate veicoli.
        Quelle decisioni stanno nel SafetySupervisor e sono mutuamente esclusive.
        """
        preferred_edge_id = waypoint.get("edge_id")
        destination_node_id = waypoint.get("node_id")

        lane_projection = self.find_nearest_lane_projection(
            self.current_x, self.current_y,
            preferred_edge_id=preferred_edge_id,
            destination_node_id=destination_node_id,
            allow_blocked=True
        )

        if destination_node_id is None:
            destination_node_id = lane_projection.get("destination_node_id")

        edge = lane_projection["edge"]

        dist_current_to_wp = self.distance_xy(
            self.current_x, self.current_y,
            waypoint["x"], waypoint["y"]
        )

        # Waypoint artificiali di incrocio: inseguiti direttamente
        if waypoint.get("kind") in ("intersection_corner", "exit_intersection"):
            return {
                "x": waypoint["x"],
                "y": waypoint["y"],
                "mode_prefix": "EDGE_INTERSECTION",
                "lane_error": lane_projection["distance"],
                "edge_id": edge["id"],
            }

        # Recovery corsia se fuori dal corridoio bordo
        if lane_projection["distance"] > self.lane_recovery_threshold:
            return {
                "x": lane_projection["x"],
                "y": lane_projection["y"],
                "mode_prefix": "EDGE_RECOVERY",
                "lane_error": lane_projection["distance"],
                "edge_id": edge["id"],
            }

        # Approccio diretto al waypoint se vicini
        if dist_current_to_wp <= max(self.lookahead_distance, self.waypoint_tolerance * 4.0):
            return {
                "x": waypoint["x"],
                "y": waypoint["y"],
                "mode_prefix": "WAYPOINT_FINAL_APPROACH",
                "lane_error": lane_projection["distance"],
                "edge_id": edge["id"],
            }

        lookahead = self.compute_lane_lookahead_point(
            edge, lane_projection["t"], destination_node_id, self.lookahead_distance
        )

        dist_lookahead_to_wp = self.distance_xy(
            lookahead["x"], lookahead["y"],
            waypoint["x"], waypoint["y"]
        )

        if dist_lookahead_to_wp > dist_current_to_wp:
            return {
                "x": waypoint["x"],
                "y": waypoint["y"],
                "mode_prefix": "WAYPOINT_APPROACH",
                "lane_error": lane_projection["distance"],
                "edge_id": edge["id"],
            }

        return {
            "x": lookahead["x"],
            "y": lookahead["y"],
            "mode_prefix": "EDGE_LANE_FOLLOW",
            "lane_error": lane_projection["distance"],
            "edge_id": edge["id"],
        }

    def compute_lane_lookahead_point(self, edge, current_t, destination_node_id, lookahead_distance):
        a = self.nodes[edge["from"]]
        b = self.nodes[edge["to"]]

        edge_length = max(edge["length"], 0.000001)
        delta_t = lookahead_distance / edge_length

        if destination_node_id == edge["to"]:
            next_t = min(1.0, current_t + delta_t)
            lane_destination = edge["to"]
        else:
            next_t = max(0.0, current_t - delta_t)
            lane_destination = edge["from"]

        center_x = a["x"] + (b["x"] - a["x"]) * next_t
        center_y = a["y"] + (b["y"] - a["y"]) * next_t

        lane = self.project_center_projection_to_right_lane(
            {"edge": edge, "x": center_x, "y": center_y, "t": next_t},
            lane_destination
        )

        return lane

    def project_point_on_segment(self, px, py, ax, ay, bx, by):
        dx = bx - ax
        dy = by - ay
        denom = dx * dx + dy * dy

        if denom <= 0.000001:
            return {"x": ax, "y": ay, "t": 0.0}

        t = ((px - ax) * dx + (py - ay) * dy) / denom
        t = max(0.0, min(1.0, t))

        return {"x": ax + t * dx, "y": ay + t * dy, "t": t}

    def get_edge_between(self, a, b):
        for edge in self.edges:
            if (edge["from"] == a and edge["to"] == b) or \
               (edge["from"] == b and edge["to"] == a):
                return edge
        raise RuntimeError(f"nessun edge tra {a} e {b}")

    def simplify_waypoints(self, waypoints):
        if not waypoints:
            return []

        result = [waypoints[0]]

        for wp in waypoints[1:]:
            last = result[-1]
            d = self.distance_xy(last["x"], last["y"], wp["x"], wp["y"])
            # FIX: mantieni i waypoint di tipo speciale anche se troppo vicini
            if d > 0.05 or wp.get("kind") in ("approach_intersection", "intersection_corner", "exit_intersection"):
                result.append(wp)

        return result

    # ============================================================
    # CONTROLLO MOVIMENTO
    # ============================================================

    def apply_navigation_decision(self, decision, goal, current_wp):
        self.last_decision_type = decision.type.value
        self.last_decision_reason = decision.reason
        self.last_decision_payload = decision.payload

        self.log_vehicle_collision_decision(decision)

        """Esegue una sola decisione per tick. Ritorna True se i timer del waypoint vanno resettati."""
        if decision.type == DecisionType.STOP:
            self.stop_vehicle()
            return True

        if decision.type == DecisionType.VEHICLE_DEADLOCK_RECOVERY:
            conflict = decision.payload.get("conflict", {})
            self.recovery_controller.run_vehicle_deadlock(conflict)
            return True

        if decision.type == DecisionType.LIDAR_OBSTACLE_RECOVERY:
            self.recovery_controller.run_lidar_obstacle_escape(goal, current_wp)
            return True

        if decision.type == DecisionType.GENERIC_STUCK_RECOVERY:
            self.recovery_controller.run_generic_stuck_recovery(current_wp, goal)
            return True

        if decision.type == DecisionType.RESET_WAYPOINT_TIMER:
            self.stop_vehicle()
            return True

        if decision.type == DecisionType.TEMPORARY_TARGET:
            if decision.target_x is None or decision.target_y is None:
                self.stop_vehicle()
                return True
            self.move_towards_temporary_target(
                decision.target_x,
                decision.target_y,
                goal.max_speed,
                decision.speed_factor,
                decision.reason,
            )
            return False

        if decision.type == DecisionType.SLOW_FOLLOW:
            self.move_towards_waypoint(current_wp, goal.max_speed, decision.speed_factor)
            return False

        # FOLLOW_LANE normale
        if self.state == ExecutorState.OBSTACLE_STOP:
            self.state = ExecutorState.NAVIGATING
            self.obstacle_stop_start_time = None
        self.move_towards_waypoint(current_wp, goal.max_speed, 1.0)
        return False

    def move_towards_temporary_target(
        self,
        target_x,
        target_y,
        requested_max_speed,
        speed_factor=1.0,
        mode_prefix="TEMPORARY_TARGET"
    ):
        """
        Controllo locale per target temporanei scelti da una policy.
        Per le manovre anti-veicolo NON deve mai diventare rotazione sul posto:
        deve fare archi larghi e decisi.
        """
        dx = target_x - self.current_x
        dy = target_y - self.current_y

        target_angle = math.atan2(dy, dx)
        angle_error = self.normalize_angle(target_angle - self.current_yaw)
        distance = math.sqrt(dx * dx + dy * dy)

        max_speed = (
            float(requested_max_speed)
            if float(requested_max_speed) > 0.0
            else self.default_max_speed
        )
        max_speed = min(max_speed, self.default_map_speed_limit)

        angular_speed = -self.clamp(
            self.angular_k * angle_error,
            -self.max_angular_speed,
            self.max_angular_speed
        )

        abs_error = abs(angle_error)

        aggressive = (
            "priority_pass" in str(mode_prefix)
            or "headon" in str(mode_prefix)
            or "close_priority" in str(mode_prefix)
        )

        if aggressive:
            # Mai linear_speed = 0 nelle manovre anti-incastro.
            if abs_error > 1.05:
                linear_speed = min(max_speed * 0.26, 0.42)
                motion_mode = "AGGRESSIVE_ARC_HARD"
            elif abs_error > 0.70:
                linear_speed = min(max_speed * 0.38, 0.58)
                motion_mode = "AGGRESSIVE_ARC"
            elif abs_error > 0.35:
                linear_speed = min(max_speed * 0.55, 0.78)
                motion_mode = "AGGRESSIVE_SOFT_ARC"
            else:
                linear_speed = min(max_speed * 0.85, self.linear_k * distance, max_speed)
                motion_mode = "AGGRESSIVE_FORWARD"
        else:
            if abs_error > 0.85:
                linear_speed = min(max_speed * 0.18, 0.30)
                motion_mode = "ARC_TURN_HARD"
            elif abs_error > 0.55:
                linear_speed = min(max_speed * 0.25, 0.40)
                motion_mode = "ARC_TURN"
            elif abs_error > 0.35:
                linear_speed = min(max_speed * 0.38, 0.55)
                motion_mode = "SLOW_ARC"
            elif abs_error > 0.15:
                linear_speed = min(max_speed * 0.60, self.linear_k * distance, max_speed)
                motion_mode = "SOFT_TURN"
            else:
                linear_speed = min(max_speed, self.linear_k * distance)
                motion_mode = "FORWARD"

        linear_speed = self.clamp(linear_speed * speed_factor, 0.0, max_speed)

        cmd = Twist()
        cmd.linear.x = float(linear_speed)
        cmd.angular.z = float(angular_speed)

        self.last_cmd_linear_x = float(cmd.linear.x)
        self.last_cmd_angular_z = float(cmd.angular.z)

        self.cmd_vel_pub.publish(cmd)

        self._last_temporary_control = {
            "target_x": target_x,
            "target_y": target_y,
            "target_angle": target_angle,
            "angle_error": angle_error,
            "distance": distance,
            "linear_speed": linear_speed,
            "angular_speed": angular_speed,
            "motion_mode": f"{mode_prefix}_{motion_mode}",
            "speed_factor": speed_factor,
        }

    def move_towards_waypoint(self, waypoint, requested_max_speed, obstacle_factor=1.0):
        """
        FIX principali rispetto all'originale:
        - La velocità angolare ora usa la convenzione corretta (segno positivo = CCW).
          Nell'originale il segno era invertito (-clamp), il che produceva sterzate
          nella direzione sbagliata in certi scenari.
        - Le soglie di riduzione velocità in curva ora scalano con max_speed
          anziché essere valori fissi: evita il problema "troppo lento" a bassa
          speed_limit e "troppo veloce" ad alta speed_limit.
        - La modalità TURN_IN_PLACE non azzera mai la velocità in avanti
          se il target è molto vicino (< waypoint_tolerance * 2): evita
          oscillazioni sul posto davanti al waypoint finale.
        - VEHICLE_AVOIDANCE_RIGHT: velocità minima rimossa. Il veicolo
          accelera naturalmente verso il target di schivata invece di
          avere un floor artificiale che causava sobbalzi.
        """
        follow_target = self.get_lane_follow_target(waypoint)

        if follow_target["mode_prefix"] == "VEHICLE_CROSSING_STOP":
            self.stop_vehicle()
            waypoint["_last_control"] = {
                "target_x": self.current_x, "target_y": self.current_y,
                "target_angle": self.current_yaw, "angle_error": 0.0,
                "distance": 0.0, "lane_error": follow_target["lane_error"],
                "linear_speed": 0.0, "angular_speed": 0.0,
                "motion_mode": "VEHICLE_CROSSING_STOP",
                "max_speed": 0.0, "obstacle_factor": obstacle_factor,
                "edge_id": follow_target["edge_id"],
            }
            return

        target_x = follow_target["x"]
        target_y = follow_target["y"]

        dx = target_x - self.current_x
        dy = target_y - self.current_y

        target_angle = math.atan2(dy, dx)
        angle_error = self.normalize_angle(target_angle - self.current_yaw)

        distance = math.sqrt(dx * dx + dy * dy)

        edge_speed_limit = float(waypoint.get("speed_limit", self.default_map_speed_limit))
        max_speed = (
            float(requested_max_speed) if float(requested_max_speed) > 0.0
            else self.default_max_speed
        )
        max_speed = min(max_speed, edge_speed_limit)

        angular_speed = -self.clamp(
            self.angular_k * angle_error,
            -self.max_angular_speed,
            self.max_angular_speed
        )

        abs_error = abs(angle_error)

        special_wp = waypoint.get("kind") in (
            "start_lane_projection",
            "approach_intersection",
            "intersection_corner",
            "exit_intersection",
        )

        near_wp = distance < 3.5

        if abs_error > 0.85:
            # Prima qui spesso faceva linear_speed = 0.
            # Negli incroci è pessimo: il veicolo resta lì a ruotare/dondolare.
            if special_wp or near_wp:
                linear_speed = min(max_speed * 0.22, 0.38)
                motion_mode = "ARC_TURN_HARD"
            else:
                linear_speed = 0.0
                motion_mode = "TURN_IN_PLACE"

        elif abs_error > 0.55:
            linear_speed = min(max_speed * 0.28, 0.48)
            motion_mode = "ARC_TURN"

        elif abs_error > 0.35:
            linear_speed = min(max_speed * 0.42, 0.68)
            motion_mode = "SLOW_ARC"

        elif abs_error > 0.15:
            linear_speed = min(max_speed * 0.65, self.linear_k * distance, max_speed)
            motion_mode = "SOFT_TURN"

        else:
            linear_speed = min(max_speed, self.linear_k * distance)
            motion_mode = "FORWARD"

        linear_speed = self.clamp(linear_speed * obstacle_factor, 0.0, max_speed)

        if follow_target["mode_prefix"]:
            motion_mode = f'{follow_target["mode_prefix"]}_{motion_mode}'

        cmd = Twist()
        cmd.linear.x = float(linear_speed)
        cmd.angular.z = float(angular_speed)

        self.last_cmd_linear_x = float(cmd.linear.x)
        self.last_cmd_angular_z = float(cmd.angular.z)

        self.cmd_vel_pub.publish(cmd)

        waypoint["_last_control"] = {
            "target_x": target_x, "target_y": target_y,
            "target_angle": target_angle, "angle_error": angle_error,
            "distance": distance, "lane_error": follow_target["lane_error"],
            "linear_speed": linear_speed, "angular_speed": angular_speed,
            "motion_mode": motion_mode, "max_speed": max_speed,
            "obstacle_factor": obstacle_factor, "edge_id": follow_target["edge_id"],
        }

    def stop_vehicle(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0

        self.last_cmd_linear_x = 0.0
        self.last_cmd_angular_z = 0.0

        self.cmd_vel_pub.publish(cmd)

    # ============================================================
    # DIAGNOSTICA / RECOVERY
    # ============================================================

    def validate_runtime_parameters(self):
        # Correzioni silenziose: l'output console deve restare pulito.
        if self.obstacle_slow_distance <= self.obstacle_stop_distance:
            self.obstacle_slow_distance = self.obstacle_stop_distance + 1.0

        if self.traffic_light_stop_distance >= self.intersection_clearance * 4.0:
            # Mantengo il valore scelto, ma non stampo warning.
            pass

    def alert(self, code, message, throttle=True):
        # Gli alert restano disponibili sul topic navigation_alerts.
        # La console viene gestita solo dai metodi log_* mirati qui sopra.
        now_sec = self.get_clock().now().nanoseconds / 1e9
        key = str(code)
        last = self.last_alert_time_by_key.get(key, 0.0)

        if throttle and now_sec - last < self.alert_log_period_sec:
            return

        self.last_alert_time_by_key[key] = now_sec

        msg = String()
        msg.data = json.dumps({
            "code": code,
            "vehicle_id": self.vehicle_id,
            "mission_id": self.current_mission_id,
            "message": message,
            "state": self.state.value,
            "x": self.current_x,
            "y": self.current_y,
            "yaw": self.current_yaw,
            "stamp": now_sec,
        })
        self.alert_pub.publish(msg)

    def update_progress_watchdog(self, distance_to_wp):
        now = self.get_clock().now()

        if self.last_progress_distance is None:
            self.last_progress_distance = distance_to_wp
            self.last_progress_time = now
            return

        if distance_to_wp < self.last_progress_distance - self.stuck_progress_epsilon:
            self.last_progress_distance = distance_to_wp
            self.last_progress_time = now
            return

        if distance_to_wp > self.last_progress_distance + 1.0:
            self.last_progress_distance = distance_to_wp

    def is_stuck_without_reason(self, distance_to_wp):
        if self.state in (
            ExecutorState.WAITING_TRAFFIC_LIGHT,
            ExecutorState.OBSTACLE_STOP,
            ExecutorState.RECALCULATING,
            ExecutorState.IDLE,
        ):
            return False

        if self.obstacle_min_distance <= self.obstacle_stop_distance:
            return False

        if distance_to_wp <= self.waypoint_tolerance * 2.0:
            return False

        elapsed = (self.get_clock().now() - self.last_progress_time).nanoseconds / 1e9
        return elapsed > self.stuck_timeout_sec

    def reset_motion_watchdog(self):
        self.last_motion_x = self.current_x
        self.last_motion_y = self.current_y
        self.last_motion_time = self.get_clock().now()

    def update_motion_watchdog(self):
        now = self.get_clock().now()

        if self.last_motion_x is None or self.last_motion_y is None:
            self.reset_motion_watchdog()
            return

        moved = self.distance_xy(
            self.current_x, self.current_y,
            self.last_motion_x, self.last_motion_y
        )

        if moved >= self.vehicle_deadlock_min_displacement:
            self.last_motion_x = self.current_x
            self.last_motion_y = self.current_y
            self.last_motion_time = now

    def detect_vehicle_deadlock(self):
        if not self.has_odom:
            return None

        if self.state in (
            ExecutorState.WAITING_TRAFFIC_LIGHT,
            ExecutorState.RECALCULATING,
            ExecutorState.IDLE,
        ):
            return None

        now = self.get_clock().now()
        now_sec = now.nanoseconds / 1e9

        if now_sec - self.last_vehicle_deadlock_recovery_time < 1.6:
            return None

        stopped_for = (now - self.last_motion_time).nanoseconds / 1e9
        if stopped_for < self.vehicle_deadlock_detection_sec:
            return None

        conflict = self.get_nearest_vehicle_conflict()
        if conflict is None:
            return None

        conflict["stopped_for"] = stopped_for
        return conflict

    def get_nearest_vehicle_conflict(self):
        now_sec = self.get_clock().now().nanoseconds / 1e9

        forward_x = math.cos(self.current_yaw)
        forward_y = math.sin(self.current_yaw)
        right_x = math.sin(self.current_yaw)
        right_y = -math.cos(self.current_yaw)

        best = None
        best_score = float("inf")

        for vehicle_id, other in list(self.other_vehicles.items()):
            stamp = float(other.get("stamp", 0.0))
            if now_sec - stamp > self.vehicle_state_stale_timeout_sec:
                continue

            other_x = float(other.get("x", 0.0))
            other_y = float(other.get("y", 0.0))
            other_yaw = float(other.get("yaw", 0.0))

            dx = other_x - self.current_x
            dy = other_y - self.current_y

            euclidean_dist = math.sqrt(dx * dx + dy * dy)
            forward_dist = dx * forward_x + dy * forward_y
            side_dist = dx * right_x + dy * right_y

            heading_diff = abs(self.normalize_angle(other_yaw - self.current_yaw))
            same_direction = heading_diff < math.radians(45)
            opposite_direction = heading_diff > math.radians(135)

            # Frontale/contatto: anche se il LiDAR o il target laterale stanno
            # provando a risolvere, se il modello non si muove significa che
            # è fisicamente incastrato.
            physical_contact = euclidean_dist < 3.0
            frontal_contact = (
                -0.8 < forward_dist < 4.5
                and abs(side_dist) < self.lane_width * 1.15
                and euclidean_dist < 4.8
            )
            headon_contact = opposite_direction and (physical_contact or frontal_contact)

            # Evita di fare retromarcia in una normale coda: se è stessa
            # direzione e non c'è contatto fisico stretto, aspetta.
            if same_direction and not physical_contact:
                continue

            if not (physical_contact or frontal_contact or headon_contact):
                continue

            score = euclidean_dist + max(0.0, forward_dist) * 0.25 + abs(side_dist) * 0.15
            if score < best_score:
                best_score = score
                best = {
                    "vehicle_id": vehicle_id,
                    "euclidean_dist": euclidean_dist,
                    "forward_dist": forward_dist,
                    "side_dist": side_dist,
                    "heading_diff": heading_diff,
                    "same_direction": same_direction,
                    "opposite_direction": opposite_direction,
                    "physical_contact": physical_contact,
                }

        return best

    def perform_vehicle_unjam_maneuver(self, conflict):
        if not self.enable_recovery_maneuver:
            return

        other_id = conflict.get("vehicle_id")

        # Se devo cedere, non faccio disostruzione attiva.
        # Mi fermo e lascio liberare il veicolo con priorità.
        if other_id and self.should_yield_to_vehicle(other_id):
            self.stop_vehicle()
            return

        self.last_vehicle_deadlock_recovery_time = self.get_clock().now().nanoseconds / 1e9

        previous_state = self.state
        self.state = ExecutorState.STUCK_RECOVERY

        side_dist = float(conflict.get("side_dist", 0.0))
        opposite_direction = bool(conflict.get("opposite_direction", False))

        if opposite_direction:
            # Frontale: retromarcia + propria destra.
            turn_sign = -1.0
        else:
            # Se l'altro è a destra, giro a sinistra; se è a sinistra, giro a destra.
            turn_sign = 1.0 if side_dist >= 0.0 else -1.0

        turn_speed = self.clamp(
            self.vehicle_unjam_turn_speed,
            0.55,
            self.max_angular_speed
        )

        reverse_speed = min(self.vehicle_unjam_reverse_speed, -0.45)

        cmd = Twist()

        # Fase 1: arretra deciso.
        end_reverse = self.get_clock().now().nanoseconds / 1e9 + self.vehicle_unjam_reverse_sec
        while rclpy.ok() and self.get_clock().now().nanoseconds / 1e9 < end_reverse:
            cmd.linear.x = reverse_speed
            cmd.angular.z = turn_sign * turn_speed * 0.55
            self.cmd_vel_pub.publish(cmd)
            time.sleep(0.02)

        # Fase 2: continua ad arretrare, ma gira forte.
        end_turn = self.get_clock().now().nanoseconds / 1e9 + self.vehicle_unjam_turn_sec
        while rclpy.ok() and self.get_clock().now().nanoseconds / 1e9 < end_turn:
            cmd.linear.x = reverse_speed * 0.75
            cmd.angular.z = turn_sign * turn_speed
            self.cmd_vel_pub.publish(cmd)
            time.sleep(0.02)

        # Fase 3: piccolo arco in avanti per uscire dall'incastro.
        end_forward = self.get_clock().now().nanoseconds / 1e9 + 0.55
        while rclpy.ok() and self.get_clock().now().nanoseconds / 1e9 < end_forward:
            cmd.linear.x = 0.35
            cmd.angular.z = turn_sign * turn_speed * 0.45
            self.cmd_vel_pub.publish(cmd)
            time.sleep(0.02)

        self.stop_vehicle()
        self.state = previous_state

    def run_recovery_maneuver(self):
        now_sec = self.get_clock().now().nanoseconds / 1e9
        if now_sec - self.last_recovery_time < 3.0:
            return

        self.last_recovery_time = now_sec
        self.state = ExecutorState.STUCK_RECOVERY

        cmd = Twist()
        cmd.linear.x = -0.25
        cmd.angular.z = 0.35
        end_time = self.get_clock().now().nanoseconds / 1e9 + 0.45

        while rclpy.ok() and self.get_clock().now().nanoseconds / 1e9 < end_time:
            self.cmd_vel_pub.publish(cmd)
            time.sleep(0.02)

        self.stop_vehicle()
        self.state = ExecutorState.NAVIGATING

    # ============================================================
    # LOGGING
    # ============================================================

    def log_mission_start(self, goal):
        return

    def log_mission_arrival(self, goal):
        return

    def log_navigation_event(self, message):
        # No-op intenzionale: la console del navigator mostra solo eventi di attesa/sicurezza.
        return

    def log_console_event(self, key, message, throttle_sec=6.0):
        now_sec = self.get_clock().now().nanoseconds / 1e9
        key = str(key)
        last = self.last_console_event_time_by_key.get(key, 0.0)

        if now_sec - last < throttle_sec:
            return

        self.last_console_event_time_by_key[key] = now_sec
        self.get_logger().info(message)

    def log_traffic_light_wait_start(self, intersection_node_id, color):
        color = str(color).lower()
        color_label = "rosso" if color == "red" else color

        self.log_console_event(
            key=f"traffic_light_wait:{intersection_node_id}",
            message=(
                f"{self.vehicle_id}: inizio attesa per semaforo "
                f"{intersection_node_id} {color_label}"
            ),
            throttle_sec=30.0
        )

    def log_lidar_obstacle_wait_start(self):
        distance = self.obstacle_min_distance
        if math.isfinite(distance):
            detail = f" a {distance:.2f}m"
        else:
            detail = ""

        self.log_console_event(
            key="lidar_obstacle_wait",
            message=f"{self.vehicle_id}: ostacolo rilevato con LiDAR{detail}, attendere",
            throttle_sec=10.0
        )

    def log_vehicle_collision_decision(self, decision):
        reason = str(decision.reason or "")

        if not (reason.startswith("vehicle_") or reason.startswith("deadlock_")):
            return

        other_id = self.extract_vehicle_id_from_decision(decision)
        if not other_id:
            return

        if decision.type in [
            DecisionType.TEMPORARY_TARGET,
            DecisionType.VEHICLE_DEADLOCK_RECOVERY,
        ]:
            action = "manovra"
        else:
            action = "attesa"

        self.log_console_event(
            key=f"vehicle_collision:{other_id}:{action}",
            message=(
                f"{self.vehicle_id}: possibile collisione rilevata con veicolo "
                f"{other_id}, {action}"
            ),
            throttle_sec=8.0
        )

    def extract_vehicle_id_from_decision(self, decision):
        payload = decision.payload if isinstance(decision.payload, dict) else {}

        vehicle_id = payload.get("vehicle_id")
        if vehicle_id:
            return str(vehicle_id)

        conflict = payload.get("conflict")
        if isinstance(conflict, dict) and conflict.get("vehicle_id"):
            return str(conflict.get("vehicle_id"))

        reason = str(decision.reason or "")
        for marker in ["private_car_", "taxi_", "bus_"]:
            index = reason.find(marker)
            if index >= 0:
                return reason[index:]

        return None

    def log_built_path(self, start_x, start_y, target_x, target_y, node_path, waypoints):
        if not self.path_log_enabled:
            return

        self.log_navigation_event(
            f"path build | start=({start_x:.2f},{start_y:.2f}) "
            f"target=({target_x:.2f},{target_y:.2f}) | "
            f"node_path={' -> '.join(node_path)} | "
            f"waypoints={len(waypoints)}"
        )

        for i, wp in enumerate(waypoints):
            self.log_navigation_event(
                f"wp[{i}] kind={wp.get('kind')} "
                f"node={wp.get('node_id')} "
                f"edge={wp.get('edge_id')} "
                f"pos=({wp['x']:.2f},{wp['y']:.2f}) "
                f"speed={wp.get('speed_limit', -1):.2f}"
            )

    def log_path(self, path):
        if not self.path_log_enabled:
            return

        if not self.node_path:
            self.log_navigation_event(f"path calcolato con {len(path)} waypoint, ma senza node_path")
            return

        self.log_navigation_event(
            "percorso grafo scelto: " + " -> ".join(self.node_path)
        )

        for i in range(len(self.node_path) - 1):
            a = self.node_path[i]
            b = self.node_path[i + 1]
            edge = self.get_edge_between(a, b)

            self.log_navigation_event(
                f"tratto {i + 1}: nodo {a} -> nodo {b} "
                f"sull'edge {edge['id']} lungo {edge['length']:.2f} m"
            )

        self.log_navigation_event(f"waypoint generati: {len(path)}")

    def describe_graph_position(self):
        if not self.edges:
            return "grafo non disponibile"

        lane_projection = self.find_nearest_lane_projection(self.current_x, self.current_y)

        edge = lane_projection["edge"]
        from_node_id = edge["from"]
        to_node_id = edge["to"]

        from_node = self.nodes[from_node_id]
        to_node = self.nodes[to_node_id]

        dist_from = self.distance_xy(self.current_x, self.current_y, from_node["x"], from_node["y"])
        dist_to = self.distance_xy(self.current_x, self.current_y, to_node["x"], to_node["y"])

        nearest_node = from_node_id if dist_from <= dist_to else to_node_id
        nearest_dist = min(dist_from, dist_to)

        return (
            f"sono a ({self.current_x:.2f},{self.current_y:.2f}), "
            f"yaw={self.current_yaw:.2f}; "
            f"corsia più vicina=edge {edge['id']} tra {from_node_id} e {to_node_id}; "
            f"lane_projection=({lane_projection['x']:.2f},{lane_projection['y']:.2f}), "
            f"center_projection=({lane_projection['center_x']:.2f},{lane_projection['center_y']:.2f}), "
            f"t={lane_projection['t']:.2f}, "
            f"fuori corsia={lane_projection['distance']:.2f} m; "
            f"nodo più vicino={nearest_node} ({nearest_dist:.2f} m)"
        )

    def describe_waypoint(self, waypoint, distance):
        if waypoint is None:
            return "nessun waypoint attivo"

        edge_id = waypoint.get("edge_id")
        edge = self.edge_by_id.get(edge_id)

        if edge:
            street = f"edge {edge['id']} tra {edge['from']} e {edge['to']}"
        else:
            street = f"edge={edge_id}"

        node_id = waypoint.get("node_id") or "-"
        kind = waypoint.get("kind") or "-"

        return (
            f"sto puntando wp {self.current_waypoint_index + 1}/{len(self.current_path)} "
            f"[{kind}] a ({waypoint['x']:.2f},{waypoint['y']:.2f}), "
            f"dist={distance:.2f} m, node={node_id}, {street}"
        )

    def log_navigation_snapshot(self, reason, waypoint=None, distance=None, force=False):
        if not self.diagnostic_log_enabled and not force:
            return

        now = self.get_clock().now()
        elapsed = (now - self.last_diag_time).nanoseconds / 1e9

        if not force and elapsed < self.diagnostic_log_period_sec:
            return

        self.last_diag_time = now

        if waypoint is None and self.current_waypoint_index < len(self.current_path):
            waypoint = self.current_path[self.current_waypoint_index]

        if distance is None and waypoint is not None:
            distance = self.distance_xy(
                self.current_x, self.current_y, waypoint["x"], waypoint["y"]
            )
        elif distance is None:
            distance = 0.0

        ctrl = waypoint.get("_last_control", {}) if waypoint else {}
        self.log_navigation_event(
            f"{reason} | stato={self.state.value} | "
            f"{self.describe_graph_position()} | "
            f"{self.describe_waypoint(waypoint, distance)} | "
            f"cmd: mode={ctrl.get('motion_mode', '?')}, "
            f"target=({ctrl.get('target_x', 0.0):.2f},{ctrl.get('target_y', 0.0):.2f}), "
            f"lane_err={ctrl.get('lane_error', 0.0):.2f}, "
            f"v={ctrl.get('linear_speed', 0.0):.2f}, "
            f"w={ctrl.get('angular_speed', 0.0):.2f}"
        )

    def maybe_log_diagnostics(self, waypoint, distance):
        if not self.diagnostic_log_enabled:
            return
        self.log_navigation_snapshot("navigazione in corso", waypoint, distance)

    # ============================================================
    # UTILITY
    # ============================================================

    def on_debug_state_request(self, request, response):
        """
        Service di debug brutale.

        Chiamata:
        ros2 service call /<vehicle>/navigation_executor/debug_state std_srvs/srv/Trigger "{}"
        """

        try:
            data = self.build_debug_state()
            response.success = True
            response.message = json.dumps(data, ensure_ascii=False, indent=2)
            return response

        except Exception as ex:
            response.success = False
            response.message = f"Errore debug_state: {ex}"
            return response


    def build_debug_state(self):
        now_sec = self.get_clock().now().nanoseconds / 1e9

        current_wp = None
        distance_to_wp = None

        if self.current_path and self.current_waypoint_index < len(self.current_path):
            current_wp = self.current_path[self.current_waypoint_index]
            distance_to_wp = self.distance_xy(
                self.current_x,
                self.current_y,
                current_wp["x"],
                current_wp["y"]
            )

        return {
            "vehicle": {
                "id": self.vehicle_id,
                "state": self.state.value,
                "mission_id": self.current_mission_id,
                "has_odom": self.has_odom,
                "pose": {
                    "x": self.current_x,
                    "y": self.current_y,
                    "yaw": self.current_yaw,
                },
                "last_cmd": {
                    "linear_x": self.last_cmd_linear_x,
                    "angular_z": self.last_cmd_angular_z,
                },
            },

            "decision": {
                "type": self.last_decision_type,
                "reason": self.last_decision_reason,
                "payload": self._json_safe(self.last_decision_payload),
            },

            "path": {
                "current_waypoint_index": self.current_waypoint_index,
                "path_len": len(self.current_path),
                "node_path": self.node_path,
                "current_wp": self._debug_waypoint(current_wp),
                "distance_to_wp": distance_to_wp,
                "remaining_distance": self.compute_remaining_distance(),
            },

            "traffic_light": {
                "stop_distance": self.traffic_light_stop_distance,
                "wait_timeout_sec": self.traffic_light_wait_timeout_sec,
                "known_status_nodes": list(self.traffic_light_statuses.keys()),
                "current_wp_signal": self._debug_current_traffic_light(current_wp, distance_to_wp),
                "wait_started_at": self.traffic_light_wait_started_at,
                "committed": self._debug_expiring_dict(self.committed_traffic_lights, now_sec),
                "last_priority_request_time": self.last_priority_request_time,
            },

            "lidar": {
                "obstacle_min_distance": self.obstacle_min_distance,
                "obstacle_bearing": self.obstacle_bearing,
                "left_min_distance": self.obstacle_left_min_distance,
                "right_min_distance": self.obstacle_right_min_distance,
                "stop_distance": self.obstacle_stop_distance,
                "slow_distance": self.obstacle_slow_distance,
                "has_known_vehicle_in_front_stop_zone": self.has_known_vehicle_in_front(
                    self.obstacle_stop_distance + 2.0
                ),
                "has_known_vehicle_in_front_slow_zone": self.has_known_vehicle_in_front(
                    self.obstacle_slow_distance + 2.0
                ),
            },

            "vehicles": {
                "count": len(self.other_vehicles),
                "stale_timeout_sec": self.vehicle_state_stale_timeout_sec,
                "nearby": self._debug_nearby_vehicles(now_sec),
            },

            "watchdogs": {
                "last_progress_distance": self.last_progress_distance,
                "seconds_since_progress": self._seconds_since_ros_time(self.last_progress_time),
                "last_motion_x": self.last_motion_x,
                "last_motion_y": self.last_motion_y,
                "seconds_since_motion": self._seconds_since_ros_time(self.last_motion_time),
                "last_recovery_time_age": now_sec - self.last_recovery_time if self.last_recovery_time else None,
                "last_vehicle_deadlock_recovery_age": (
                    now_sec - self.last_vehicle_deadlock_recovery_time
                    if self.last_vehicle_deadlock_recovery_time
                    else None
                ),
            },

            "blocked_edges": self._debug_expiring_dict(self.blocked_edges, now_sec),

            "parameters": {
                "default_max_speed": self.default_max_speed,
                "linear_k": self.linear_k,
                "angular_k": self.angular_k,
                "max_angular_speed": self.max_angular_speed,
                "lane_width": self.lane_width,
                "vehicle_stop_distance": self.vehicle_stop_distance,
                "vehicle_slow_distance": self.vehicle_slow_distance,
                "vehicle_corridor_width": self.vehicle_corridor_width,
                "vehicle_headon_warn_distance": self.vehicle_headon_warn_distance,
                "vehicle_headon_stop_distance": self.vehicle_headon_stop_distance,
                "vehicle_headon_side_corridor": self.vehicle_headon_side_corridor,
                "stuck_timeout_sec": self.stuck_timeout_sec,
                "stuck_progress_epsilon": self.stuck_progress_epsilon,
            },
        }


    def _debug_waypoint(self, wp):
        if wp is None:
            return None

        return {
            "kind": wp.get("kind"),
            "x": wp.get("x"),
            "y": wp.get("y"),
            "edge_id": wp.get("edge_id"),
            "node_id": wp.get("node_id"),
            "from_node_id": wp.get("from_node_id"),
            "to_node_id": wp.get("to_node_id"),
            "speed_limit": wp.get("speed_limit"),
        }


    def _debug_current_traffic_light(self, current_wp, distance_to_wp):
        if current_wp is None:
            return None

        if current_wp.get("kind") != "approach_intersection":
            return {
                "relevant": False,
                "reason": "current waypoint is not approach_intersection",
                "wp_kind": current_wp.get("kind"),
            }

        intersection_node_id = current_wp.get("node_id")

        if intersection_node_id not in self.traffic_light_node_ids:
            return {
                "relevant": False,
                "reason": "intersection is not traffic-light controlled",
                "intersection_node_id": intersection_node_id,
                "wp_kind": current_wp.get("kind"),
            }

        from_node_id = current_wp.get("from_node_id")
        to_node_id = current_wp.get("to_node_id")

        status = self.traffic_light_statuses.get(intersection_node_id)

        if status is None:
            color = "unknown"
        else:
            color = self.get_signal_color_for_branch(
                intersection_node_id,
                from_node_id,
                to_node_id
            )

        return {
            "relevant": True,
            "intersection_node_id": intersection_node_id,
            "from_node_id": from_node_id,
            "to_node_id": to_node_id,
            "distance_to_wp": distance_to_wp,
            "inside_stop_distance": (
                distance_to_wp is not None
                and distance_to_wp <= self.traffic_light_stop_distance
            ),
            "color": color,
            "phase": status.get("phase") if status else None,
            "green_axis": status.get("green_axis") if status else None,
            "yellow_axis": status.get("yellow_axis") if status else None,
            "signal_states": status.get("signal_states", []) if status else [],
        }


    def _debug_nearby_vehicles(self, now_sec):
        if not self.has_odom:
            return []

        forward_x = math.cos(self.current_yaw)
        forward_y = math.sin(self.current_yaw)
        right_x = math.sin(self.current_yaw)
        right_y = -math.cos(self.current_yaw)

        rows = []

        for vehicle_id, other in list(self.other_vehicles.items()):
            stamp = float(other.get("stamp", 0.0))
            age = now_sec - stamp

            other_x = float(other.get("x", 0.0))
            other_y = float(other.get("y", 0.0))
            other_yaw = float(other.get("yaw", 0.0))

            dx = other_x - self.current_x
            dy = other_y - self.current_y

            euclidean_dist = math.sqrt(dx * dx + dy * dy)
            forward_dist = dx * forward_x + dy * forward_y
            side_dist = dx * right_x + dy * right_y

            heading_diff = abs(self.normalize_angle(other_yaw - self.current_yaw))

            same_direction = heading_diff < math.radians(45)
            opposite_direction = heading_diff > math.radians(135)
            crossing_direction = not same_direction and not opposite_direction

            rows.append({
                "vehicle_id": vehicle_id,
                "age_sec": age,
                "stale": age > self.vehicle_state_stale_timeout_sec,
                "x": other_x,
                "y": other_y,
                "yaw": other_yaw,
                "euclidean_dist": euclidean_dist,
                "forward_dist": forward_dist,
                "side_dist": side_dist,
                "heading_diff_deg": math.degrees(heading_diff),
                "same_direction": same_direction,
                "opposite_direction": opposite_direction,
                "crossing_direction": crossing_direction,
                "i_must_yield": self.should_yield_to_vehicle(vehicle_id),
            })

        rows.sort(key=lambda r: r["euclidean_dist"])
        return rows


    def _debug_expiring_dict(self, values, now_sec):
        result = {}

        for key, expire_at in values.items():
            try:
                expire_at = float(expire_at)
                result[key] = {
                    "expire_at": expire_at,
                    "expires_in": expire_at - now_sec,
                    "expired": expire_at <= now_sec,
                }
            except Exception:
                result[key] = str(expire_at)

        return result


    def _seconds_since_ros_time(self, ros_time):
        if ros_time is None:
            return None

        now = self.get_clock().now()
        return (now - ros_time).nanoseconds / 1e9


    def _json_safe(self, value):
        try:
            json.dumps(value)
            return value
        except TypeError:
            return str(value)

    def compute_remaining_distance(self):
        if not self.current_path:
            return 0.0
        if self.current_waypoint_index >= len(self.current_path):
            return 0.0

        total = 0.0
        current = {"x": self.current_x, "y": self.current_y}

        for i in range(self.current_waypoint_index, len(self.current_path)):
            wp = self.current_path[i]
            total += self.distance_xy(current["x"], current["y"], wp["x"], wp["y"])
            current = wp

        return total

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def distance_xy(self, x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        return math.sqrt(dx * dx + dy * dy)

    def clamp(self, value, min_value, max_value):
        return max(min_value, min(max_value, value))


def main(args=None):
    rclpy.init(args=args)

    node = NavigationExecutor()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    node.stop_vehicle()
    executor.shutdown()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
