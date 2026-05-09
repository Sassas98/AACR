import json
import time
from enum import Enum

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from std_srvs.srv import Trigger


class ClaimState(str, Enum):
    IDLE = "IDLE"
    CLAIMING = "CLAIMING"


class TaxiCoordinator(Node):

    def __init__(self):
        super().__init__("taxi_coordinator")

        self.declare_parameter("taxi_id", "taxi_1")
        self.declare_parameter("claim_window_sec", 1.5)

        self.taxi_id = self.get_parameter("taxi_id").value
        self.claim_window_sec = float(
            self.get_parameter("claim_window_sec").value
        )

        self.state = ClaimState.IDLE

        self.pending_request = None
        self.claim_started_at = None

        self.claims_by_request = {}
        self.assigned_requests = set()
        self.ignored_requests = set()

        self.claims_pub = self.create_publisher(
            String,
            "/taxi_claims",
            10
        )

        self.assignment_pub = self.create_publisher(
            String,
            "/taxi_assignment",
            10
        )

        self.request_sub = self.create_subscription(
            String,
            "/gazebo/taxi_request",
            self.on_taxi_request,
            10
        )

        self.claims_sub = self.create_subscription(
            String,
            "/taxi_claims",
            self.on_taxi_claim,
            10
        )

        self.status_client = self.create_client(
            Trigger,
            "/taxi_status"
        )

        self.decision_timer = self.create_timer(
            0.2,
            self.decision_loop
        )

        self.get_logger().info(
            f"taxi_coordinator avviato per taxi_id={self.taxi_id}"
        )

    # ------------------------------------------------------------------
    # TAXI REQUEST DA GAZEBO
    # ------------------------------------------------------------------

    def on_taxi_request(self, msg):
        try:
            request = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn(f"Taxi request non JSON: {msg.data}")
            return

        required = [
            "request_id",
            "pickup_x",
            "pickup_y",
            "dropoff_x",
            "dropoff_y"
        ]

        if not all(k in request for k in required):
            self.get_logger().warn(f"Taxi request incompleta: {request}")
            return

        request_id = request["request_id"]

        if request_id in self.assigned_requests:
            return

        if request_id in self.ignored_requests:
            return

        if self.state != ClaimState.IDLE:
            return

        self.ask_status_and_maybe_claim(request)

    # ------------------------------------------------------------------
    # SERVICE /taxi_status
    # ------------------------------------------------------------------

    def ask_status_and_maybe_claim(self, request):
        if not self.status_client.wait_for_service(timeout_sec=0.2):
            self.get_logger().warn("/taxi_status non disponibile")
            return

        future = self.status_client.call_async(Trigger.Request())
        future.add_done_callback(
            lambda f: self.on_status_response(f, request)
        )

    def on_status_response(self, future, request):
        try:
            response = future.result()
        except Exception as ex:
            self.get_logger().warn(f"Errore chiamando /taxi_status: {ex}")
            return

        if not response.success:
            self.get_logger().info(
                f"{self.taxi_id}: taxi non disponibile, ignoro request {request['request_id']}"
            )
            self.ignored_requests.add(request["request_id"])
            return

        self.start_claim(request)

    # ------------------------------------------------------------------
    # CLAIM DISTRIBUITI
    # ------------------------------------------------------------------

    def start_claim(self, request):
        request_id = request["request_id"]

        self.pending_request = request
        self.claim_started_at = time.time()
        self.state = ClaimState.CLAIMING

        self.register_claim(
            request_id=request_id,
            taxi_id=self.taxi_id,
            request=request
        )

        payload = {
            "request_id": request_id,
            "taxi_id": self.taxi_id,
            "timestamp": time.time()
        }

        msg = String()
        msg.data = json.dumps(payload)

        self.claims_pub.publish(msg)

        self.get_logger().info(
            f"{self.taxi_id}: claim pubblicato per request {request_id}"
        )

    def on_taxi_claim(self, msg):
        try:
            claim = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        request_id = claim.get("request_id")
        taxi_id = claim.get("taxi_id")

        if request_id is None or taxi_id is None:
            return

        if taxi_id == self.taxi_id:
            return

        self.register_claim(
            request_id=request_id,
            taxi_id=taxi_id,
            request=None
        )

    def register_claim(self, request_id, taxi_id, request):
        if request_id not in self.claims_by_request:
            self.claims_by_request[request_id] = {
                "request": request,
                "claims": set(),
                "first_seen_at": time.time()
            }

        if request is not None:
            self.claims_by_request[request_id]["request"] = request

        self.claims_by_request[request_id]["claims"].add(taxi_id)

    # ------------------------------------------------------------------
    # DECISION LOOP
    # ------------------------------------------------------------------

    def decision_loop(self):
        if self.state != ClaimState.CLAIMING:
            return

        if self.pending_request is None:
            self.state = ClaimState.IDLE
            return

        elapsed = time.time() - self.claim_started_at

        if elapsed < self.claim_window_sec:
            return

        request_id = self.pending_request["request_id"]

        winner = self.compute_winner(request_id)

        if winner == self.taxi_id:
            self.publish_assignment(self.pending_request)
            self.assigned_requests.add(request_id)
        else:
            self.get_logger().info(
                f"{self.taxi_id}: request {request_id} persa, vincitore={winner}"
            )

        self.pending_request = None
        self.claim_started_at = None
        self.state = ClaimState.IDLE

    def compute_winner(self, request_id):
        info = self.claims_by_request.get(request_id)

        if info is None:
            return self.taxi_id

        claims = list(info["claims"])
        claims.sort()

        return claims[0]

    # ------------------------------------------------------------------
    # ASSIGNMENT VERSO TAXI_REQUEST_MANAGER
    # ------------------------------------------------------------------

    def publish_assignment(self, request):
        payload = {
            "request_id": request["request_id"],
            "taxi_id": self.taxi_id,
            "pickup_x": request["pickup_x"],
            "pickup_y": request["pickup_y"],
            "dropoff_x": request["dropoff_x"],
            "dropoff_y": request["dropoff_y"],
            "assigned_at": time.time()
        }

        msg = String()
        msg.data = json.dumps(payload)

        self.assignment_pub.publish(msg)

        self.get_logger().info(
            f"{self.taxi_id}: assignment pubblicato per request {request['request_id']}"
        )


def main(args=None):
    rclpy.init(args=args)

    node = TaxiCoordinator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()