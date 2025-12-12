# SPDX-License-Identifier: MIT
# Authors: 
# Qibin Liang <physechan@gmail.com>
# Ning Sun <sunning1888@gmail.com>
# ShuangShuang Zou <547685355@qq.com>
# Organization: Algorithm theory assignment
# Date: 2025-12-12
# License: MIT

from src.core import BaseScheduler
from src.logger import get_logger

logger = get_logger()


class LAHScheduler(BaseScheduler):
    """Least Stops First: SCAN with scoring and a lookahead delay window."""

    def __init__(
        self,
        elevators,
        smoothing_load=False,
        alpha=1.0,
        beta=2.0,
        gamma=5.0,
        lookahead_window=0,
    ):
        super().__init__(elevators, smoothing_load=smoothing_load)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lookahead_window = lookahead_window  # N

    def _closest_start_distance(self, elevator, request):
        if elevator.assigned_requests:
            return min(abs(r.start - request.start) for r in elevator.assigned_requests)
        return abs(elevator.current_floor - request.start)

    def _closest_end_distance(self, elevator, request):
        if elevator.assigned_requests:
            return min(abs(r.end - request.end) for r in elevator.assigned_requests)
        return abs(elevator.current_floor - request.end)

    def _direction_match(self, elevator, request):
        if elevator.direction == 0:
            return 0
        req_dir = 1 if request.end > request.start else -1
        # if directions match and request is on the way
        if elevator.direction == req_dir:
            if (elevator.direction == 1 and request.start >= elevator.current_floor) or \
                (elevator.direction == -1 and request.start <= elevator.current_floor):
                return 1
        return -1

    def _score(self, elevator, request):
        dist_start = self._closest_start_distance(elevator, request)
        dist_end = self._closest_end_distance(elevator, request)
        same_dir = self._direction_match(elevator, request)
        return self.alpha * dist_start + self.beta * dist_end - self.gamma * same_dir

    def assign_requests(self, current_time):
        if not self.request_queue:
            return

        # Wait until the earliest request has stayed in the queue for the full
        # lookahead window, so we can batch together requests arriving in
        # [t, t + lookahead_window].
        oldest_arrival = min(req.arrival_time for req in self.request_queue)
        if current_time < oldest_arrival + self.lookahead_window:
            return

        while True:
            if not self.request_queue:
                break

            best_pair = None  # (score, elevator, request)
            for request in self.request_queue:
                candidates = [ele for ele in self.elevators if len(ele.assigned_requests) < ele.capacity]
                for elevator in candidates:
                    score = self._score(elevator, request)
                    if best_pair is None or score < best_pair[0]:
                        best_pair = (score, elevator, request)

            if best_pair is None:
                logger.debug(
                    "Current Time: %s, No available elevators for eligible requests, waiting",
                    current_time,
                )
                break

            score, chosen_elevator, chosen_request = best_pair
            chosen_elevator.add_request(chosen_request)
            self.request_queue.remove(chosen_request)
            logger.debug(
                "Current Time: %s, Assigning request %s to Elevator %s (score=%.2f)",
                current_time,
                chosen_request.id,
                chosen_elevator.elevator_id,
                score,
            )

            # keep elevator moving in the same direction as the first request
            for elevator in self.elevators:
                if elevator.assigned_requests:
                    if elevator.assigned_requests[0].end >= elevator.current_floor:
                        elevator.direction = 1
                    else:
                        elevator.direction = -1
                else:
                    elevator.direction = 0
