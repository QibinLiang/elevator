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

class SCANScheduler(BaseScheduler):
    # Elevator SCAN Scheduler, moves in one direction fulfilling requests until the end, then reverses direction
    def assign_requests(self, current_time):
        while self.request_queue != []:
            request = self.request_queue[0]
            # Find the first available elevator
            unfulfilled_elevators = [elevator for elevator in self.elevators if len(elevator.assigned_requests) < elevator.capacity]
            if not unfulfilled_elevators:
                logger.debug(f"Current Time: {current_time}, No available elevators for request {request.id}, waiting")
                break  # No available elevators, wait for the next time step
            else:
                logger.debug(f"Current Time: {current_time}, Assigning request {request.id} from queue")
                # Assign to the elevator that will pass by the request's start floor soonest
                best_elevator = None
                best_distance = float('inf')
                for elevator in unfulfilled_elevators:
                    if elevator.direction == 1 and elevator.current_floor <= request.start:
                        distance = request.start - elevator.current_floor
                    elif elevator.direction == -1 and elevator.current_floor >= request.start:
                        distance = elevator.current_floor - request.start
                    elif elevator.direction == 0:
                        distance = abs(elevator.current_floor - request.start)
                    else:
                        distance = float('inf')  # Elevator moving away from request
                    if distance < best_distance:
                        best_distance = distance
                        best_elevator = elevator
                
                if best_elevator:
                    best_elevator.add_request(request)
                    self.request_queue.pop(0)
                else:
                    logger.debug(f"Current Time: {current_time}, No suitable elevator found for request {request.id}, waiting")
                    break  # No suitable elevator, wait for the next time step
            # keep elevator moving in the same direction as the first request
            for elevator in self.elevators:
                if elevator.assigned_requests:
                    if elevator.assigned_requests[0].end >= elevator.current_floor:
                        elevator.direction = 1
                    else:
                        elevator.direction = -1
                else:
                    elevator.direction = 0
            
