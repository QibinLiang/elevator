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

class FCFSScheduler(BaseScheduler):
    # First-Come, First-Served Scheduler, assigns requests to elevators in the order they arrive
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
                if self.smoothing_load:
                    # Assign to the elevator with the least load
                    unfulfilled_elevators.sort(key=lambda ele: len(ele.assigned_requests))
                    unfulfilled_elevators[0].add_request(request)
                    self.request_queue.pop(0)
                else:
                    unfulfilled_elevators[0].add_request(request)
                    self.request_queue.pop(0)
            # keep elevator moving in the same direction as the first request
            for elevator in self.elevators:
                if elevator.assigned_requests:
                    if elevator.assigned_requests[0].end >= elevator.current_floor:
                        elevator.direction = 1
                    else:
                        elevator.direction = -1
                else:
                    elevator.direction = 0
                
                
                    
