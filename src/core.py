# SPDX-License-Identifier: MIT
# Authors: 
# Qibin Liang <physechan@gmail.com>
# Ning Sun <sunning1888@gmail.com>
# ShuangShuang Zou <547685355@qq.com>
# Organization: Algorithm theory assignment
# Date: 2025-12-12
# License: MIT

import tqdm
import numpy as np
from src.logger import get_logger

logger = get_logger()

class Timer:
    def __init__(self, max_time):
        self.current_time = 0
        self.max_time = max_time
        
    def __next__(self):
        if self.current_time < self.max_time:
            self.current_time += 1
            return self.current_time
        else:
            raise StopIteration
    
    def __iter__(self):
        return self

# singleton ID generator for unique request IDs
class IDGenerator:
    _instance = None
    _id_counter = 0

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(IDGenerator, cls).__new__(cls)
        return cls._instance

    def get_next_id(self):
        self._id_counter += 1
        return self._id_counter

class EleRequest:
    def __init__(self,
                 start=None,
                 end=None,
                 arrival_time=None):
        self.id = IDGenerator().get_next_id()
        self.start = start
        self.end = end
        self.arrival_time = arrival_time
        self.assigned_elevator = None
        self.pickup_time = None
        self.dropoff_time = None
        
    def __repr__(self):
        return f"EleRequest(id={self.id}, start={self.start}, end={self.end}, arrival_time={self.arrival_time}, assigned_elevator={self.assigned_elevator}, pickup_time={self.pickup_time}, dropoff_time={self.dropoff_time})"
    
    def pickup(self, current_time, elevator_id):
        self.pickup_time = current_time
        self.assigned_elevator = elevator_id
        
    def dropoff(self, current_time):
        self.dropoff_time = current_time
    
def create_requests(random_seed, num_requests, num_floors, req_max_time):
    if random_seed is not None:
        np.random.seed(random_seed)
    requests = []
    for _ in range(num_requests):
        start = np.random.randint(1, num_floors + 1)
        end = np.random.randint(1, num_floors + 1)
        while end == start:
            end = np.random.randint(1, num_floors + 1)
        arrival_time = np.random.randint(0, req_max_time)
        req = EleRequest(start=start, end=end, arrival_time=arrival_time)
        requests.append(req)
    requests.sort(key=lambda r: r.arrival_time)
    return requests

class Elevator:
    def __init__(self, elevator_id, current_floor=1, capacity=10, n_floors=100):
        self.elevator_id = elevator_id
        self.current_floor = current_floor
        self.direction = 0  # -1 for down, 0 for idle, 1 for up
        self.capacity = capacity
        self.current_load = 0
        self.assigned_requests = []
        self.n_floors = n_floors
        self.total_distance = 0
        self.stop_count = 0
        
    def __repr__(self):
        return f"Elevator(id={self.elevator_id}, current_floor={self.current_floor}, direction={self.direction}, capacity={self.capacity}, current_load={self.current_load}, assigned_requests={self.assigned_requests})"
    
    def dropoff_requests(self, current_time):
        drop_offs = [req for req in self.assigned_requests if req.end == self.current_floor and req.pickup_time is not None and req.dropoff_time is None]
        for req in drop_offs:
            logger.debug(f"Elevator {self.elevator_id} dropping off request {req.id} at floor {self.current_floor} at time {current_time}")
            self.assigned_requests.remove(req)
            req.dropoff(current_time=current_time)
            self.current_load -= 1
        return bool(drop_offs)
    
    def pickup_requests(self, current_time):
        pick_ups = [req for req in self.assigned_requests if req.start == self.current_floor and req.pickup_time is None]
        for req in pick_ups:
            logger.debug(f"Elevator {self.elevator_id} picking up request {req.id} at floor {self.current_floor} at time {current_time}")
            req.pickup(current_time=current_time, elevator_id=self.elevator_id)
            self.current_load += 1
        return bool(pick_ups)
    
    def add_request(self, request):
        self.assigned_requests.append(request)
    
    def move(self):
        prev_floor = self.current_floor
        self.current_floor += self.direction
        if self.current_floor <= 1 or self.current_floor >= self.n_floors:
            self.direction = self.direction * -1 if self.assigned_requests!= [] else 0  # switch direction
        self.current_floor = max(1, self.current_floor)  # Ensure floor doesn't go below 1
        self.current_floor = min(self.current_floor, self.n_floors)  # Assuming max floor is 100 for safety
        self.total_distance += abs(self.current_floor - prev_floor)
            

class BaseScheduler:
    def __init__(self, elevators, smoothing_load=False):
        self.elevators = elevators
        self.request_queue = []
        self.smoothing_load = smoothing_load # only for fcfs
        
    def add_request(self, request):
        self.request_queue.append(request)
        
    def assign_requests(self, current_time):
        raise NotImplementedError("This method should be overridden by subclasses")
    

class Evaluator:
    def __init__(self,
                n_floors,
                n_elevators,
                elevator_capacity,
                n_requests,
                random_seed,
                ele_max_time,
                req_max_time,
                scheduler_cls,
                smoothing_load,
                n_exps=1):
        self.n_floors = n_floors
        self.n_elevators = n_elevators
        self.elevator_capacity = elevator_capacity
        self.ele_max_time = ele_max_time
        self.elevators = [Elevator(elevator_id=i+1, capacity=elevator_capacity, n_floors=n_floors) for i in range(n_elevators)]
        self.timer = Timer(max_time=ele_max_time)
        self.random_seed = random_seed
        self.requests = create_requests(random_seed=random_seed,
                                        num_requests=n_requests,
                                        num_floors=n_floors,
                                        req_max_time=req_max_time)
        self.req_max_time = req_max_time
        self.smoothing_load = smoothing_load
        self.scheduler = scheduler_cls(self.elevators, smoothing_load=smoothing_load)
    
    def eval_n_experiments(self, n_exps): 
        all_results = []
        for exp in tqdm.tqdm(range(n_exps)):
            logger.debug(f"Starting experiment {exp+1}/{n_exps}")
            self.__init__(n_floors=self.n_floors,
                          n_elevators=self.n_elevators,
                          elevator_capacity=self.elevator_capacity,
                          n_requests=len(self.requests),
                          random_seed=self.random_seed,
                          ele_max_time=self.ele_max_time,
                          req_max_time=self.req_max_time,
                          scheduler_cls=type(self.scheduler),
                          smoothing_load=self.smoothing_load)
            self.eval()
            results = self.get_results()
            all_results.append(results)
            logger.debug(f"Completed experiment {exp+1}/{n_exps} with results: {results}")
        # avg results over all experiments
        avg_results = {
            "total_requests": np.mean([res["total_requests"] for res in all_results]),
            "completed_requests": np.mean([res["completed_requests"] for res in all_results]),
            "avg_wait_time": np.mean([res["avg_wait_time"] for res in all_results]),
            "avg_trip_time": np.mean([res["avg_trip_time"] for res in all_results]),
            "avg_total_time": np.mean([res["avg_total_time"] for res in all_results]),
            "avg_elevator_distance": np.mean([res["avg_elevator_distance"] for res in all_results]),
            "avg_elevator_stops": np.mean([res["avg_elevator_stops"] for res in all_results]),
        }
        return avg_results    
    
    def eval(self):
        request_index = 0
        for current_time in self.timer:
            logger.trace(f"current elevator positions: {[elevator.current_floor for elevator in self.elevators]}")
            logger.trace(f"num of assigned requests per elevator: {[len(elevator.assigned_requests) for elevator in self.elevators]}")
            while (request_index < len(self.requests) and 
                   self.requests[request_index].arrival_time <= current_time):
                self.scheduler.add_request(self.requests[request_index])
                request_index += 1
            self.scheduler.assign_requests(current_time=current_time)
            for elevator in self.elevators:
                dropped = elevator.dropoff_requests(current_time=current_time)
                picked = elevator.pickup_requests(current_time=current_time)
                if dropped or picked:
                    elevator.stop_count += 1
                elevator.move() 
    
    def get_results(self):
        total_wait_time = 0
        total_trip_time = 0
        total_time = 0
        completed_requests = [req for req in self.requests if req.dropoff_time is not None]
        for req in completed_requests:
            wait_time = req.pickup_time - req.arrival_time
            trip_time = req.dropoff_time - req.pickup_time
            total_wait_time += wait_time
            total_trip_time += trip_time
            total_time += (wait_time + trip_time)
            
        avg_wait_time = total_wait_time / len(completed_requests) if completed_requests else 0
        avg_trip_time = total_trip_time / len(completed_requests) if completed_requests else 0
        avg_total_time = total_time / len(completed_requests) if completed_requests else 0
        
        elevator_distances = [e.total_distance for e in self.elevators]
        elevator_stops = [e.stop_count for e in self.elevators]

        results = {
            "total_requests": len(self.requests),
            "completed_requests": len(completed_requests),
            "avg_wait_time": avg_wait_time,
            "avg_trip_time": avg_trip_time,
            "avg_total_time": avg_total_time,
            "per_elevator_distance": elevator_distances,
            "per_elevator_stops": elevator_stops,
            "avg_elevator_distance": np.mean(elevator_distances) if elevator_distances else 0,
            "avg_elevator_stops": np.mean(elevator_stops) if elevator_stops else 0,
        }
        return results
