# SPDX-License-Identifier: MIT
# Authors: 
# Qibin Liang <physechan@gmail.com>
# Ning Sun <sunning1888@gmail.com>
# ShuangShuang Zou <547685355@qq.com>
# Organization: Algorithm theory assignment
# Date: 2025-12-12
# License: MIT

import argparse

from src.core import Evaluator
from src.logger import get_logger, set_logger_level
from src.strategies import FCFSScheduler, LAHScheduler, SCANScheduler

logger = get_logger()


SCHEDULERS = {
    "fcfs": FCFSScheduler,
    "scan": SCANScheduler,
    "lah": LAHScheduler,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run a single elevator simulation.")
    parser.add_argument("--scheduler", type=str, default="scan", choices=SCHEDULERS.keys(), help="strategy to use")
    parser.add_argument("--n-floors", type=int, default=20, help="number of floors")
    parser.add_argument("--n-elevators", type=int, default=3, help="number of elevators")
    parser.add_argument("--elevator-capacity", type=int, default=10, help="elevator capacity")
    parser.add_argument("--n-requests", type=int, default=50, help="number of elevator requests")
    parser.add_argument("--req-max-time", type=int, default=100, help="maximum request arrival timestamp")
    parser.add_argument("--ele-max-time", type=int, default=300, help="maximum elevator simulation timestamp")
    parser.add_argument("--experiments", type=int, default=1, help="number of repeated experiments to average metrics")
    parser.add_argument("--random-seed", type=int, default=None, help="random seed; default is random")
    parser.add_argument("--smoothing-load", action="store_true", help="whether to smooth load distribution (optional for FCFS)")
    parser.add_argument("--verbose", action="store_true", help="enable verbose logging")
    parser.add_argument(
        "--request-distribution",
        type=str,
        default="uniform",
        choices=["uniform", "morning_peak", "evening_peak"],
        help="请求分布类型：uniform（默认），morning_peak（大多数从大堂出发），evening_peak（大多数回到大堂）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.verbose:
        set_logger_level("DEBUG")
    else:
        set_logger_level("INFO")
    scheduler_cls = SCHEDULERS[args.scheduler]
    evaluator = Evaluator(
        n_floors=args.n_floors,
        n_elevators=args.n_elevators,
        elevator_capacity=args.elevator_capacity,
        n_requests=args.n_requests,
        random_seed=args.random_seed,
        ele_max_time=args.ele_max_time,
        req_max_time=args.req_max_time,
        smoothing_load=args.smoothing_load,
        scheduler_cls=scheduler_cls,
        request_distribution=args.request_distribution,
    )

    results = evaluator.eval_n_experiments(n_exps=args.experiments)
    print(f"Evaluation Results over {args.experiments} experiments with {args.scheduler}:")
    for key, value in results.items():
        print(f"{key}: {value}")
