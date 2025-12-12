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
from src.strategies import FCFSScheduler, SCANScheduler

set_logger_level("INFO")
logger = get_logger()

SCHEDULERS = {
    "fcfs": FCFSScheduler,
    "scan": SCANScheduler,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run a single elevator simulation.")
    parser.add_argument("--scheduler", type=str, default="scan", choices=SCHEDULERS.keys(), help="调度策略")
    parser.add_argument("--n-floors", type=int, default=20, help="楼层数")
    parser.add_argument("--n-elevators", type=int, default=1, help="电梯数量")
    parser.add_argument("--elevator-capacity", type=int, default=10, help="电梯容量")
    parser.add_argument("--n-requests", type=int, default=50, help="乘梯请求数量")
    parser.add_argument("--req-max-time", type=int, default=100, help="请求到达的最大时间戳")
    parser.add_argument("--ele-max-time", type=int, default=300, help="电梯模拟的最大时间戳")
    parser.add_argument("--experiments", type=int, default=100, help="重复实验次数，取平均指标")
    parser.add_argument("--random-seed", type=int, default=None, help="随机种子；缺省为随机")
    parser.add_argument("--smoothing-load", action="store_true", help="是否平滑分配负载（FCFS 可选）")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
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
    )

    results = evaluator.eval_n_experiments(n_exps=args.experiments)
    print(f"Evaluation Results over {args.experiments} experiments with {args.scheduler}:")
    for key, value in results.items():
        print(f"{key}: {value}")
