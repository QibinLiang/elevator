# Introduction
This project is the source code of the assignment of Algorithm Theory(MUST DMIE07 E1-1). This project aims to figure out algorithms that schedule the elevator efficiently, including FCFS, SCAN and LookAhead.

# Authors
Qibin Liang <physechan@gmail.com>

Ning Sun <sunning1888@gmail.com>

ShuangShuang Zou <547685355@qq.com>

# Usage

## eval.py
- Default strategy SCAN: 
  ```bash
  python eval.py
  ```
- Specify parameters:
  ```bash 
  python eval.py --scheduler fcfs \
  --n-floors 30 --n-elevators 3 \ 
  --elevator-capacity 12 \
  --n-requests 80 --req-max-time 120 \
  --ele-max-time 600 --experiments 50 \ 
  --random-seed 42 \
  --request-distribution morning_peak  # uniform|morning_peak|evening_peak
  ```
- Available schedulers: `scan` (默认), `fcfs`, `lah`（最少停站，评分公式 `score = alpha*dist_start + beta*dist_end - gamma*same_dir`，可设置 lookahead 窗口 N 用于延迟分配、等待未来请求加入比较）
- Request 分布：`uniform`（默认），`morning_peak`（大多数从大堂出发，向上均匀分布），`evening_peak`（起点均匀，大多数回到大堂）；当前用 1 层代表大堂

## Grid Search & Visualization（grid_search.py）
- Default setting:
  ```bash
  python grid_search.py --experiments 1 --max-configs 1
  ```
- Specify the number of experiments：
  ```bash
  python grid_search.py --experiments 50
  ```
- Specify grid_search parameters：
  ```bash  
  python grid_search.py --schedulers fcfs,scan --n-floors-list 10,20,30 \
   --n-elevators-list 1,2,4 --elevator-capacity-list 6,10 \
   --n-requests-list 40,80 --req-max-time-list 100 \
   --ele-max-time-list 1500 --experiments 20
  ```

Notice：
- Results：`results/grid_search_results_*.json|csv`
- Data visualizations：`results/plots/`

## License
MIT License (see LICENSE).
