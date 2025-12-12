# SPDX-License-Identifier: MIT
# Authors: 
# Qibin Liang <physechan@gmail.com>
# Ning Sun <sunning1888@gmail.com>
# ShuangShuang Zou <547685355@qq.com>
# Organization: Algorithm theory assignment
# Date: 2025-12-12
# License: MIT

import argparse
import csv
import itertools
import json
import os
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING

from src.strategies import *
from src.core import Evaluator
from src.logger import get_logger, set_logger_level

set_logger_level("INFO")
logger = get_logger()

RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
DEFAULT_EXPERIMENTS = 50
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PLOTS_DIR.resolve()))

if TYPE_CHECKING:  # hint for type checkers; keeps import after MPLCONFIGDIR is set
    import matplotlib.pyplot as plt  # pragma: no cover
else:
    import matplotlib.pyplot as plt

DEFAULT_PARAM_GRID = {
    "n_floors": [20, 30, 40, 50],
    "n_elevators": [1, 2, 3, 5, 8],
    "elevator_capacity": [5, 10, 15, 20],
    "n_requests": [20, 40, 60, 80, 100],
    "req_max_time": [100],
    "ele_max_time": [1500],
}

DEFAULT_SCHEDULERS = {
    "fcfs": FCFSScheduler,
    "scan": SCANScheduler,
}


def expand_param_grid(param_grid):
    keys = list(param_grid.keys())
    for combo in itertools.product(*(param_grid[key] for key in keys)):
        yield dict(zip(keys, combo))


def parse_int_list(arg, default):
    if arg is None:
        return default
    if isinstance(arg, list):
        return arg
    return [int(x) for x in str(arg).split(",") if str(x).strip()]


def resolve_schedulers(names):
    if not names:
        return list(DEFAULT_SCHEDULERS.values())
    scheds = []
    for name in names.split(","):
        key = name.strip().lower()
        if key not in DEFAULT_SCHEDULERS:
            raise ValueError(f"Unknown scheduler '{name}'. Options: {list(DEFAULT_SCHEDULERS.keys())}")
        scheds.append(DEFAULT_SCHEDULERS[key])
    return scheds


def build_param_grid(args):
    return {
        "n_floors": parse_int_list(args.n_floors_list, DEFAULT_PARAM_GRID["n_floors"]),
        "n_elevators": parse_int_list(args.n_elevators_list, DEFAULT_PARAM_GRID["n_elevators"]),
        "elevator_capacity": parse_int_list(args.elevator_capacity_list, DEFAULT_PARAM_GRID["elevator_capacity"]),
        "n_requests": parse_int_list(args.n_requests_list, DEFAULT_PARAM_GRID["n_requests"]),
        "req_max_time": parse_int_list(args.req_max_time_list, DEFAULT_PARAM_GRID["req_max_time"]),
        "ele_max_time": parse_int_list(args.ele_max_time_list, DEFAULT_PARAM_GRID["ele_max_time"]),
    }


def run_single_configuration(config, scheduler_cls, n_exps):
    logger.info(
        "Starting evaluation with parameters: %s",
        {**config, "scheduler": scheduler_cls.__name__},
    )
    evaluator = Evaluator(
        n_floors=config["n_floors"],
        n_elevators=config["n_elevators"],
        elevator_capacity=config["elevator_capacity"],
        n_requests=config["n_requests"],
        random_seed=None,
        ele_max_time=config["ele_max_time"],
        req_max_time=config["req_max_time"],
        smoothing_load=False,
        scheduler_cls=scheduler_cls,
    )
    metrics = evaluator.eval_n_experiments(n_exps=n_exps)
    completion_rate = (
        metrics["completed_requests"] / metrics["total_requests"]
        if metrics["total_requests"]
        else 0
    )
    return {
        **config,
        "scheduler": scheduler_cls.__name__,
        **metrics,
        "completion_rate": completion_rate,
    }


def run_grid_search(n_exps, param_grid, scheduler_classes, max_configs=None):
    results = []
    for idx, config in enumerate(expand_param_grid(param_grid), start=1):
        if max_configs is not None and idx > max_configs:
            break
        for scheduler_cls in scheduler_classes:
            result = run_single_configuration(config, scheduler_cls, n_exps)
            results.append(result)
    return results


def save_results(results, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = output_dir / f"grid_search_results_{timestamp}.json"
    csv_path = output_dir / f"grid_search_results_{timestamp}.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    field_order = [
        "scheduler",
        "n_floors",
        "n_elevators",
        "elevator_capacity",
        "n_requests",
        "req_max_time",
        "ele_max_time",
        "total_requests",
        "completed_requests",
        "completion_rate",
        "avg_wait_time",
        "avg_trip_time",
        "avg_total_time",
        "avg_elevator_distance",
        "avg_elevator_stops",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_order)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key, "") for key in field_order})

    logger.info("Results saved to %s and %s", csv_path, json_path)
    return json_path, csv_path


def aggregate_metric(results, group_key, metric):
    grouped = {}
    for row in results:
        grouped.setdefault(row[group_key], []).append(row[metric])
    return {key: mean(values) for key, values in grouped.items()}


def plot_metric_vs_param(results, metric, param_name, plots_dir, scheduler_classes):
    """保留基础折线图，用于单指标对比。"""
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for scheduler_cls in scheduler_classes:
        scheduler_results = [
            row for row in results if row["scheduler"] == scheduler_cls.__name__
        ]
        aggregated = aggregate_metric(scheduler_results, param_name, metric)
        x_vals = sorted(aggregated.keys())
        y_vals = [aggregated[x] for x in x_vals]
        plt.plot(x_vals, y_vals, marker="o", label=scheduler_cls.__name__)

    plt.xlabel(param_name)
    plt.ylabel(metric)
    plt.title(f"{metric} vs {param_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plot_path = plots_dir / f"{metric}_vs_{param_name}.png"
    plt.savefig(plot_path)
    plt.close()
    logger.info("Saved plot: %s", plot_path)
    return plot_path


def plot_param_dashboard(results, plots_dir, scheduler_classes):
    """Combined dashboard with shared legend to reduce duplicated legends."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    configs = [
        ("avg_total_time", "n_elevators", "Avg total time vs elevators"),
        ("avg_wait_time", "elevator_capacity", "Avg wait time vs capacity"),
        ("completion_rate", "n_floors", "Completion rate vs floors"),
    ]
    fig, axes = plt.subplots(len(configs), 1, figsize=(8, 12), sharex=False)
    scheduler_colors = {
        "FCFSScheduler": "#4c72b0",
        "SCANScheduler": "#c44e52",
    }
    handles = {}
    for ax, (metric, param_name, title) in zip(axes, configs):
        for scheduler_cls in scheduler_classes:
            name = scheduler_cls.__name__
            scheduler_results = [r for r in results if r["scheduler"] == name]
            aggregated = aggregate_metric(scheduler_results, param_name, metric)
            x_vals = sorted(aggregated.keys())
            y_vals = [aggregated[x] for x in x_vals]
            (line,) = ax.plot(
                x_vals,
                y_vals,
                marker="o",
                label=name,
                color=scheduler_colors.get(name),
            )
            handles[name] = line
        ax.set_title(title)
        ax.set_xlabel(param_name)
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)

    fig.legend(
        handles.values(),
        handles.keys(),
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        title="Schedulers",
        bbox_to_anchor=(0.5, 0.98),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = plots_dir / "dashboard_multi_metric.png"
    plt.savefig(plot_path)
    plt.close(fig)
    logger.info("Saved plot: %s", plot_path)
    return plot_path


def plot_scheduler_combo_bar(results, metrics, plots_dir, scheduler_classes):
    """Stack multiple metrics in a dual-axis bar chart with value labels."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    scheduler_names = [cls.__name__ for cls in scheduler_classes]
    x = range(len(scheduler_names))
    colors = ["#4c72b0", "#c44e52", "#55a868", "#8172b3", "#64b5cd"]

    primary_metrics = []
    secondary_metrics = []
    for metric in metrics:
        aggregated = [
            mean([row[metric] for row in results if row["scheduler"] == name])
            for name in scheduler_names
        ]
        max_val = max(aggregated) if aggregated else 0
        if "rate" in metric or max_val <= 1:
            secondary_metrics.append((metric, aggregated))
        else:
            primary_metrics.append((metric, aggregated))

    fig, ax_left = plt.subplots(figsize=(9, 5))
    ax_right = ax_left.twinx() if secondary_metrics else None

    def plot_group(ax, metric_list, side, offset_bias):
        handles, labels = [], []
        count = max(1, len(metric_list))
        bar_width = 0.35 if len(primary_metrics) and len(secondary_metrics) else 0.8 / count
        for i, (metric, aggregated) in enumerate(metric_list):
            offsets = [xi + (i - (count - 1) / 2) * bar_width for xi in x]
            bars = ax.bar(
                offsets,
                aggregated,
                width=bar_width,
                color=colors[(i + offset_bias) % len(colors)],
                label=metric,
                alpha=0.9,
            )
            ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
            handles.append(bars[0])
            labels.append(metric)
        return handles, labels

    left_handles, left_labels = plot_group(ax_left, primary_metrics, side="left", offset_bias=0)
    right_handles, right_labels = ([], [])
    if secondary_metrics and ax_right:
        right_handles, right_labels = plot_group(ax_right, secondary_metrics, side="right", offset_bias=2)

    ax_left.set_xticks(list(x))
    ax_left.set_xticklabels(scheduler_names)
    ax_left.set_ylabel("Time / count")
    if ax_right:
        ax_right.set_ylabel("Rate")
    ax_left.set_title("Scheduler multi-metric comparison (dual axes)")
    ax_left.grid(alpha=0.2, axis="y")

    handles = left_handles + right_handles
    labels = left_labels + right_labels
    ax_left.legend(handles, labels, title="Metrics", ncol=len(labels) if labels else 1, loc="upper left")

    fig.tight_layout()
    plot_path = plots_dir / "scheduler_multi_metric_bar.png"
    plt.savefig(plot_path)
    plt.close(fig)
    logger.info("Saved plot: %s", plot_path)
    return plot_path


def plot_param_heatmap(results, metric, x_param, y_param, plots_dir, scheduler_classes):
    """Highlight comparison across two parameters with scheduler side-by-side heatmaps."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    x_vals = sorted({row[x_param] for row in results})
    y_vals = sorted({row[y_param] for row in results})

    fig, axes = plt.subplots(1, len(scheduler_classes), figsize=(6 * len(scheduler_classes), 5), sharey=True)
    if len(scheduler_classes) == 1:
        axes = [axes]

    for ax, scheduler_cls in zip(axes, scheduler_classes):
        name = scheduler_cls.__name__
        scheduler_results = [r for r in results if r["scheduler"] == name]
        matrix = []
        for y in y_vals:
            row_vals = []
            for x in x_vals:
                filtered = [
                    r for r in scheduler_results if r[x_param] == x and r[y_param] == y
                ]
                val = mean([r[metric] for r in filtered]) if filtered else float("nan")
                row_vals.append(val)
            matrix.append(row_vals)

        im = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels(x_vals)
        ax.set_yticks(range(len(y_vals)))
        ax.set_yticklabels(y_vals)
        ax.set_xlabel(x_param)
        ax.set_title(f"{name} - {metric}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=metric)
    axes[0].set_ylabel(y_param)
    fig.tight_layout()
    plot_path = plots_dir / f"heatmap_{metric}_{x_param}_vs_{y_param}.png"
    plt.savefig(plot_path)
    plt.close(fig)
    logger.info("Saved plot: %s", plot_path)
    return plot_path


def generate_all_plots(results, plots_dir, scheduler_classes):
    # 兼容：保留基础单图，也生成合并图例的仪表盘
    plot_metric_vs_param(results, "avg_total_time", "n_elevators", plots_dir, scheduler_classes)
    plot_metric_vs_param(results, "avg_wait_time", "elevator_capacity", plots_dir, scheduler_classes)
    plot_metric_vs_param(results, "completion_rate", "n_floors", plots_dir, scheduler_classes)
    plot_metric_vs_param(results, "avg_elevator_distance", "n_elevators", plots_dir, scheduler_classes)
    plot_metric_vs_param(results, "avg_elevator_stops", "n_elevators", plots_dir, scheduler_classes)
    plot_param_dashboard(results, plots_dir, scheduler_classes)
    plot_scheduler_combo_bar(
        results,
        metrics=["avg_wait_time", "avg_total_time", "completion_rate"],
        plots_dir=plots_dir,
        scheduler_classes=scheduler_classes,
    )
    plot_param_heatmap(
        results,
        metric="avg_total_time",
        x_param="n_elevators",
        y_param="elevator_capacity",
        plots_dir=plots_dir,
        scheduler_classes=scheduler_classes,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run elevator grid search and plot results."
    )
    parser.add_argument(
        "--experiments",
        type=int,
        default=DEFAULT_EXPERIMENTS,
        help="Number of experiments to average for each parameter combination.",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Optional cap on number of parameter combinations (useful for quick smoke tests).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory to save raw results.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=PLOTS_DIR,
        help="Directory to save plots.",
    )
    parser.add_argument(
        "--schedulers",
        type=str,
        default=None,
        help="Comma-separated scheduler names to include (fcfs, scan). Default: both.",
    )
    parser.add_argument(
        "--n-floors-list",
        type=str,
        default=None,
        help="Comma-separated floors values, e.g., 10,20,30.",
    )
    parser.add_argument(
        "--n-elevators-list",
        type=str,
        default=None,
        help="Comma-separated elevator counts, e.g., 1,2,4.",
    )
    parser.add_argument(
        "--elevator-capacity-list",
        type=str,
        default=None,
        help="Comma-separated capacities, e.g., 6,10,14.",
    )
    parser.add_argument(
        "--n-requests-list",
        type=str,
        default=None,
        help="Comma-separated request counts, e.g., 20,40,80.",
    )
    parser.add_argument(
        "--req-max-time-list",
        type=str,
        default=None,
        help="Comma-separated request max times, e.g., 100,200.",
    )
    parser.add_argument(
        "--ele-max-time-list",
        type=str,
        default=None,
        help="Comma-separated elevator simulation max times, e.g., 1000,1500.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info(
        "Grid search starting with %d experiments per configuration", args.experiments
    )
    scheduler_classes = resolve_schedulers(args.schedulers)
    param_grid = build_param_grid(args)
    search_results = run_grid_search(
        n_exps=args.experiments,
        param_grid=param_grid,
        scheduler_classes=scheduler_classes,
        max_configs=args.max_configs,
    )
    if not search_results:
        logger.warning("No results produced; check parameter grid or limits.")
        raise SystemExit(1)

    save_results(search_results, args.results_dir)
    generate_all_plots(search_results, args.plots_dir, scheduler_classes)
    logger.info("Grid search complete.")
