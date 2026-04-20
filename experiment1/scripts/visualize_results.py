#!/usr/bin/env python3
"""Visualization utilities for experiment 1 results."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GROUP_ORDER = [
    "group_a_single_robot",
    "group_b_ordered_nearest",
    "group_c_distance_greedy",
    "group_d_full_method",
]

GROUP_LABELS = {
    "group_a_single_robot": "A Single Robot",
    "group_b_ordered_nearest": "B Ordered Nearest",
    "group_c_distance_greedy": "C Distance Greedy",
    "group_d_full_method": "D Full Method",
}

GROUP_COLORS = {
    "group_a_single_robot": "#d62728",
    "group_b_ordered_nearest": "#2ca02c",
    "group_c_distance_greedy": "#1f77b4",
    "group_d_full_method": "#ff7f0e",
}


@dataclass
class RunArtifacts:
    group_name: str
    run_name: str
    summary: Dict
    map_growth: List[Dict]
    frontier_series: List[Dict]


def resolve_package_root() -> Path:
    try:
        import rospkg  # type: ignore

        return Path(rospkg.RosPack().get_path("experiment1"))
    except Exception:
        return Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    package_root = resolve_package_root()
    parser = argparse.ArgumentParser(description="Visualize experiment1 result artifacts.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=package_root / "results",
        help="Directory containing group run folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where plots will be written. Defaults to results/figures/<timestamp>.",
    )
    parser.add_argument(
        "--mode",
        choices=["latest", "mean"],
        default="mean",
        help="Use the latest run per group or aggregate all runs per group.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Figure DPI.",
    )
    return parser


def read_csv_rows(path: Path) -> List[Dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_run(run_dir: Path, group_name: str) -> RunArtifacts:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    summary = json.loads(summary_path.read_text())
    return RunArtifacts(
        group_name=group_name,
        run_name=run_dir.name,
        summary=summary,
        map_growth=read_csv_rows(run_dir / "map_growth.csv"),
        frontier_series=read_csv_rows(run_dir / "frontier_series.csv"),
    )


def collect_runs(results_root: Path) -> Dict[str, List[RunArtifacts]]:
    grouped_runs: Dict[str, List[RunArtifacts]] = {}
    for group_name in GROUP_ORDER:
        group_dir = results_root / group_name
        if not group_dir.exists():
            continue
        run_dirs = sorted(
            [path for path in group_dir.iterdir() if path.is_dir() and (path / "summary.json").exists()],
            key=lambda path: path.name,
        )
        if not run_dirs:
            continue
        grouped_runs[group_name] = [load_run(run_dir, group_name) for run_dir in run_dirs]
    return grouped_runs


def summarize_metric(runs: Sequence[RunArtifacts], key: str) -> Dict[str, float]:
    values = [float(run.summary.get(key, 0.0)) for run in runs]
    return {
        "mean": float(statistics.mean(values)) if values else 0.0,
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "latest": float(values[-1]) if values else 0.0,
        "count": len(values),
    }


def _relative_time_and_values(rows: Sequence[Dict], time_key: str, value_key: str) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        return np.asarray([]), np.asarray([])
    times = np.asarray([float(row[time_key]) for row in rows], dtype=np.float64)
    values = np.asarray([float(row[value_key]) for row in rows], dtype=np.float64)
    return times - times[0], values


def build_group_curve_latest(runs: Sequence[RunArtifacts], series_name: str, value_key: str) -> tuple[np.ndarray, np.ndarray]:
    latest_run = runs[-1]
    series = latest_run.map_growth if series_name == "map_growth" else latest_run.frontier_series
    return _relative_time_and_values(series, "sim_time", value_key)


def build_group_curve_mean(runs: Sequence[RunArtifacts], series_name: str, value_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    curves = []
    max_duration = 0.0
    for run in runs:
        series = run.map_growth if series_name == "map_growth" else run.frontier_series
        rel_time, values = _relative_time_and_values(series, "sim_time", value_key)
        if rel_time.size == 0:
            continue
        max_duration = max(max_duration, float(rel_time[-1]))
        curves.append((rel_time, values))

    if not curves:
        return np.asarray([]), np.asarray([]), np.asarray([])

    grid = np.linspace(0.0, max_duration, num=250)
    interpolated = []
    for rel_time, values in curves:
        interp_values = np.interp(grid, rel_time, values, left=values[0], right=values[-1])
        interpolated.append(interp_values)

    stacked = np.vstack(interpolated)
    return grid, stacked.mean(axis=0), stacked.std(axis=0)


def ensure_output_dir(path: Optional[Path], results_root: Path) -> Path:
    if path is not None:
        output_dir = path.expanduser().resolve()
    else:
        output_dir = (results_root / "figures" / time.strftime("%Y%m%d_%H%M%S")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def configure_axes(ax, title: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.3)


def save_figure(fig, output_path: Path, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_summary_bars(grouped_runs: Dict[str, List[RunArtifacts]], output_dir: Path, mode: str, dpi: int) -> List[Path]:
    metric_specs = [
        ("exploration_completion_time_sec", "Completion Time", "Time (s)", "completion_time.png"),
        ("final_known_area_m2", "Final Known Area", "Area (m²)", "final_known_area.png"),
        ("total_path_length_m", "Total Path Length", "Length (m)", "total_path_length.png"),
        ("total_goals_failed", "Failed Goals", "Count", "failed_goals.png"),
    ]
    generated = []
    groups = [group for group in GROUP_ORDER if group in grouped_runs]
    labels = [GROUP_LABELS[group] for group in groups]
    colors = [GROUP_COLORS[group] for group in groups]

    for key, title, ylabel, filename in metric_specs:
        values = []
        errors = []
        for group in groups:
            stats = summarize_metric(grouped_runs[group], key)
            values.append(stats["latest"] if mode == "latest" else stats["mean"])
            errors.append(0.0 if mode == "latest" else stats["std"])

        fig, ax = plt.subplots(figsize=(8, 4.8))
        x_positions = np.arange(len(groups))
        ax.bar(x_positions, values, yerr=errors if mode == "mean" else None, color=colors, alpha=0.9, capsize=6)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=12, ha="right")
        configure_axes(ax, f"Experiment 1: {title}", ylabel)
        output_path = output_dir / filename
        save_figure(fig, output_path, dpi)
        generated.append(output_path)

    return generated


def plot_goal_outcomes(grouped_runs: Dict[str, List[RunArtifacts]], output_dir: Path, mode: str, dpi: int) -> Path:
    groups = [group for group in GROUP_ORDER if group in grouped_runs]
    labels = [GROUP_LABELS[group] for group in groups]

    success_values = []
    fail_values = []
    for group in groups:
        success_stats = summarize_metric(grouped_runs[group], "total_goals_succeeded")
        fail_stats = summarize_metric(grouped_runs[group], "total_goals_failed")
        if mode == "latest":
            success_values.append(success_stats["latest"])
            fail_values.append(fail_stats["latest"])
        else:
            success_values.append(success_stats["mean"])
            fail_values.append(fail_stats["mean"])

    fig, ax = plt.subplots(figsize=(8, 4.8))
    x_positions = np.arange(len(groups))
    ax.bar(x_positions, success_values, color="#2ca02c", label="Succeeded")
    ax.bar(x_positions, fail_values, bottom=success_values, color="#d62728", label="Failed")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    configure_axes(ax, "Experiment 1: Goal Outcomes", "Count")
    ax.legend()
    output_path = output_dir / "goal_outcomes.png"
    save_figure(fig, output_path, dpi)
    return output_path


def plot_path_lengths_per_robot(grouped_runs: Dict[str, List[RunArtifacts]], output_dir: Path, mode: str, dpi: int) -> Path:
    groups = [group for group in GROUP_ORDER if group in grouped_runs]
    robot_order = ["tb3_1", "tb3_2", "tb3_3"]
    labels = [GROUP_LABELS[group] for group in groups]

    width = 0.22
    x_positions = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(9, 5.2))

    for robot_index, robot_name in enumerate(robot_order):
        robot_values = []
        for group in groups:
            runs = grouped_runs[group]
            per_run_values = [
                float(run.summary.get("per_robot_path_length_m", {}).get(robot_name, 0.0))
                for run in runs
            ]
            if mode == "latest":
                robot_values.append(per_run_values[-1] if per_run_values else 0.0)
            else:
                robot_values.append(float(statistics.mean(per_run_values)) if per_run_values else 0.0)
        ax.bar(
            x_positions + (robot_index - 1) * width,
            robot_values,
            width=width,
            label=robot_name,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    configure_axes(ax, "Experiment 1: Path Length per Robot", "Length (m)")
    ax.legend()
    output_path = output_dir / "path_length_per_robot.png"
    save_figure(fig, output_path, dpi)
    return output_path


def plot_curve_comparison(
    grouped_runs: Dict[str, List[RunArtifacts]],
    output_dir: Path,
    mode: str,
    dpi: int,
    series_name: str,
    value_key: str,
    title: str,
    ylabel: str,
    filename: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(8.8, 5.0))

    for group in GROUP_ORDER:
        runs = grouped_runs.get(group)
        if not runs:
            continue
        color = GROUP_COLORS[group]
        label = GROUP_LABELS[group]

        if mode == "latest":
            rel_time, values = build_group_curve_latest(runs, series_name, value_key)
            if rel_time.size == 0:
                continue
            ax.plot(rel_time, values, label=label, color=color, linewidth=2.0)
        else:
            rel_time, mean_values, std_values = build_group_curve_mean(runs, series_name, value_key)
            if rel_time.size == 0:
                continue
            ax.plot(rel_time, mean_values, label=label, color=color, linewidth=2.0)
            ax.fill_between(rel_time, mean_values - std_values, mean_values + std_values, color=color, alpha=0.18)

    ax.set_xlabel("Exploration Time (s)")
    configure_axes(ax, title, ylabel)
    ax.legend()
    output_path = output_dir / filename
    save_figure(fig, output_path, dpi)
    return output_path


def write_comparison_summary_csv(grouped_runs: Dict[str, List[RunArtifacts]], output_dir: Path, mode: str) -> Path:
    output_path = output_dir / "comparison_summary.csv"
    fieldnames = [
        "group_name",
        "group_label",
        "run_count",
        "completion_time_sec",
        "final_known_area_m2",
        "total_path_length_m",
        "total_goals_succeeded",
        "total_goals_failed",
        "map_growth_rate_m2_per_sec",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group in GROUP_ORDER:
            runs = grouped_runs.get(group)
            if not runs:
                continue
            row = {
                "group_name": group,
                "group_label": GROUP_LABELS[group],
                "run_count": len(runs),
            }
            for metric in fieldnames[3:]:
                stats = summarize_metric(runs, metric)
                row[metric] = stats["latest"] if mode == "latest" else stats["mean"]
            writer.writerow(row)
    return output_path


def write_manifest(output_dir: Path, mode: str, grouped_runs: Dict[str, List[RunArtifacts]], generated_paths: Sequence[Path]) -> Path:
    manifest = {
        "mode": mode,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "groups": {
            group: [run.run_name for run in runs]
            for group, runs in grouped_runs.items()
        },
        "figures": [str(path) for path in generated_paths],
    }
    output_path = output_dir / "visualization_manifest.json"
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return output_path


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    results_root = args.results_root.expanduser().resolve()
    output_dir = ensure_output_dir(args.output_dir, results_root)
    grouped_runs = collect_runs(results_root)
    if not grouped_runs:
        raise RuntimeError(f"No experiment runs found under {results_root}")

    generated_paths: List[Path] = []
    generated_paths.extend(plot_summary_bars(grouped_runs, output_dir, args.mode, args.dpi))
    generated_paths.append(plot_goal_outcomes(grouped_runs, output_dir, args.mode, args.dpi))
    generated_paths.append(plot_path_lengths_per_robot(grouped_runs, output_dir, args.mode, args.dpi))
    generated_paths.append(
        plot_curve_comparison(
            grouped_runs,
            output_dir,
            args.mode,
            args.dpi,
            series_name="map_growth",
            value_key="known_area_m2",
            title="Experiment 1: Known Area Growth",
            ylabel="Known Area (m²)",
            filename="known_area_growth.png",
        )
    )
    generated_paths.append(
        plot_curve_comparison(
            grouped_runs,
            output_dir,
            args.mode,
            args.dpi,
            series_name="frontier_series",
            value_key="valid_frontier_count",
            title="Experiment 1: Valid Frontier Count",
            ylabel="Valid Frontier Count",
            filename="valid_frontiers.png",
        )
    )
    generated_paths.append(
        plot_curve_comparison(
            grouped_runs,
            output_dir,
            args.mode,
            args.dpi,
            series_name="frontier_series",
            value_key="idle_robot_count",
            title="Experiment 1: Idle Robot Count",
            ylabel="Idle Robot Count",
            filename="idle_robots.png",
        )
    )
    generated_paths.append(write_comparison_summary_csv(grouped_runs, output_dir, args.mode))
    generated_paths.append(write_manifest(output_dir, args.mode, grouped_runs, generated_paths))

    print(f"[experiment1] Wrote visualization artifacts to: {output_dir}")
    for path in generated_paths:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
