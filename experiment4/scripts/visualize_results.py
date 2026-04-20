#!/usr/bin/env python3
"""Visualization utilities for experiment 4 result artifacts."""

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
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GROUP_ORDER = [
    "group_a_no_gt_seed",
    "group_b_gt_seed_transform_only",
    "group_c_gt_seed_direct_icp",
    "group_d_gt_seed_overlap_icp",
]

GROUP_LABELS = {
    "group_a_no_gt_seed": "A No GT Seed",
    "group_b_gt_seed_transform_only": "B GT Transform Only",
    "group_c_gt_seed_direct_icp": "C GT + Direct ICP",
    "group_d_gt_seed_overlap_icp": "D GT + Overlap ICP",
}

GROUP_COLORS = {
    "group_a_no_gt_seed": "#c44e52",
    "group_b_gt_seed_transform_only": "#4c72b0",
    "group_c_gt_seed_direct_icp": "#dd8452",
    "group_d_gt_seed_overlap_icp": "#55a868",
}


@dataclass
class RunArtifacts:
    group_name: str
    run_name: str
    summary: Dict


def resolve_package_root() -> Path:
    try:
        import rospkg  # type: ignore

        return Path(rospkg.RosPack().get_path("experiment4"))
    except Exception:
        return Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    package_root = resolve_package_root()
    parser = argparse.ArgumentParser(description="Visualize experiment4 result artifacts.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=package_root / "results",
        help="Directory containing experiment4 group run folders.",
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
        default=220,
        help="Figure DPI.",
    )
    return parser


def read_summary(run_dir: Path) -> Dict:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def load_run(run_dir: Path, group_name: str) -> RunArtifacts:
    return RunArtifacts(
        group_name=group_name,
        run_name=run_dir.name,
        summary=read_summary(run_dir),
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


def ensure_output_dir(path: Optional[Path], results_root: Path) -> Path:
    if path is not None:
        output_dir = path.expanduser().resolve()
    else:
        output_dir = (results_root / "figures" / time.strftime("%Y%m%d_%H%M%S")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def configure_paper_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "font.size": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    })


def nested_get(payload: Dict, path: Sequence[str], default=np.nan):
    current = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def summarize_metric(runs: Sequence[RunArtifacts], path: Sequence[str]) -> Dict[str, float]:
    values = []
    for run in runs:
        raw_value = nested_get(run.summary, path, np.nan)
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            value = float("nan")
        if np.isfinite(value):
            values.append(value)

    if not values:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "latest": float("nan"),
            "count": 0,
        }

    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "latest": float(values[-1]),
        "count": len(values),
    }


def current_value(stats: Dict[str, float], mode: str) -> float:
    return stats["latest"] if mode == "latest" else stats["mean"]


def current_error(stats: Dict[str, float], mode: str) -> float:
    if mode == "latest" or stats["count"] <= 1 or not np.isfinite(stats["std"]):
        return 0.0
    return stats["std"]


def save_figure(fig, output_path: Path, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def decorate_axes(ax, title: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)


def draw_metric_bars(
    ax,
    grouped_runs: Dict[str, List[RunArtifacts]],
    mode: str,
    metric_path: Sequence[str],
    title: str,
    ylabel: str,
    better: str,
) -> None:
    groups = [group for group in GROUP_ORDER if group in grouped_runs]
    labels = [GROUP_LABELS[group] for group in groups]
    x_positions = np.arange(len(groups), dtype=np.float64)

    values: List[float] = []
    errors: List[float] = []
    missing_indices: List[int] = []
    valid_entries: List[Tuple[int, float]] = []

    for index, group in enumerate(groups):
        stats = summarize_metric(grouped_runs[group], metric_path)
        value = current_value(stats, mode)
        error = current_error(stats, mode)
        if np.isfinite(value):
            values.append(value)
            errors.append(error)
            valid_entries.append((index, value))
        else:
            values.append(0.0)
            errors.append(0.0)
            missing_indices.append(index)

    colors = [GROUP_COLORS[group] for group in groups]
    bars = ax.bar(x_positions, values, yerr=errors if any(err > 0 for err in errors) else None, color=colors, capsize=5)

    for index in missing_indices:
        bars[index].set_facecolor("#f3f3f3")
        bars[index].set_edgecolor("#666666")
        bars[index].set_hatch("//")

    for index, value in valid_entries:
        ax.text(
            x_positions[index],
            value + max(errors[index], 0.0) + (0.02 * max(values) if max(values) > 0 else 0.02),
            f"{value:.3f}" if abs(value) < 1000 else f"{value:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    for index in missing_indices:
        ax.text(
            x_positions[index],
            0.02 * max(values) if max(values) > 0 else 0.02,
            "N/A",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#555555",
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    decorate_axes(ax, title, ylabel)
    subtitle = "Higher is better" if better == "higher" else "Lower is better"
    ax.text(0.99, 1.02, subtitle, transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="#555555")


def plot_registration_fitness(grouped_runs: Dict[str, List[RunArtifacts]], output_dir: Path, mode: str, dpi: int) -> Path:
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    draw_metric_bars(
        ax=ax,
        grouped_runs=grouped_runs,
        mode=mode,
        metric_path=["registration_metrics", "mean_fitness"],
        title="Experiment 4: Valid-Pair Registration Fitness",
        ylabel="Fitness (uniform evaluation)",
        better="higher",
    )
    output_path = output_dir / "registration_fitness.png"
    save_figure(fig, output_path, dpi)
    return output_path


def plot_registration_rmse(grouped_runs: Dict[str, List[RunArtifacts]], output_dir: Path, mode: str, dpi: int) -> Path:
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    draw_metric_bars(
        ax=ax,
        grouped_runs=grouped_runs,
        mode=mode,
        metric_path=["registration_metrics", "mean_inlier_rmse"],
        title="Experiment 4: Valid-Pair Registration RMSE",
        ylabel="Inlier RMSE (m, uniform evaluation)",
        better="lower",
    )
    output_path = output_dir / "registration_rmse.png"
    save_figure(fig, output_path, dpi)
    return output_path


def plot_registration_validity(grouped_runs: Dict[str, List[RunArtifacts]], output_dir: Path, mode: str, dpi: int) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    draw_metric_bars(
        ax=axes[0],
        grouped_runs=grouped_runs,
        mode=mode,
        metric_path=["registration_metrics", "valid_pair_count"],
        title="Valid Pair Count",
        ylabel="Pair Count",
        better="higher",
    )
    draw_metric_bars(
        ax=axes[1],
        grouped_runs=grouped_runs,
        mode=mode,
        metric_path=["registration_metrics", "valid_pair_ratio"],
        title="Valid Pair Ratio",
        ylabel="Ratio",
        better="higher",
    )
    fig.suptitle("Experiment 4: Pair Validity", fontsize=13, y=1.03)
    output_path = output_dir / "registration_validity.png"
    save_figure(fig, output_path, dpi)
    return output_path


def plot_gt_transform_error(grouped_runs: Dict[str, List[RunArtifacts]], output_dir: Path, mode: str, dpi: int) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    draw_metric_bars(
        ax=axes[0],
        grouped_runs=grouped_runs,
        mode=mode,
        metric_path=["registration_metrics", "mean_gt_translation_error_valid_only_m"],
        title="Relative GT Translation Error",
        ylabel="Translation Error (m)",
        better="lower",
    )
    draw_metric_bars(
        ax=axes[1],
        grouped_runs=grouped_runs,
        mode=mode,
        metric_path=["registration_metrics", "mean_gt_rotation_error_valid_only_deg"],
        title="Relative GT Rotation Error",
        ylabel="Rotation Error (deg)",
        better="lower",
    )
    fig.suptitle("Experiment 4: Relative Ground-Truth Transform Error", fontsize=13, y=1.03)
    output_path = output_dir / "gt_transform_error.png"
    save_figure(fig, output_path, dpi)
    return output_path


def plot_cloud_completeness(grouped_runs: Dict[str, List[RunArtifacts]], output_dir: Path, mode: str, dpi: int) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    draw_metric_bars(
        ax=axes[0],
        grouped_runs=grouped_runs,
        mode=mode,
        metric_path=["global_metrics", "filtered_point_count"],
        title="Filtered Point Count",
        ylabel="Points",
        better="higher",
    )
    draw_metric_bars(
        ax=axes[1],
        grouped_runs=grouped_runs,
        mode=mode,
        metric_path=["global_metrics", "filtered_bbox_volume"],
        title="Spatial Coverage Volume",
        ylabel="Bounding Box Volume (m³)",
        better="higher",
    )
    fig.suptitle("Experiment 4: Point-Cloud Completeness", fontsize=13, y=1.03)
    output_path = output_dir / "cloud_completeness.png"
    save_figure(fig, output_path, dpi)
    return output_path


def plot_mesh_continuity(grouped_runs: Dict[str, List[RunArtifacts]], output_dir: Path, mode: str, dpi: int) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    draw_metric_bars(
        ax=axes[0],
        grouped_runs=grouped_runs,
        mode=mode,
        metric_path=["global_metrics", "mesh_largest_component_triangle_ratio"],
        title="Largest Connected Component Ratio",
        ylabel="Triangle Ratio",
        better="higher",
    )
    draw_metric_bars(
        ax=axes[1],
        grouped_runs=grouped_runs,
        mode=mode,
        metric_path=["global_metrics", "mesh_component_count"],
        title="Connected Component Count",
        ylabel="Component Count",
        better="lower",
    )
    fig.suptitle("Experiment 4: Mesh Continuity", fontsize=13, y=1.03)
    output_path = output_dir / "mesh_continuity.png"
    save_figure(fig, output_path, dpi)
    return output_path


def write_comparison_summary_csv(grouped_runs: Dict[str, List[RunArtifacts]], output_dir: Path, mode: str) -> Path:
    output_path = output_dir / "comparison_summary.csv"
    fieldnames = [
        "group_name",
        "group_label",
        "run_count",
        "registration_valid_pair_count",
        "registration_invalid_pair_count",
        "registration_valid_pair_ratio",
        "registration_mean_fitness",
        "registration_mean_inlier_rmse",
        "registration_mean_fitness_valid_only",
        "registration_mean_inlier_rmse_valid_only",
        "registration_mean_gt_translation_error_m",
        "registration_mean_gt_rotation_error_deg",
        "registration_mean_gt_translation_error_valid_only_m",
        "registration_mean_gt_rotation_error_valid_only_deg",
        "filtered_point_count",
        "filtered_bbox_volume",
        "mesh_component_count",
        "mesh_largest_component_triangle_ratio",
        "mesh_triangle_count",
        "mesh_vertex_count",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
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
                "registration_valid_pair_count": current_value(summarize_metric(runs, ["registration_metrics", "valid_pair_count"]), mode),
                "registration_invalid_pair_count": current_value(summarize_metric(runs, ["registration_metrics", "invalid_pair_count"]), mode),
                "registration_valid_pair_ratio": current_value(summarize_metric(runs, ["registration_metrics", "valid_pair_ratio"]), mode),
                "registration_mean_fitness": current_value(summarize_metric(runs, ["registration_metrics", "mean_fitness"]), mode),
                "registration_mean_inlier_rmse": current_value(summarize_metric(runs, ["registration_metrics", "mean_inlier_rmse"]), mode),
                "registration_mean_fitness_valid_only": current_value(
                    summarize_metric(runs, ["registration_metrics", "mean_fitness_valid_only"]),
                    mode,
                ),
                "registration_mean_inlier_rmse_valid_only": current_value(
                    summarize_metric(runs, ["registration_metrics", "mean_inlier_rmse_valid_only"]),
                    mode,
                ),
                "registration_mean_gt_translation_error_m": current_value(
                    summarize_metric(runs, ["registration_metrics", "mean_gt_translation_error_m"]),
                    mode,
                ),
                "registration_mean_gt_rotation_error_deg": current_value(
                    summarize_metric(runs, ["registration_metrics", "mean_gt_rotation_error_deg"]),
                    mode,
                ),
                "registration_mean_gt_translation_error_valid_only_m": current_value(
                    summarize_metric(runs, ["registration_metrics", "mean_gt_translation_error_valid_only_m"]),
                    mode,
                ),
                "registration_mean_gt_rotation_error_valid_only_deg": current_value(
                    summarize_metric(runs, ["registration_metrics", "mean_gt_rotation_error_valid_only_deg"]),
                    mode,
                ),
                "filtered_point_count": current_value(summarize_metric(runs, ["global_metrics", "filtered_point_count"]), mode),
                "filtered_bbox_volume": current_value(summarize_metric(runs, ["global_metrics", "filtered_bbox_volume"]), mode),
                "mesh_component_count": current_value(summarize_metric(runs, ["global_metrics", "mesh_component_count"]), mode),
                "mesh_largest_component_triangle_ratio": current_value(
                    summarize_metric(runs, ["global_metrics", "mesh_largest_component_triangle_ratio"]),
                    mode,
                ),
                "mesh_triangle_count": current_value(summarize_metric(runs, ["global_metrics", "mesh_triangle_count"]), mode),
                "mesh_vertex_count": current_value(summarize_metric(runs, ["global_metrics", "mesh_vertex_count"]), mode),
            }
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
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_paper_style()

    results_root = args.results_root.expanduser().resolve()
    output_dir = ensure_output_dir(args.output_dir, results_root)
    grouped_runs = collect_runs(results_root)
    if not grouped_runs:
        raise RuntimeError(f"No experiment runs found under {results_root}")

    generated_paths: List[Path] = []
    generated_paths.append(plot_registration_validity(grouped_runs, output_dir, args.mode, args.dpi))
    generated_paths.append(plot_registration_fitness(grouped_runs, output_dir, args.mode, args.dpi))
    generated_paths.append(plot_registration_rmse(grouped_runs, output_dir, args.mode, args.dpi))
    generated_paths.append(plot_gt_transform_error(grouped_runs, output_dir, args.mode, args.dpi))
    generated_paths.append(plot_cloud_completeness(grouped_runs, output_dir, args.mode, args.dpi))
    generated_paths.append(plot_mesh_continuity(grouped_runs, output_dir, args.mode, args.dpi))
    generated_paths.append(write_comparison_summary_csv(grouped_runs, output_dir, args.mode))
    generated_paths.append(write_manifest(output_dir, args.mode, grouped_runs, generated_paths))

    print(f"[experiment4] Wrote visualization artifacts to: {output_dir}")
    for path in generated_paths:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
