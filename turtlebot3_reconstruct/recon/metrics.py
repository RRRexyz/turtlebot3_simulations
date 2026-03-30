from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recon.config import Config, load_config
from recon.io_utils import ProjectPaths, load_json


@dataclass
class EdgeMetricRecord:
    """Per-edge metrics used in plots and tables."""

    source_key: str
    target_key: str
    pair_type: str
    selected_method: str
    initial_fitness: float
    initial_rmse: float
    final_fitness: float
    final_rmse: float
    is_valid: bool


def load_registration_summary(paths: ProjectPaths) -> dict[str, object]:
    """Load stage_04 registration summary."""

    return load_json(paths.stage_file("stage_04_register", "registration_summary.json"))


def load_optimization_summary(paths: ProjectPaths) -> dict[str, object]:
    """Load stage_05 optimization summary."""

    return load_json(paths.stage_file("stage_05_optimize", "optimization_summary.json"))


def load_pose_graph_initial(paths: ProjectPaths) -> dict[str, object]:
    """Load serialized initial pose graph."""

    return load_json(paths.stage_file("stage_04_register", "pose_graph_initial.json"))


def load_pose_graph_optimized(paths: ProjectPaths) -> dict[str, object]:
    """Load serialized optimized pose graph."""

    return load_json(paths.stage_file("stage_05_optimize", "pose_graph_optimized.json"))


def load_keyframe_counts(paths: ProjectPaths, config: Config) -> dict[str, int]:
    """Load per-robot keyframe counts from stage_02 outputs."""

    counts: dict[str, int] = {}
    for robot_name in config.robots:
        stats_path = paths.stage_file("stage_02_keyframes", "keyframe_stats.json", robot_name)
        if not stats_path.exists():
            continue
        stats = load_json(stats_path)
        counts[robot_name] = int(stats["keyframe_count"])
    return counts


def collect_edge_metrics(summary_payload: dict[str, object]) -> list[EdgeMetricRecord]:
    """Flatten registration summary into per-edge metric records."""

    records: list[EdgeMetricRecord] = []
    for item in summary_payload["registration_summary"]:
        initial = item.get("initial", {})
        final = item.get("final", {})
        records.append(
            EdgeMetricRecord(
                source_key=str(item["source_key"]),
                target_key=str(item["target_key"]),
                pair_type=str(item["pair_type"]),
                selected_method=str(item["selected_method"]),
                initial_fitness=float(initial.get("fitness", 0.0)),
                initial_rmse=float(initial.get("inlier_rmse", 0.0)),
                final_fitness=float(final.get("fitness", item.get("fitness", 0.0))),
                final_rmse=float(final.get("inlier_rmse", item.get("inlier_rmse", 0.0))),
                is_valid=bool(item["is_valid"]),
            )
        )
    return records


def build_metrics_summary(config: Config, paths: ProjectPaths) -> dict[str, object]:
    """Aggregate the metrics needed for stage_07 evaluation outputs."""

    registration_summary = load_registration_summary(paths)
    optimization_summary = load_optimization_summary(paths)
    integration_stats = load_json(paths.stage_file("stage_06_integrate", "integration_stats.json"))
    edge_metrics = collect_edge_metrics(registration_summary)

    success_edge_count = sum(1 for metric in edge_metrics if metric.is_valid)
    failure_edge_count = sum(1 for metric in edge_metrics if not metric.is_valid)
    keyframe_counts = load_keyframe_counts(paths, config)

    return {
        "edge_metrics": [
            {
                "source_key": metric.source_key,
                "target_key": metric.target_key,
                "pair_type": metric.pair_type,
                "selected_method": metric.selected_method,
                "initial_fitness": metric.initial_fitness,
                "initial_rmse": metric.initial_rmse,
                "final_fitness": metric.final_fitness,
                "final_rmse": metric.final_rmse,
                "is_valid": metric.is_valid,
            }
            for metric in edge_metrics
        ],
        "success_edge_count": success_edge_count,
        "failure_edge_count": failure_edge_count,
        "keyframe_counts": keyframe_counts,
        "registration_summary": registration_summary,
        "optimization_summary": optimization_summary,
        "integration_stats": integration_stats,
    }


def main() -> None:
    """Simple example for manual verification."""

    config_path = PROJECT_ROOT / "config" / "default.yaml"
    config = load_config(config_path)
    paths = ProjectPaths(config)
    summary = build_metrics_summary(config, paths)

    print("Success edges:", summary["success_edge_count"])
    print("Failure edges:", summary["failure_edge_count"])
    print("Keyframe counts:", summary["keyframe_counts"])


if __name__ == "__main__":
    main()
