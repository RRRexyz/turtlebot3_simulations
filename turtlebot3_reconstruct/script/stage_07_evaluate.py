from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recon.config import load_config
from recon.io_utils import ProjectPaths, save_csv, save_json, setup_logger
from recon.metrics import (
    build_metrics_summary,
    load_pose_graph_initial,
    load_pose_graph_optimized,
)
from recon.viz import (
    save_edge_metrics_plot,
    save_final_map_plot,
    save_keyframe_count_plot,
    save_pose_graph_before_after_plot,
    save_registration_before_after_plot,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation and figure generation."""

    parser = argparse.ArgumentParser(description="Generate evaluation metrics and paper-ready figures.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "default.yaml",
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    config = load_config(args.config)
    paths = ProjectPaths(config)
    logger = setup_logger("stage_07_evaluate", paths.log_file("stage_07_evaluate"))
    stage_dir = paths.stage_dir("stage_07_evaluate")

    metrics_summary = build_metrics_summary(config, paths)
    initial_pose_graph = load_pose_graph_initial(paths)
    optimized_pose_graph = load_pose_graph_optimized(paths)

    save_json(stage_dir / "metrics_summary.json", metrics_summary)
    save_csv(stage_dir / "edge_metrics.csv", metrics_summary["edge_metrics"])

    registration_plot = save_registration_before_after_plot(
        metrics_summary["edge_metrics"],
        stage_dir / "registration_before_after.png",
    )
    optimization_plot = save_pose_graph_before_after_plot(
        initial_pose_graph,
        optimized_pose_graph,
        stage_dir / "optimization_before_after.png",
    )
    edge_metrics_plot = save_edge_metrics_plot(
        metrics_summary["edge_metrics"],
        stage_dir / "edge_metrics.png",
    )
    keyframe_count_plot = save_keyframe_count_plot(
        metrics_summary["keyframe_counts"],
        stage_dir / "keyframe_counts.png",
    )
    final_map_plot = save_final_map_plot(
        paths.stage_file("stage_06_integrate", "final_map.ply"),
        paths.stage_file("stage_06_integrate", "final_mesh.ply"),
        stage_dir / "final_reconstruction.png",
    )

    logger.info(
        "Generated evaluation assets | success_edges=%d failure_edges=%d",
        metrics_summary["success_edge_count"],
        metrics_summary["failure_edge_count"],
    )

    print("metrics:", stage_dir / "metrics_summary.json")
    print("registration_plot:", registration_plot)
    print("optimization_plot:", optimization_plot)
    print("edge_metrics_plot:", edge_metrics_plot)
    print("keyframe_count_plot:", keyframe_count_plot)
    print("final_map_plot:", final_map_plot)


if __name__ == "__main__":
    main()
