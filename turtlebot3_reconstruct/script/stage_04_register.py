from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recon.config import Config, load_config
from recon.io_utils import ProjectPaths, save_json, setup_logger
from recon.pair_selector import PairCandidate, load_all_keyframes, select_pair_candidates
from recon.pose_graph_builder import build_pose_graph, serialize_pose_graph
from recon.registration_global import run_fgr_global_registration, run_ransac_global_registration
from recon.registration_local import (
    LocalRegistrationResult,
    evaluate_registration_result,
    run_colored_icp,
    run_point_to_plane_icp,
    to_registration_edge,
)
from recon.rgbd_utils import build_pointcloud_result
from recon.types import KeyframeRecord, RegistrationEdge


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for registration and initial graph construction."""

    parser = argparse.ArgumentParser(description="Register keyframe pairs and build an initial pose graph.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "default.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--robots",
        nargs="*",
        default=None,
        help="Optional subset of robots to process. Defaults to all robots in config.",
    )
    return parser.parse_args()


def _result_score(result: LocalRegistrationResult) -> tuple[int, float, float]:
    """Rank local registration outputs by validity, fitness, and RMSE."""

    return (1 if result.is_valid else 0, result.fitness, -result.inlier_rmse)


def _choose_best_local_result(results: list[LocalRegistrationResult]) -> LocalRegistrationResult:
    """Choose the best local registration result from multiple refinements."""

    return max(results, key=_result_score)


def _global_result_score(result) -> tuple[int, float, float]:
    """Rank coarse registration outputs."""

    return (1 if result.is_valid else 0, result.fitness, -result.inlier_rmse)


def _choose_best_global_result(results: list) -> object:
    """Choose the best coarse registration result."""

    return max(results, key=_global_result_score)


def _pointcloud_cache(keyframes: Iterable[KeyframeRecord], config: Config) -> dict[str, object]:
    """Build reusable camera-frame point clouds for registration."""

    cache: dict[str, object] = {}
    for keyframe in keyframes:
        cache[keyframe.key] = build_pointcloud_result(keyframe, config, coordinate_frame="camera").pointcloud
    return cache


def _register_odometry_candidate(
    candidate: PairCandidate,
    pointcloud_by_key: dict[str, object],
    config: Config,
) -> tuple[RegistrationEdge, dict[str, object]]:
    """Refine an adjacent same-robot edge with local ICP."""

    source_cloud = pointcloud_by_key[candidate.source_key]
    target_cloud = pointcloud_by_key[candidate.target_key]
    initial_result = evaluate_registration_result(
        source_cloud,
        target_cloud,
        candidate.initial_transformation,
        config,
        method="odometry_initial",
    )
    icp_result = run_point_to_plane_icp(source_cloud, target_cloud, candidate.initial_transformation, config)
    colored_init = icp_result.transformation if icp_result.is_valid else candidate.initial_transformation
    colored_result = run_colored_icp(source_cloud, target_cloud, colored_init, config)
    final_result = _choose_best_local_result([initial_result, icp_result, colored_result])
    edge = to_registration_edge(candidate.source_key, candidate.target_key, final_result)
    return edge, {
        "source_key": candidate.source_key,
        "target_key": candidate.target_key,
        "pair_type": candidate.pair_type,
        "selected_method": final_result.method,
        "is_valid": final_result.is_valid,
        "translation_distance_m": candidate.translation_distance_m,
        "rotation_difference_deg": candidate.rotation_difference_deg,
        "initial": {
            "method": initial_result.method,
            "fitness": initial_result.fitness,
            "inlier_rmse": initial_result.inlier_rmse,
            "is_valid": initial_result.is_valid,
        },
        "final": {
            "method": final_result.method,
            "fitness": final_result.fitness,
            "inlier_rmse": final_result.inlier_rmse,
            "is_valid": final_result.is_valid,
        },
    }


def _register_cross_robot_candidate(
    candidate: PairCandidate,
    pointcloud_by_key: dict[str, object],
    config: Config,
) -> tuple[RegistrationEdge | None, dict[str, object]]:
    """Estimate a coarse alignment for cross-robot pairs, then refine locally."""

    source_cloud = pointcloud_by_key[candidate.source_key]
    target_cloud = pointcloud_by_key[candidate.target_key]
    fgr_result = run_fgr_global_registration(source_cloud, target_cloud, config)
    ransac_result = run_ransac_global_registration(source_cloud, target_cloud, config)
    coarse_result = _choose_best_global_result([fgr_result, ransac_result])

    if not coarse_result.is_valid:
        return None, {
            "source_key": candidate.source_key,
            "target_key": candidate.target_key,
            "pair_type": candidate.pair_type,
            "selected_method": coarse_result.method,
            "is_valid": False,
            "translation_distance_m": candidate.translation_distance_m,
            "rotation_difference_deg": candidate.rotation_difference_deg,
            "initial": {
                "method": coarse_result.method,
                "fitness": coarse_result.fitness,
                "inlier_rmse": coarse_result.inlier_rmse,
                "is_valid": coarse_result.is_valid,
            },
            "final": {
                "method": coarse_result.method,
                "fitness": coarse_result.fitness,
                "inlier_rmse": coarse_result.inlier_rmse,
                "is_valid": False,
            },
        }

    icp_result = run_point_to_plane_icp(source_cloud, target_cloud, coarse_result.t_init, config)
    colored_init = icp_result.transformation if icp_result.is_valid else coarse_result.t_init
    colored_result = run_colored_icp(source_cloud, target_cloud, colored_init, config)
    refined_result = _choose_best_local_result([icp_result, colored_result])

    if not refined_result.is_valid:
        return None, {
            "source_key": candidate.source_key,
            "target_key": candidate.target_key,
            "pair_type": candidate.pair_type,
            "selected_method": refined_result.method,
            "is_valid": False,
            "translation_distance_m": candidate.translation_distance_m,
            "rotation_difference_deg": candidate.rotation_difference_deg,
            "initial": {
                "method": coarse_result.method,
                "fitness": coarse_result.fitness,
                "inlier_rmse": coarse_result.inlier_rmse,
                "is_valid": coarse_result.is_valid,
            },
            "final": {
                "method": refined_result.method,
                "fitness": refined_result.fitness,
                "inlier_rmse": refined_result.inlier_rmse,
                "is_valid": False,
            },
        }

    edge = to_registration_edge(
        candidate.source_key,
        candidate.target_key,
        refined_result,
        is_cross_robot=True,
    )
    return edge, {
        "source_key": candidate.source_key,
        "target_key": candidate.target_key,
        "pair_type": candidate.pair_type,
        "selected_method": refined_result.method,
        "is_valid": True,
        "translation_distance_m": candidate.translation_distance_m,
        "rotation_difference_deg": candidate.rotation_difference_deg,
        "initial": {
            "method": coarse_result.method,
            "fitness": coarse_result.fitness,
            "inlier_rmse": coarse_result.inlier_rmse,
            "is_valid": coarse_result.is_valid,
        },
        "final": {
            "method": refined_result.method,
            "fitness": refined_result.fitness,
            "inlier_rmse": refined_result.inlier_rmse,
            "is_valid": True,
        },
    }


def run_stage(config: Config, robot_names: Iterable[str] | None = None) -> dict[str, object]:
    """Build an initial pose graph by registering selected keyframe pairs."""

    paths = ProjectPaths(config)
    logger = setup_logger("stage_04_register", paths.log_file("stage_04_register"))
    stage_dir = paths.stage_dir("stage_04_register")

    keyframes = load_all_keyframes(config, list(robot_names) if robot_names is not None else None)
    pair_selection = select_pair_candidates(keyframes, config)
    pointcloud_by_key = _pointcloud_cache(keyframes, config)

    registration_edges: list[RegistrationEdge] = []
    registration_summary: list[dict[str, object]] = []

    for candidate in pair_selection.odometry_candidates:
        edge, summary = _register_odometry_candidate(candidate, pointcloud_by_key, config)
        registration_edges.append(edge)
        registration_summary.append(summary)

    for candidate in pair_selection.cross_robot_candidates:
        edge, summary = _register_cross_robot_candidate(candidate, pointcloud_by_key, config)
        registration_summary.append(summary)
        if edge is not None:
            registration_edges.append(edge)

    pose_graph_result = build_pose_graph(keyframes, registration_edges)
    pose_graph_payload = serialize_pose_graph(
        keyframes=keyframes,
        registration_edges=registration_edges,
        pose_graph=pose_graph_result.pose_graph,
        node_index_to_key=pose_graph_result.node_index_to_key,
    )

    save_json(stage_dir / "pose_graph_initial.json", pose_graph_payload)
    summary_payload = {
        "node_count": len(pose_graph_result.pose_graph.nodes),
        "edge_count": len(pose_graph_result.pose_graph.edges),
        "odometry_edge_count": len(pair_selection.odometry_candidates),
        "loop_edge_count": len(pair_selection.loop_candidates),
        "cross_robot_candidate_count": len(pair_selection.cross_robot_candidates),
        "accepted_cross_robot_edge_count": sum(1 for edge in registration_edges if edge.is_cross_robot),
        "registration_summary": registration_summary,
    }
    save_json(stage_dir / "registration_summary.json", summary_payload)

    logger.info(
        "Built initial pose graph | nodes=%d edges=%d odom=%d cross=%d",
        len(pose_graph_result.pose_graph.nodes),
        len(pose_graph_result.pose_graph.edges),
        len(pair_selection.odometry_candidates),
        sum(1 for edge in registration_edges if edge.is_cross_robot),
    )
    return summary_payload


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    config = load_config(args.config)
    summary = run_stage(config, args.robots)
    print(
        "nodes={node_count}, edges={edge_count}, odom={odometry_edge_count}, accepted_cross={accepted_cross_robot_edge_count}".format(
            **summary
        )
    )


if __name__ == "__main__":
    main()
