from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recon.config import Config, load_config
from recon.io_utils import ProjectPaths, save_json, setup_logger
from recon.pair_selector import load_all_keyframes
from recon.pose_graph_builder import (
    load_pose_graph_payload,
    optimize_pose_graph,
    optimized_pose_payload,
    pose_graph_from_payload,
    summarize_pose_graph_residuals,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for pose graph optimization."""

    parser = argparse.ArgumentParser(description="Optimize the initial pose graph built in stage_04_register.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "default.yaml",
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


def _connected_components(payload: dict[str, object]) -> list[list[str]]:
    """Split the pose graph payload into connected components by node key."""

    adjacency: dict[str, set[str]] = {str(node["key"]): set() for node in payload["nodes"]}
    for edge in payload["edges"]:
        source_key = str(edge["source_key"])
        target_key = str(edge["target_key"])
        adjacency[source_key].add(target_key)
        adjacency[target_key].add(source_key)

    visited: set[str] = set()
    components: list[list[str]] = []
    for node_key in adjacency:
        if node_key in visited:
            continue

        stack = [node_key]
        component: list[str] = []
        visited.add(node_key)
        while stack:
            current_key = stack.pop()
            component.append(current_key)
            for neighbor_key in adjacency[current_key]:
                if neighbor_key in visited:
                    continue
                visited.add(neighbor_key)
                stack.append(neighbor_key)
        components.append(sorted(component))

    return components


def _subgraph_payload(payload: dict[str, object], component_keys: list[str]) -> dict[str, object]:
    """Extract a connected component as its own serialized pose graph payload."""

    key_set = set(component_keys)
    return {
        "nodes": [node for node in payload["nodes"] if str(node["key"]) in key_set],
        "edges": [
            edge
            for edge in payload["edges"]
            if str(edge["source_key"]) in key_set and str(edge["target_key"]) in key_set
        ],
    }


def run_stage(config: Config) -> dict[str, object]:
    """Load the initial pose graph, optimize it, and export optimized poses."""

    paths = ProjectPaths(config)
    logger = setup_logger("stage_05_optimize", paths.log_file("stage_05_optimize"))
    stage_dir = paths.stage_dir("stage_05_optimize")
    initial_pose_graph_path = paths.stage_file("stage_04_register", "pose_graph_initial.json")

    payload = load_pose_graph_payload(initial_pose_graph_path)
    pose_graph, node_index_to_key = pose_graph_from_payload(payload)
    keyframes = load_all_keyframes(config)

    before_stats = summarize_pose_graph_residuals(payload, pose_graph, node_index_to_key)
    components = _connected_components(payload)
    key_to_node_index = {key: index for index, key in enumerate(node_index_to_key)}
    for component_keys in components:
        component_payload = _subgraph_payload(payload, component_keys)
        component_pose_graph, component_node_index_to_key = pose_graph_from_payload(component_payload)
        if component_pose_graph.edges:
            optimize_pose_graph(component_pose_graph, config)
        for component_node_index, key in enumerate(component_node_index_to_key):
            pose_graph.nodes[key_to_node_index[key]].pose = component_pose_graph.nodes[component_node_index].pose

    after_stats = summarize_pose_graph_residuals(payload, pose_graph, node_index_to_key)

    optimized_payload = {
        "nodes": optimized_pose_payload(keyframes, pose_graph, node_index_to_key),
        "edges": payload["edges"],
    }
    save_json(stage_dir / "pose_graph_optimized.json", optimized_payload)
    save_json(stage_dir / "optimized_poses.json", optimized_payload["nodes"])

    summary = {
        "before": asdict(before_stats),
        "after": asdict(after_stats),
        "node_count": len(pose_graph.nodes),
        "edge_count": len(pose_graph.edges),
        "connected_component_count": len(components),
    }
    save_json(stage_dir / "optimization_summary.json", summary)

    logger.info(
        "Optimized pose graph | nodes=%d edges=%d components=%d mean_translation_before=%.6f mean_translation_after=%.6f",
        len(pose_graph.nodes),
        len(pose_graph.edges),
        len(components),
        before_stats.mean_translation_error_m,
        after_stats.mean_translation_error_m,
    )
    return summary


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    config = load_config(args.config)
    summary = run_stage(config)
    print(
        "nodes={node_count}, edges={edge_count}, before_mean_t={before[mean_translation_error_m]:.6f}, after_mean_t={after[mean_translation_error_m]:.6f}".format(
            **summary
        )
    )


if __name__ == "__main__":
    main()
