from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recon.config import Config, load_config
from recon.fragment_builder import load_keyframes_from_json
from recon.io_utils import ProjectPaths, load_json
from recon.types import KeyframeRecord, RegistrationEdge


@dataclass
class PoseGraphBuildResult:
    """Open3D pose graph plus node-key mappings for serialization."""

    pose_graph: o3d.pipelines.registration.PoseGraph
    key_to_node_index: dict[str, int]
    node_index_to_key: list[str]


@dataclass
class PoseGraphResidualStats:
    """Residual summary for pose graph edge consistency."""

    edge_count: int
    mean_translation_error_m: float
    max_translation_error_m: float
    mean_rotation_error_deg: float
    max_rotation_error_deg: float


def build_pose_graph_nodes(keyframes: Sequence[KeyframeRecord]) -> tuple[list[o3d.pipelines.registration.PoseGraphNode], dict[str, int], list[str]]:
    """Create pose graph nodes initialized from current keyframe world poses."""

    sorted_keyframes = sorted(keyframes, key=lambda item: (item.robot_name, item.frame_index, item.keyframe_index))
    nodes: list[o3d.pipelines.registration.PoseGraphNode] = []
    key_to_node_index: dict[str, int] = {}
    node_index_to_key: list[str] = []

    for node_index, keyframe in enumerate(sorted_keyframes):
        nodes.append(o3d.pipelines.registration.PoseGraphNode(keyframe.frame.pose.matrix.copy()))
        key_to_node_index[keyframe.key] = node_index
        node_index_to_key.append(keyframe.key)

    return nodes, key_to_node_index, node_index_to_key


def build_pose_graph_edges(
    registration_edges: Sequence[RegistrationEdge],
    key_to_node_index: dict[str, int],
) -> list[o3d.pipelines.registration.PoseGraphEdge]:
    """Convert shared RegistrationEdge objects into Open3D pose graph edges."""

    pose_graph_edges: list[o3d.pipelines.registration.PoseGraphEdge] = []
    for edge in registration_edges:
        if edge.source_key not in key_to_node_index or edge.target_key not in key_to_node_index:
            continue

        source_node_index = key_to_node_index[edge.source_key]
        target_node_index = key_to_node_index[edge.target_key]
        uncertain = bool(edge.is_loop_closure or edge.is_cross_robot)
        pose_graph_edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                source_node_index,
                target_node_index,
                edge.transformation.copy(),
                edge.information.copy(),
                uncertain,
            )
        )
    return pose_graph_edges


def build_pose_graph(keyframes: Sequence[KeyframeRecord], registration_edges: Sequence[RegistrationEdge]) -> PoseGraphBuildResult:
    """Build an Open3D pose graph from keyframes and accepted registration edges."""

    nodes, key_to_node_index, node_index_to_key = build_pose_graph_nodes(keyframes)
    pose_graph_edges = build_pose_graph_edges(registration_edges, key_to_node_index)
    pose_graph = o3d.pipelines.registration.PoseGraph()
    pose_graph.nodes.extend(nodes)
    pose_graph.edges.extend(pose_graph_edges)
    return PoseGraphBuildResult(
        pose_graph=pose_graph,
        key_to_node_index=key_to_node_index,
        node_index_to_key=node_index_to_key,
    )


def optimize_pose_graph(pose_graph: o3d.pipelines.registration.PoseGraph, config: Config) -> o3d.pipelines.registration.PoseGraph:
    """Run Open3D global pose graph optimization in place and return the graph."""

    options = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=config.pose_graph_optimization.max_correspondence_distance,
        edge_prune_threshold=config.pose_graph_optimization.edge_prune_threshold,
        preference_loop_closure=config.pose_graph_optimization.preference_loop_closure,
        reference_node=config.pose_graph_optimization.reference_node,
    )
    method = o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt()
    criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
    o3d.pipelines.registration.global_optimization(pose_graph, method, criteria, options)
    return pose_graph


def serialize_pose_graph(
    keyframes: Sequence[KeyframeRecord],
    registration_edges: Sequence[RegistrationEdge],
    pose_graph: o3d.pipelines.registration.PoseGraph,
    node_index_to_key: Sequence[str],
) -> dict[str, object]:
    """Serialize nodes and edges for stage handoff."""

    keyframe_by_key = {keyframe.key: keyframe for keyframe in keyframes}
    nodes_payload: list[dict[str, object]] = []
    for node_index, key in enumerate(node_index_to_key):
        keyframe = keyframe_by_key[key]
        nodes_payload.append(
            {
                "node_index": node_index,
                "key": key,
                "robot_name": keyframe.robot_name,
                "frame_index": keyframe.frame_index,
                "keyframe_index": keyframe.keyframe_index,
                "timestamp": keyframe.timestamp,
                "pose": np.asarray(pose_graph.nodes[node_index].pose, dtype=np.float64),
            }
        )

    edges_payload: list[dict[str, object]] = []
    for edge in registration_edges:
        edges_payload.append(
            {
                "source_key": edge.source_key,
                "target_key": edge.target_key,
                "transformation": edge.transformation,
                "information": edge.information,
                "fitness": edge.fitness,
                "rmse": edge.rmse,
                "uncertain": bool(edge.is_loop_closure or edge.is_cross_robot),
                "is_loop_closure": edge.is_loop_closure,
                "is_cross_robot": edge.is_cross_robot,
            }
        )

    return {"nodes": nodes_payload, "edges": edges_payload}


def load_pose_graph_payload(path: str | Path) -> dict[str, object]:
    """Load serialized pose graph data from JSON."""

    return load_json(path)


def pose_graph_from_payload(payload: dict[str, object]) -> tuple[o3d.pipelines.registration.PoseGraph, list[str]]:
    """Reconstruct an Open3D pose graph from serialized JSON payload."""

    pose_graph = o3d.pipelines.registration.PoseGraph()
    node_index_to_key: list[str] = []

    for node in payload["nodes"]:
        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(np.asarray(node["pose"], dtype=np.float64))
        )
        node_index_to_key.append(str(node["key"]))

    key_to_node_index = {key: index for index, key in enumerate(node_index_to_key)}
    for edge in payload["edges"]:
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                key_to_node_index[str(edge["source_key"])],
                key_to_node_index[str(edge["target_key"])],
                np.asarray(edge["transformation"], dtype=np.float64),
                np.asarray(edge["information"], dtype=np.float64),
                bool(edge["uncertain"]),
            )
        )

    return pose_graph, node_index_to_key


def optimized_pose_payload(
    keyframes: Sequence[KeyframeRecord],
    pose_graph: o3d.pipelines.registration.PoseGraph,
    node_index_to_key: Sequence[str],
) -> list[dict[str, object]]:
    """Export optimized node poses in a keyframe-friendly format."""

    keyframe_by_key = {keyframe.key: keyframe for keyframe in keyframes}
    payload: list[dict[str, object]] = []
    for node_index, key in enumerate(node_index_to_key):
        keyframe = keyframe_by_key[key]
        payload.append(
            {
                "node_index": node_index,
                "key": key,
                "robot_name": keyframe.robot_name,
                "frame_index": keyframe.frame_index,
                "keyframe_index": keyframe.keyframe_index,
                "timestamp": keyframe.timestamp,
                "optimized_pose": np.asarray(pose_graph.nodes[node_index].pose, dtype=np.float64),
            }
        )
    return payload


def _rotation_angle_deg(rotation_matrix: np.ndarray) -> float:
    """Convert a relative rotation matrix into a single angle in degrees."""

    trace_value = np.trace(rotation_matrix)
    cosine = np.clip((trace_value - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def summarize_pose_graph_residuals(
    payload: dict[str, object],
    pose_graph: o3d.pipelines.registration.PoseGraph,
    node_index_to_key: Sequence[str],
) -> PoseGraphResidualStats:
    """Summarize edge residuals under the current optimized node poses."""

    if not payload["edges"]:
        return PoseGraphResidualStats(
            edge_count=0,
            mean_translation_error_m=0.0,
            max_translation_error_m=0.0,
            mean_rotation_error_deg=0.0,
            max_rotation_error_deg=0.0,
        )

    key_to_pose = {
        key: np.asarray(pose_graph.nodes[node_index].pose, dtype=np.float64)
        for node_index, key in enumerate(node_index_to_key)
    }
    translation_errors: list[float] = []
    rotation_errors: list[float] = []

    for edge in payload["edges"]:
        source_pose = key_to_pose[str(edge["source_key"])]
        target_pose = key_to_pose[str(edge["target_key"])]
        predicted_transformation = np.linalg.inv(target_pose) @ source_pose
        measured_transformation = np.asarray(edge["transformation"], dtype=np.float64)
        delta = np.linalg.inv(measured_transformation) @ predicted_transformation
        translation_errors.append(float(np.linalg.norm(delta[:3, 3])))
        rotation_errors.append(_rotation_angle_deg(delta[:3, :3]))

    return PoseGraphResidualStats(
        edge_count=len(translation_errors),
        mean_translation_error_m=float(np.mean(translation_errors)),
        max_translation_error_m=float(np.max(translation_errors)),
        mean_rotation_error_deg=float(np.mean(rotation_errors)),
        max_rotation_error_deg=float(np.max(rotation_errors)),
    )


def main() -> None:
    """Simple example loading keyframes and building an empty initial graph."""

    config_path = PROJECT_ROOT / "config" / "default.yaml"
    config = load_config(config_path)
    paths = ProjectPaths(config)
    keyframes = load_keyframes_from_json(paths.stage_file("stage_02_keyframes", "keyframes.json", config.robots[1]))
    result = build_pose_graph(keyframes, [])

    print("Nodes:", len(result.pose_graph.nodes))
    print("Edges:", len(result.pose_graph.edges))


if __name__ == "__main__":
    main()
