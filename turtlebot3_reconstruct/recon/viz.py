from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from recon.io_utils import load_mesh, load_pcd


def _set_equal_3d_axes(axis, points: np.ndarray) -> None:
    """Set equal aspect bounds for 3D scatter plots."""

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) * 0.5
    radius = max(maxs - mins) * 0.5
    axis.set_xlim(centers[0] - radius, centers[0] + radius)
    axis.set_ylim(centers[1] - radius, centers[1] + radius)
    axis.set_zlim(centers[2] - radius, centers[2] + radius)


def save_registration_before_after_plot(edge_metrics: Sequence[dict[str, object]], output_path: str | Path) -> Path:
    """Save a paper-friendly comparison of registration quality before and after refinement."""

    if not edge_metrics:
        raise ValueError("edge_metrics must not be empty.")

    labels = [f"{item['source_key']}->{item['target_key']}" for item in edge_metrics]
    x = np.arange(len(edge_metrics))
    initial_fitness = np.array([float(item["initial_fitness"]) for item in edge_metrics], dtype=np.float64)
    final_fitness = np.array([float(item["final_fitness"]) for item in edge_metrics], dtype=np.float64)
    initial_rmse = np.array([float(item["initial_rmse"]) for item in edge_metrics], dtype=np.float64)
    final_rmse = np.array([float(item["final_rmse"]) for item in edge_metrics], dtype=np.float64)

    figure, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(x, initial_fitness, marker="o", color="#9aa0a6", label="Initial")
    axes[0].plot(x, final_fitness, marker="o", color="#1a73e8", label="Refined")
    axes[0].set_ylabel("fitness")
    axes[0].set_title("Registration Before vs After")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(x, initial_rmse, marker="o", color="#9aa0a6", label="Initial")
    axes[1].plot(x, final_rmse, marker="o", color="#d93025", label="Refined")
    axes[1].set_ylabel("inlier RMSE")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=60, ha="right")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(output_path), dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_pose_graph_before_after_plot(
    initial_payload: dict[str, object],
    optimized_payload: dict[str, object],
    output_path: str | Path,
) -> Path:
    """Save a trajectory comparison before and after global optimization."""

    initial_by_key = {str(node["key"]): np.asarray(node["pose"], dtype=np.float64) for node in initial_payload["nodes"]}
    optimized_by_key = {
        str(node["key"]): np.asarray(node["optimized_pose"], dtype=np.float64)
        for node in optimized_payload["nodes"]
    }
    robot_names = sorted({str(node["robot_name"]) for node in optimized_payload["nodes"]})
    colors = ["#1a73e8", "#d93025", "#188038", "#f29900", "#7b1fa2"]

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False, sharey=False)
    for robot_index, robot_name in enumerate(robot_names):
        color = colors[robot_index % len(colors)]
        robot_nodes = [node for node in optimized_payload["nodes"] if str(node["robot_name"]) == robot_name]
        robot_nodes.sort(key=lambda item: int(item["frame_index"]))

        initial_positions = np.array([initial_by_key[str(node["key"])][:3, 3] for node in robot_nodes], dtype=np.float64)
        optimized_positions = np.array([optimized_by_key[str(node["key"])][:3, 3] for node in robot_nodes], dtype=np.float64)

        axes[0].plot(initial_positions[:, 0], initial_positions[:, 1], marker="o", color=color, label=robot_name)
        axes[1].plot(optimized_positions[:, 0], optimized_positions[:, 1], marker="o", color=color, label=robot_name)

    axes[0].set_title("Before Optimization")
    axes[1].set_title("After Optimization")
    for axis in axes:
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.grid(True, alpha=0.3)
        axis.legend()

    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(output_path), dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_edge_metrics_plot(edge_metrics: Sequence[dict[str, object]], output_path: str | Path) -> Path:
    """Save bar charts for final edge fitness and RMSE."""

    if not edge_metrics:
        raise ValueError("edge_metrics must not be empty.")

    labels = [f"{item['source_key']}->{item['target_key']}" for item in edge_metrics]
    fitness_values = [float(item["final_fitness"]) for item in edge_metrics]
    rmse_values = [float(item["final_rmse"]) for item in edge_metrics]
    colors = ["#188038" if bool(item["is_valid"]) else "#d93025" for item in edge_metrics]

    figure, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].bar(range(len(edge_metrics)), fitness_values, color=colors)
    axes[0].set_ylabel("fitness")
    axes[0].set_title("Final Edge Quality")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(range(len(edge_metrics)), rmse_values, color=colors)
    axes[1].set_ylabel("inlier RMSE")
    axes[1].set_xticks(range(len(edge_metrics)))
    axes[1].set_xticklabels(labels, rotation=60, ha="right")
    axes[1].grid(True, axis="y", alpha=0.3)

    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(output_path), dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_keyframe_count_plot(keyframe_counts: dict[str, int], output_path: str | Path) -> Path:
    """Save a bar chart of keyframe counts per robot."""

    robot_names = list(keyframe_counts.keys())
    counts = [keyframe_counts[name] for name in robot_names]

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(robot_names, counts, color=["#1a73e8", "#d93025", "#188038"][: len(robot_names)])
    axis.set_title("Keyframe Count Per Robot")
    axis.set_ylabel("count")
    axis.grid(True, axis="y", alpha=0.3)

    for index, count in enumerate(counts):
        axis.text(index, count, str(count), ha="center", va="bottom")

    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(output_path), dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _sample_pointcloud_points(pointcloud: o3d.geometry.PointCloud, max_points: int = 30000) -> np.ndarray:
    """Convert point cloud to a bounded numpy array for plotting."""

    points = np.asarray(pointcloud.points)
    if len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
    return points[indices]


def save_final_map_plot(pointcloud_path: str | Path, mesh_path: str | Path, output_path: str | Path) -> Path:
    """Save a figure showing the fused point cloud and mesh."""

    pointcloud = load_pcd(pointcloud_path)
    mesh = load_mesh(mesh_path)
    pointcloud_points = _sample_pointcloud_points(pointcloud, max_points=25000)
    mesh_points = np.asarray(mesh.vertices)
    if len(mesh_points) > 25000:
        mesh_points = mesh_points[np.linspace(0, len(mesh_points) - 1, 25000, dtype=np.int64)]

    figure = plt.figure(figsize=(12, 5))
    axis_pointcloud = figure.add_subplot(1, 2, 1, projection="3d")
    axis_mesh = figure.add_subplot(1, 2, 2, projection="3d")

    axis_pointcloud.scatter(
        pointcloud_points[:, 0],
        pointcloud_points[:, 1],
        pointcloud_points[:, 2],
        s=0.2,
        c="#1a73e8",
    )
    axis_pointcloud.set_title("Fused Point Cloud")
    _set_equal_3d_axes(axis_pointcloud, pointcloud_points)

    axis_mesh.scatter(mesh_points[:, 0], mesh_points[:, 1], mesh_points[:, 2], s=0.2, c="#d93025")
    axis_mesh.set_title("Fused Mesh")
    _set_equal_3d_axes(axis_mesh, mesh_points)

    for axis in (axis_pointcloud, axis_mesh):
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")

    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path
