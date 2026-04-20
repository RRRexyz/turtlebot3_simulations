"""Pairwise local-model registration for multi-robot offline reconstruction."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


Matrix4 = Tuple[Tuple[float, float, float, float], ...]


@dataclass(frozen=True)
class PairwiseRegistrationConfig:
    voxel_size: float = 0.05
    max_correspondence_distance: float = 0.15
    fitness_threshold: float = 0.30
    pose_match_radius: float = 0.75
    min_pose_correspondences: int = 8
    overlap_crop_distance: float = 0.30
    min_overlap_points: int = 200
    enable_icp_refinement: bool = True


@dataclass(frozen=True)
class RobotExport:
    robot_name: str
    database: Path
    cloud: Path
    robot_poses: Path
    camera_poses: Path


@dataclass(frozen=True)
class GroundTruthTrajectory:
    robot_name: str
    model_name: str
    timestamps: np.ndarray
    positions: np.ndarray


def identity_transform() -> Matrix4:
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def planar_pose_to_transform(x_coord: float, y_coord: float, yaw: float, z_coord: float = 0.0) -> Matrix4:
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    return (
        (cos_yaw, -sin_yaw, 0.0, x_coord),
        (sin_yaw, cos_yaw, 0.0, y_coord),
        (0.0, 0.0, 1.0, z_coord),
        (0.0, 0.0, 0.0, 1.0),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Register tb3_2/tb3_3 local clouds into tb3_1 frame using exported RTAB-Map artifacts.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "output" / "local_exports_manifest.json",
        help="Manifest produced by export_local_models.py.",
    )
    parser.add_argument(
        "--target",
        default="tb3_1",
        help="Robot name used as registration reference frame.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["tb3_2", "tb3_3"],
        help="Source robot names to align into the target frame.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for registration outputs. Defaults to manifest output_dir.",
    )
    parser.add_argument(
        "--common-bag",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "turtlebot3_central" / "bag" / "common.bag",
        help="Bag containing /gazebo/model_states for ground-truth seeding.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.05,
        help="Voxel size used for cloud preprocessing during registration.",
    )
    parser.add_argument(
        "--max-correspondence-distance",
        type=float,
        default=0.15,
        help="ICP maximum correspondence distance in meters.",
    )
    parser.add_argument(
        "--fitness-threshold",
        type=float,
        default=0.30,
        help="Warn when registration fitness is below this value.",
    )
    parser.add_argument(
        "--pose-match-radius",
        type=float,
        default=0.75,
        help="Maximum pose-to-pose distance used to build a trajectory-based initial guess.",
    )
    parser.add_argument(
        "--min-pose-correspondences",
        type=int,
        default=8,
        help="Minimum matched robot poses required to trust a trajectory seed transform.",
    )
    parser.add_argument(
        "--gt-match-max-dt",
        type=float,
        default=0.05,
        help="Maximum timestamp difference when matching DB poses to /gazebo/model_states.",
    )
    parser.add_argument(
        "--overlap-crop-distance",
        type=float,
        default=0.30,
        help="Distance threshold used to keep only likely overlapping regions for local ICP.",
    )
    parser.add_argument(
        "--min-overlap-points",
        type=int,
        default=200,
        help="Minimum points per cloud required to trust overlap-cropped ICP.",
    )
    parser.add_argument(
        "--disable-icp-refinement",
        action="store_true",
        help="Skip local ICP refinement and keep the coarse transform only.",
    )
    return parser


def load_manifest(manifest_path: Path) -> Tuple[Dict[str, RobotExport], Path]:
    payload = json.loads(manifest_path.read_text())
    exports = {}
    for item in payload["robots"]:
        exports[item["robot_name"]] = RobotExport(
            robot_name=item["robot_name"],
            database=Path(item["database"]),
            cloud=Path(item["cloud"]),
            robot_poses=Path(item["robot_poses"]),
            camera_poses=Path(item["camera_poses"]),
        )
    output_dir = Path(payload["output_dir"])
    return exports, output_dir


def load_pose_positions(path: Path) -> np.ndarray:
    points: List[List[float]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            points.append([float(fields[1]), float(fields[2]), float(fields[3])])
    if not points:
        raise RuntimeError(f"No poses found in {path}")
    return np.asarray(points, dtype=np.float64)


def load_pose_records(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    timestamps: List[float] = []
    points: List[List[float]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            timestamps.append(float(fields[0]))
            points.append([float(fields[1]), float(fields[2]), float(fields[3])])
    if not points:
        raise RuntimeError(f"No poses found in {path}")
    return np.asarray(timestamps, dtype=np.float64), np.asarray(points, dtype=np.float64)


def rigid_transform_from_points(source_points: np.ndarray, target_points: np.ndarray) -> Tuple[np.ndarray, float]:
    if source_points.shape != target_points.shape:
        raise ValueError("Source and target correspondence arrays must have matching shapes.")

    src_centroid = source_points.mean(axis=0)
    tgt_centroid = target_points.mean(axis=0)
    src_centered = source_points - src_centroid
    tgt_centered = target_points - tgt_centroid

    covariance = src_centered.T @ tgt_centered
    u_mat, _, vh_mat = np.linalg.svd(covariance)
    rotation = vh_mat.T @ u_mat.T
    if np.linalg.det(rotation) < 0:
        vh_mat[-1, :] *= -1
        rotation = vh_mat.T @ u_mat.T

    translation = tgt_centroid - rotation @ src_centroid
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation

    aligned = (rotation @ source_points.T).T + translation
    rmse = float(np.sqrt(np.mean(np.sum((aligned - target_points) ** 2, axis=1))))
    return transform, rmse


def resolve_model_name(robot_name: str, available_names: Sequence[str]) -> str:
    if robot_name in available_names:
        return robot_name

    suffix = robot_name.split("_")[-1]
    candidates = [
        name for name in available_names
        if name.endswith(f"_{suffix}") and "turtlebot3" in name
    ]
    if len(candidates) == 1:
        return candidates[0]

    raise KeyError(
        f"Could not resolve Gazebo model name for robot {robot_name!r} from {list(available_names)}"
    )


def load_ground_truth_trajectories(common_bag_path: Path, robot_names: Sequence[str]) -> Dict[str, GroundTruthTrajectory]:
    import rosbag  # type: ignore

    timestamps_by_robot: Dict[str, List[float]] = {name: [] for name in robot_names}
    positions_by_robot: Dict[str, List[List[float]]] = {name: [] for name in robot_names}
    model_name_map: Dict[str, str] = {}

    with rosbag.Bag(str(common_bag_path)) as bag:
        for _, msg, bag_time in bag.read_messages(topics=["/gazebo/model_states"]):
            if not model_name_map:
                for robot_name in robot_names:
                    model_name_map[robot_name] = resolve_model_name(robot_name, msg.name)

            name_to_index = {name: index for index, name in enumerate(msg.name)}
            stamp_sec = float(bag_time.to_sec())
            for robot_name in robot_names:
                model_name = model_name_map[robot_name]
                index = name_to_index[model_name]
                position = msg.pose[index].position
                timestamps_by_robot[robot_name].append(stamp_sec)
                positions_by_robot[robot_name].append([position.x, position.y, position.z])

    trajectories = {}
    for robot_name in robot_names:
        timestamps = np.asarray(timestamps_by_robot[robot_name], dtype=np.float64)
        positions = np.asarray(positions_by_robot[robot_name], dtype=np.float64)
        if timestamps.size == 0:
            raise RuntimeError(f"No /gazebo/model_states data collected for {robot_name}")
        trajectories[robot_name] = GroundTruthTrajectory(
            robot_name=robot_name,
            model_name=model_name_map[robot_name],
            timestamps=timestamps,
            positions=positions,
        )
    return trajectories


def match_positions_by_timestamp(
    query_timestamps: np.ndarray,
    reference_timestamps: np.ndarray,
    reference_positions: np.ndarray,
    max_dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    matched_indices = []
    matched_positions = []
    matched_deltas = []

    for query_index, timestamp in enumerate(query_timestamps):
        insert_index = int(np.searchsorted(reference_timestamps, timestamp))
        candidate_indices = []
        if insert_index < reference_timestamps.size:
            candidate_indices.append(insert_index)
        if insert_index > 0:
            candidate_indices.append(insert_index - 1)
        if not candidate_indices:
            continue

        best_index = min(candidate_indices, key=lambda idx: abs(reference_timestamps[idx] - timestamp))
        delta = abs(reference_timestamps[best_index] - timestamp)
        if delta <= max_dt:
            matched_indices.append(query_index)
            matched_positions.append(reference_positions[best_index])
            matched_deltas.append(delta)

    if not matched_indices:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )

    return (
        np.asarray(matched_indices, dtype=np.int64),
        np.asarray(matched_positions, dtype=np.float64),
        np.asarray(matched_deltas, dtype=np.float64),
    )


def estimate_local_to_world_transform(
    robot_export: RobotExport,
    ground_truth: GroundTruthTrajectory,
    max_dt: float,
    min_correspondences: int,
) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
    local_timestamps, local_positions = load_pose_records(robot_export.robot_poses)
    matched_indices, world_positions, matched_deltas = match_positions_by_timestamp(
        query_timestamps=local_timestamps,
        reference_timestamps=ground_truth.timestamps,
        reference_positions=ground_truth.positions,
        max_dt=max_dt,
    )

    matched_count = int(matched_indices.size)
    stats: Dict[str, float] = {
        "gt_model_name": ground_truth.model_name,
        "gt_matched_pose_count": matched_count,
        "gt_match_max_dt": float(max_dt),
    }
    if matched_count == 0:
        return None, stats

    stats["gt_match_mean_dt"] = float(np.mean(matched_deltas))
    if matched_count < min_correspondences:
        return None, stats

    local_matched = local_positions[matched_indices]
    transform, rmse = rigid_transform_from_points(local_matched, world_positions)
    stats["gt_local_to_world_rmse"] = rmse
    return transform, stats


def estimate_pose_seed(
    source_pose_points: np.ndarray,
    target_pose_points: np.ndarray,
    radius: float,
    min_correspondences: int,
) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
    tree = cKDTree(target_pose_points)
    distances, indices = tree.query(source_pose_points, distance_upper_bound=radius)
    valid_mask = np.isfinite(distances)
    count = int(np.sum(valid_mask))
    stats = {
        "matched_pose_count": count,
        "pose_match_radius": float(radius),
    }
    if count < min_correspondences:
        return None, stats

    source_corr = source_pose_points[valid_mask]
    target_corr = target_pose_points[indices[valid_mask]]
    transform, rmse = rigid_transform_from_points(source_corr, target_corr)
    stats["pose_seed_rmse"] = rmse
    return transform, stats


def preprocess_cloud(path: Path, voxel_size: float):
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise RuntimeError(f"Empty point cloud: {path}")

    down = cloud.voxel_down_sample(voxel_size)
    if down.is_empty():
        raise RuntimeError(f"Downsampled point cloud became empty: {path}")

    normal_radius = max(voxel_size * 2.0, 0.05)
    feature_radius = max(voxel_size * 5.0, 0.1)

    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=feature_radius, max_nn=100),
    )
    return cloud, down, fpfh


def fast_global_registration(
    source_down: o3d.geometry.PointCloud,
    target_down: o3d.geometry.PointCloud,
    source_fpfh,
    target_fpfh,
    voxel_size: float,
):
    max_distance = max(voxel_size * 1.5, 0.075)
    option = o3d.pipelines.registration.FastGlobalRegistrationOption(
        maximum_correspondence_distance=max_distance
    )
    return o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        option,
    )


def refine_with_icp(
    source_down: o3d.geometry.PointCloud,
    target_down: o3d.geometry.PointCloud,
    initial_transform: np.ndarray,
    max_distance: float,
):
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    coarse = o3d.pipelines.registration.registration_icp(
        source_down,
        target_down,
        max_distance,
        initial_transform,
        estimation,
    )
    fine = o3d.pipelines.registration.registration_icp(
        source_down,
        target_down,
        max(max_distance * 0.5, 0.05),
        coarse.transformation,
        estimation,
    )
    return fine


def evaluate_transform(
    source_down: o3d.geometry.PointCloud,
    target_down: o3d.geometry.PointCloud,
    transform: np.ndarray,
    max_distance: float,
):
    return o3d.pipelines.registration.evaluate_registration(
        source_down,
        target_down,
        max_distance,
        transform,
    )


def select_overlap_subclouds(
    source_down: o3d.geometry.PointCloud,
    target_down: o3d.geometry.PointCloud,
    coarse_transform: np.ndarray,
    overlap_distance: float,
    min_overlap_points: int,
) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud, Dict[str, float]]:
    source_points = np.asarray(source_down.points)
    target_points = np.asarray(target_down.points)

    stats: Dict[str, float] = {
        "overlap_crop_distance": float(overlap_distance),
        "source_downsampled_points": int(source_points.shape[0]),
        "target_downsampled_points": int(target_points.shape[0]),
        "min_overlap_points": int(min_overlap_points),
    }

    if source_points.size == 0 or target_points.size == 0:
        stats["overlap_crop_used"] = False
        stats["overlap_source_points"] = int(source_points.shape[0])
        stats["overlap_target_points"] = int(target_points.shape[0])
        return source_down, target_down, stats

    source_transformed = (coarse_transform[:3, :3] @ source_points.T).T + coarse_transform[:3, 3]

    target_tree = cKDTree(target_points)
    source_distances, _ = target_tree.query(source_transformed, distance_upper_bound=overlap_distance)
    source_mask = np.isfinite(source_distances)

    source_tree_in_target_frame = cKDTree(source_transformed)
    target_distances, _ = source_tree_in_target_frame.query(target_points, distance_upper_bound=overlap_distance)
    target_mask = np.isfinite(target_distances)

    overlap_source_count = int(np.sum(source_mask))
    overlap_target_count = int(np.sum(target_mask))
    stats["overlap_source_points"] = overlap_source_count
    stats["overlap_target_points"] = overlap_target_count

    if overlap_source_count < min_overlap_points or overlap_target_count < min_overlap_points:
        stats["overlap_crop_used"] = False
        return source_down, target_down, stats

    source_indices = np.flatnonzero(source_mask).tolist()
    target_indices = np.flatnonzero(target_mask).tolist()
    source_overlap = source_down.select_by_index(source_indices)
    target_overlap = target_down.select_by_index(target_indices)
    stats["overlap_crop_used"] = True
    return source_overlap, target_overlap, stats


def evaluate_pair_with_uniform_protocol(
    source_down: o3d.geometry.PointCloud,
    target_down: o3d.geometry.PointCloud,
    transform: np.ndarray,
    max_distance: float,
    min_overlap_points: int,
) -> Tuple[Optional[float], Optional[float], bool, Dict[str, float]]:
    source_points = np.asarray(source_down.points)
    target_points = np.asarray(target_down.points)

    stats: Dict[str, float] = {
        "evaluation_distance_threshold": float(max_distance),
        "evaluation_min_overlap_points": int(min_overlap_points),
        "evaluation_source_downsampled_points": int(source_points.shape[0]),
        "evaluation_target_downsampled_points": int(target_points.shape[0]),
    }

    if source_points.size == 0 or target_points.size == 0:
        stats["evaluation_overlap_source_points"] = int(source_points.shape[0])
        stats["evaluation_overlap_target_points"] = int(target_points.shape[0])
        stats["evaluation_overlap_used"] = False
        stats["valid_pair"] = False
        return None, None, False, stats

    source_transformed = (transform[:3, :3] @ source_points.T).T + transform[:3, 3]

    target_tree = cKDTree(target_points)
    source_distances, _ = target_tree.query(source_transformed, distance_upper_bound=max_distance)
    source_mask = np.isfinite(source_distances)

    source_tree_in_target_frame = cKDTree(source_transformed)
    target_distances, _ = source_tree_in_target_frame.query(target_points, distance_upper_bound=max_distance)
    target_mask = np.isfinite(target_distances)

    overlap_source_count = int(np.sum(source_mask))
    overlap_target_count = int(np.sum(target_mask))
    stats["evaluation_overlap_source_points"] = overlap_source_count
    stats["evaluation_overlap_target_points"] = overlap_target_count

    if overlap_source_count < min_overlap_points or overlap_target_count < min_overlap_points:
        stats["evaluation_overlap_used"] = False
        stats["valid_pair"] = False
        return None, None, False, stats

    source_indices = np.flatnonzero(source_mask).tolist()
    target_indices = np.flatnonzero(target_mask).tolist()
    source_overlap = source_down.select_by_index(source_indices)
    target_overlap = target_down.select_by_index(target_indices)
    evaluation = evaluate_transform(
        source_down=source_overlap,
        target_down=target_overlap,
        transform=transform,
        max_distance=max_distance,
    )
    stats["evaluation_overlap_used"] = True
    stats["valid_pair"] = True
    stats["evaluation_fitness"] = float(evaluation.fitness)
    stats["evaluation_inlier_rmse"] = float(evaluation.inlier_rmse)
    return float(evaluation.fitness), float(evaluation.inlier_rmse), True, stats


def transform_to_serializable(matrix: np.ndarray) -> List[List[float]]:
    return [[float(value) for value in row] for row in matrix]


def rotation_error_degrees(rotation_delta: np.ndarray) -> float:
    trace_value = float(np.trace(rotation_delta))
    cosine = max(-1.0, min(1.0, 0.5 * (trace_value - 1.0)))
    return float(np.degrees(np.arccos(cosine)))


def compute_relative_transform_error(
    estimated_transform: np.ndarray,
    ground_truth_transform: np.ndarray,
) -> Dict[str, float]:
    delta_transform = np.linalg.inv(ground_truth_transform) @ estimated_transform
    translation_error = float(np.linalg.norm(delta_transform[:3, 3]))
    rotation_error = rotation_error_degrees(delta_transform[:3, :3])
    return {
        "gt_relative_translation_error_m": translation_error,
        "gt_relative_rotation_error_deg": rotation_error,
    }


def save_transformed_cloud(
    source_cloud: o3d.geometry.PointCloud,
    transform: np.ndarray,
    output_path: Path,
):
    source_copy = o3d.geometry.PointCloud(source_cloud)
    source_copy.transform(transform)
    o3d.io.write_point_cloud(str(output_path), source_copy)


def save_merged_cloud(
    target_cloud: o3d.geometry.PointCloud,
    source_cloud: o3d.geometry.PointCloud,
    transform: np.ndarray,
    output_path: Path,
):
    source_copy = o3d.geometry.PointCloud(source_cloud)
    source_copy.transform(transform)
    merged = target_cloud + source_copy
    o3d.io.write_point_cloud(str(output_path), merged)


def register_pair(
    source_export: RobotExport,
    target_export: RobotExport,
    config: PairwiseRegistrationConfig,
    output_dir: Path,
    initial_transform_override: Optional[np.ndarray] = None,
    initial_transform_stats: Optional[Dict[str, float]] = None,
    ground_truth_transform: Optional[np.ndarray] = None,
):
    source_pose_points = load_pose_positions(source_export.robot_poses)
    target_pose_points = load_pose_positions(target_export.robot_poses)

    source_cloud, source_down, source_fpfh = preprocess_cloud(source_export.cloud, config.voxel_size)
    target_cloud, target_down, target_fpfh = preprocess_cloud(target_export.cloud, config.voxel_size)

    pose_stats: Dict[str, float] = {}
    if initial_transform_stats:
        pose_stats.update(initial_transform_stats)

    if initial_transform_override is not None:
        coarse_transform = np.asarray(initial_transform_override, dtype=np.float64)
        coarse_method = "ground_truth_seed"
    else:
        pose_seed, pose_seed_stats = estimate_pose_seed(
            source_pose_points=source_pose_points,
            target_pose_points=target_pose_points,
            radius=config.pose_match_radius,
            min_correspondences=config.min_pose_correspondences,
        )
        pose_stats.update(pose_seed_stats)

        if pose_seed is not None:
            coarse_transform = pose_seed
            coarse_method = "pose_seed"
        else:
            global_result = fast_global_registration(
                source_down=source_down,
                target_down=target_down,
                source_fpfh=source_fpfh,
                target_fpfh=target_fpfh,
                voxel_size=config.voxel_size,
            )
            coarse_transform = np.asarray(global_result.transformation)
            coarse_method = "fpfh_global"
            pose_stats["global_fitness"] = float(global_result.fitness)
            pose_stats["global_rmse"] = float(global_result.inlier_rmse)

    icp_source_down, icp_target_down, overlap_stats = select_overlap_subclouds(
        source_down=source_down,
        target_down=target_down,
        coarse_transform=coarse_transform,
        overlap_distance=config.overlap_crop_distance,
        min_overlap_points=config.min_overlap_points,
    )
    pose_stats.update(overlap_stats)

    initial_eval = evaluate_transform(
        source_down=icp_source_down,
        target_down=icp_target_down,
        transform=coarse_transform,
        max_distance=config.max_correspondence_distance,
    )
    pose_stats["initial_fitness"] = float(initial_eval.fitness)
    pose_stats["initial_inlier_rmse"] = float(initial_eval.inlier_rmse)

    if not config.enable_icp_refinement:
        transform = coarse_transform
        optimization_fitness = float(initial_eval.fitness)
        optimization_rmse = float(initial_eval.inlier_rmse)
        refinement_used = False
    else:
        icp_result = refine_with_icp(
            source_down=icp_source_down,
            target_down=icp_target_down,
            initial_transform=coarse_transform,
            max_distance=config.max_correspondence_distance,
        )

        if (
            float(icp_result.fitness) > float(initial_eval.fitness) + 1e-6
            or (
                abs(float(icp_result.fitness) - float(initial_eval.fitness)) <= 1e-6
                and float(icp_result.inlier_rmse) < float(initial_eval.inlier_rmse)
            )
        ):
            transform = np.asarray(icp_result.transformation)
            optimization_fitness = float(icp_result.fitness)
            optimization_rmse = float(icp_result.inlier_rmse)
            refinement_used = True
        else:
            transform = coarse_transform
            optimization_fitness = float(initial_eval.fitness)
            optimization_rmse = float(initial_eval.inlier_rmse)
            refinement_used = False

    evaluation_fitness, evaluation_rmse, valid_pair, evaluation_stats = evaluate_pair_with_uniform_protocol(
        source_down=source_down,
        target_down=target_down,
        transform=transform,
        max_distance=config.max_correspondence_distance,
        min_overlap_points=config.min_overlap_points,
    )
    pose_stats.update(evaluation_stats)
    if ground_truth_transform is not None:
        pose_stats["gt_relative_transform_available"] = True
        pose_stats.update(compute_relative_transform_error(transform, ground_truth_transform))
    else:
        pose_stats["gt_relative_transform_available"] = False

    pair_prefix = f"{source_export.robot_name}_to_{target_export.robot_name}"
    transformed_cloud_path = output_dir / f"{pair_prefix}_cloud.ply"
    merged_cloud_path = output_dir / f"{pair_prefix}_merged.ply"
    summary_path = output_dir / f"{pair_prefix}_registration.json"

    save_transformed_cloud(source_cloud, transform, transformed_cloud_path)
    save_merged_cloud(target_cloud, source_cloud, transform, merged_cloud_path)

    summary = {
        "source_robot": source_export.robot_name,
        "target_robot": target_export.robot_name,
        "source_cloud": str(source_export.cloud),
        "target_cloud": str(target_export.cloud),
        "source_robot_poses": str(source_export.robot_poses),
        "target_robot_poses": str(target_export.robot_poses),
        "coarse_method": coarse_method,
        "coarse_transform": transform_to_serializable(coarse_transform),
        "transform": transform_to_serializable(transform),
        "ground_truth_transform": (
            transform_to_serializable(ground_truth_transform)
            if ground_truth_transform is not None
            else None
        ),
        "fitness": evaluation_fitness,
        "inlier_rmse": evaluation_rmse,
        "optimization_fitness": optimization_fitness,
        "optimization_inlier_rmse": optimization_rmse,
        "valid_pair": valid_pair,
        "refinement_used": refinement_used,
        "transformed_cloud": str(transformed_cloud_path),
        "merged_cloud": str(merged_cloud_path),
        "voxel_size": config.voxel_size,
        "max_correspondence_distance": config.max_correspondence_distance,
        "overlap_crop_distance": config.overlap_crop_distance,
        "min_overlap_points": config.min_overlap_points,
        "enable_icp_refinement": config.enable_icp_refinement,
        "fit_threshold_ok": bool(
            valid_pair
            and evaluation_fitness is not None
            and evaluation_fitness >= config.fitness_threshold
        ),
    }
    summary.update(pose_stats)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def run_registration(
    manifest_path: Path,
    target_robot: str,
    source_robots: Sequence[str],
    output_dir: Optional[Path],
    config: PairwiseRegistrationConfig,
    common_bag_path: Optional[Path] = None,
    gt_match_max_dt: float = 0.05,
    use_ground_truth_seed: bool = True,
):
    exports, manifest_output_dir = load_manifest(manifest_path)
    if target_robot not in exports:
        raise KeyError(f"Target robot {target_robot!r} not found in manifest.")

    destination = (output_dir or manifest_output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    target_export = exports[target_robot]
    local_to_world_transforms: Dict[str, np.ndarray] = {}
    gt_stats_by_robot: Dict[str, Dict[str, float]] = {}
    robot_names_for_gt = [target_robot] + [name for name in source_robots if name != target_robot]
    if common_bag_path is not None and common_bag_path.exists():
        trajectories = load_ground_truth_trajectories(common_bag_path, robot_names_for_gt)
        for robot_name in robot_names_for_gt:
            transform, stats = estimate_local_to_world_transform(
                robot_export=exports[robot_name],
                ground_truth=trajectories[robot_name],
                max_dt=gt_match_max_dt,
                min_correspondences=config.min_pose_correspondences,
            )
            gt_stats_by_robot[robot_name] = stats
            if transform is not None:
                local_to_world_transforms[robot_name] = transform

    results = {
        "manifest": str(manifest_path.resolve()),
        "target_robot": target_robot,
        "common_bag": str(common_bag_path.resolve()) if common_bag_path is not None and common_bag_path.exists() else None,
        "sources": [],
        "config": {
            "voxel_size": config.voxel_size,
            "max_correspondence_distance": config.max_correspondence_distance,
            "fitness_threshold": config.fitness_threshold,
            "pose_match_radius": config.pose_match_radius,
            "min_pose_correspondences": config.min_pose_correspondences,
            "overlap_crop_distance": config.overlap_crop_distance,
            "min_overlap_points": config.min_overlap_points,
            "enable_icp_refinement": config.enable_icp_refinement,
            "gt_match_max_dt": gt_match_max_dt,
            "use_ground_truth_seed": use_ground_truth_seed,
        },
        "ground_truth_alignment": {
            robot_name: {
                **stats,
                **(
                    {"local_to_world_transform": transform_to_serializable(local_to_world_transforms[robot_name])}
                    if robot_name in local_to_world_transforms
                    else {}
                ),
            }
            for robot_name, stats in gt_stats_by_robot.items()
        },
    }

    for source_robot in source_robots:
        if source_robot not in exports:
            raise KeyError(f"Source robot {source_robot!r} not found in manifest.")
        print(f"[register] {source_robot} -> {target_robot}")
        initial_transform = None
        initial_stats: Dict[str, float] = {}
        ground_truth_transform = None
        if (
            source_robot in local_to_world_transforms
            and target_robot in local_to_world_transforms
        ):
            ground_truth_transform = (
                np.linalg.inv(local_to_world_transforms[target_robot])
                @ local_to_world_transforms[source_robot]
            )
            initial_stats = {
                "gt_seed_available": 1.0,
                "source_gt_matched_pose_count": gt_stats_by_robot[source_robot]["gt_matched_pose_count"],
                "target_gt_matched_pose_count": gt_stats_by_robot[target_robot]["gt_matched_pose_count"],
                "source_gt_local_to_world_rmse": gt_stats_by_robot[source_robot].get("gt_local_to_world_rmse", float("nan")),
                "target_gt_local_to_world_rmse": gt_stats_by_robot[target_robot].get("gt_local_to_world_rmse", float("nan")),
            }
            if use_ground_truth_seed:
                initial_transform = ground_truth_transform
        else:
            initial_stats["gt_seed_available"] = 0.0
        summary = register_pair(
            source_export=exports[source_robot],
            target_export=target_export,
            config=config,
            output_dir=destination,
            initial_transform_override=initial_transform,
            initial_transform_stats=initial_stats,
            ground_truth_transform=ground_truth_transform,
        )
        results["sources"].append(summary)
        if not summary["fit_threshold_ok"]:
            fitness_text = (
                f"{summary['fitness']:.4f}"
                if summary.get("fitness") is not None
                else "N/A"
            )
            reason_text = (
                "has no valid overlap under the uniform evaluation protocol"
                if not bool(summary.get("valid_pair", False))
                else f"is below threshold {config.fitness_threshold:.4f}"
            )
            print(
                f"[warn] {source_robot} -> {target_robot} fitness={fitness_text} {reason_text}",
                file=sys.stderr,
            )

    results_path = destination / f"{target_robot}_local_registration_summary.json"
    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(f"[done] wrote registration summary: {results_path}")
    return results_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = PairwiseRegistrationConfig(
        voxel_size=args.voxel_size,
        max_correspondence_distance=args.max_correspondence_distance,
        fitness_threshold=args.fitness_threshold,
        pose_match_radius=args.pose_match_radius,
        min_pose_correspondences=args.min_pose_correspondences,
        overlap_crop_distance=args.overlap_crop_distance,
        min_overlap_points=args.min_overlap_points,
        enable_icp_refinement=not args.disable_icp_refinement,
    )
    run_registration(
        manifest_path=args.manifest.expanduser().resolve(),
        target_robot=args.target,
        source_robots=args.sources,
        output_dir=args.output_dir.expanduser().resolve() if args.output_dir else None,
        config=config,
        common_bag_path=args.common_bag.expanduser().resolve() if args.common_bag else None,
        gt_match_max_dt=args.gt_match_max_dt,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
