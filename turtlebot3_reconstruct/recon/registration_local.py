from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recon.config import Config, load_config
from recon.fragment_builder import load_keyframes_from_json
from recon.io_utils import ProjectPaths
from recon.rgbd_utils import build_pointcloud_result, estimate_normals
from recon.types import RegistrationEdge


@dataclass
class LocalRegistrationResult:
    """Unified local registration output for downstream graph construction."""

    transformation: np.ndarray
    fitness: float
    inlier_rmse: float
    information_matrix: np.ndarray
    is_valid: bool
    method: str


def _failed_local_result(transformation: np.ndarray, method: str) -> LocalRegistrationResult:
    """Return a standardized invalid registration result after Open3D failure."""

    return LocalRegistrationResult(
        transformation=np.asarray(transformation, dtype=np.float64),
        fitness=0.0,
        inlier_rmse=float("inf"),
        information_matrix=np.zeros((6, 6), dtype=np.float64),
        is_valid=False,
        method=method,
    )


def _copy_pointcloud(pointcloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Create a detached copy of a point cloud."""

    copied = o3d.geometry.PointCloud()
    copied.points = o3d.utility.Vector3dVector(np.asarray(pointcloud.points))
    if pointcloud.has_colors():
        copied.colors = o3d.utility.Vector3dVector(np.asarray(pointcloud.colors))
    if pointcloud.has_normals():
        copied.normals = o3d.utility.Vector3dVector(np.asarray(pointcloud.normals))
    return copied


def _ensure_normals(pointcloud: o3d.geometry.PointCloud, voxel_size: float) -> o3d.geometry.PointCloud:
    """Estimate normals when a point cloud does not already have them."""

    prepared = _copy_pointcloud(pointcloud)
    if not prepared.has_normals():
        estimate_normals(prepared, voxel_size=voxel_size)
    return prepared


def _information_matrix(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    max_correspondence_distance: float,
    transformation: np.ndarray,
) -> np.ndarray:
    """Compute the 6x6 information matrix for a registration result."""

    return np.asarray(
        o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source,
            target,
            max_correspondence_distance,
            transformation,
        )
    )


def _result_from_open3d(
    result: o3d.pipelines.registration.RegistrationResult,
    information_matrix: np.ndarray,
    is_valid: bool,
    method: str,
) -> LocalRegistrationResult:
    """Normalize Open3D registration results into a dataclass."""

    return LocalRegistrationResult(
        transformation=np.asarray(result.transformation, dtype=np.float64),
        fitness=float(result.fitness),
        inlier_rmse=float(result.inlier_rmse),
        information_matrix=np.asarray(information_matrix, dtype=np.float64),
        is_valid=is_valid,
        method=method,
    )


def is_registration_acceptable(
    result: LocalRegistrationResult,
    config: Config,
    fitness_threshold: float | None = None,
    rmse_threshold: float | None = None,
) -> bool:
    """Filter out weak registration edges using configured thresholds."""

    resolved_fitness_threshold = config.icp.fitness_threshold if fitness_threshold is None else fitness_threshold
    resolved_rmse_threshold = config.icp.rmse_threshold if rmse_threshold is None else rmse_threshold
    return result.fitness >= resolved_fitness_threshold and result.inlier_rmse <= resolved_rmse_threshold


def evaluate_registration_result(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transformation: np.ndarray,
    config: Config,
    max_correspondence_distance: float | None = None,
    method: str = "evaluation",
) -> LocalRegistrationResult:
    """Evaluate a candidate transformation and derive its information matrix."""

    correspondence_distance = (
        config.icp.max_correspondence_distance
        if max_correspondence_distance is None
        else max_correspondence_distance
    )
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source,
        target,
        correspondence_distance,
        transformation,
    )
    information_matrix = _information_matrix(
        source,
        target,
        correspondence_distance,
        np.asarray(transformation, dtype=np.float64),
    )
    provisional_result = LocalRegistrationResult(
        transformation=np.asarray(transformation, dtype=np.float64),
        fitness=float(evaluation.fitness),
        inlier_rmse=float(evaluation.inlier_rmse),
        information_matrix=information_matrix,
        is_valid=False,
        method=method,
    )
    provisional_result.is_valid = is_registration_acceptable(provisional_result, config)
    return provisional_result


def run_point_to_plane_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    initial_transformation: np.ndarray,
    config: Config,
    max_correspondence_distance: float | None = None,
) -> LocalRegistrationResult:
    """Run point-to-plane ICP given a reasonable initial transformation."""

    correspondence_distance = (
        config.icp.max_correspondence_distance
        if max_correspondence_distance is None
        else max_correspondence_distance
    )
    source_prepared = _ensure_normals(source, config.pointcloud_voxel_size)
    target_prepared = _ensure_normals(target, config.pointcloud_voxel_size)
    initial = np.asarray(initial_transformation, dtype=np.float64)
    try:
        result = o3d.pipelines.registration.registration_icp(
            source_prepared,
            target_prepared,
            correspondence_distance,
            initial,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=config.icp.max_iteration),
        )
    except RuntimeError:
        return _failed_local_result(initial, method="point_to_plane_icp")
    information_matrix = _information_matrix(
        source_prepared,
        target_prepared,
        correspondence_distance,
        result.transformation,
    )
    local_result = _result_from_open3d(result, information_matrix, is_valid=False, method="point_to_plane_icp")
    local_result.is_valid = is_registration_acceptable(local_result, config)
    return local_result


def run_colored_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    initial_transformation: np.ndarray,
    config: Config,
    max_correspondence_distance: float | None = None,
) -> LocalRegistrationResult:
    """Run colored ICP given a reasonable initial transformation."""

    correspondence_distance = (
        config.colored_icp.max_correspondence_distance
        if max_correspondence_distance is None
        else max_correspondence_distance
    )
    source_prepared = _ensure_normals(source, config.pointcloud_voxel_size)
    target_prepared = _ensure_normals(target, config.pointcloud_voxel_size)
    initial = np.asarray(initial_transformation, dtype=np.float64)
    try:
        result = o3d.pipelines.registration.registration_colored_icp(
            source_prepared,
            target_prepared,
            correspondence_distance,
            initial,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(
                lambda_geometric=config.colored_icp.lambda_geometric
            ),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=config.colored_icp.max_iteration),
        )
    except RuntimeError:
        return _failed_local_result(initial, method="colored_icp")
    information_matrix = _information_matrix(
        source_prepared,
        target_prepared,
        correspondence_distance,
        result.transformation,
    )
    local_result = _result_from_open3d(result, information_matrix, is_valid=False, method="colored_icp")
    local_result.is_valid = is_registration_acceptable(local_result, config)
    return local_result


def to_registration_edge(
    source_key: str,
    target_key: str,
    result: LocalRegistrationResult,
    is_loop_closure: bool = False,
    is_cross_robot: bool = False,
) -> RegistrationEdge:
    """Convert a registration result into the shared pose-graph edge dataclass."""

    return RegistrationEdge(
        source_key=source_key,
        target_key=target_key,
        transformation=result.transformation,
        information=result.information_matrix,
        fitness=result.fitness,
        rmse=result.inlier_rmse,
        is_loop_closure=is_loop_closure,
        is_cross_robot=is_cross_robot,
    )


def main() -> None:
    """Simple example using neighboring keyframes from stage_02 output."""

    config_path = PROJECT_ROOT / "config" / "default.yaml"
    config = load_config(config_path)
    paths = ProjectPaths(config)
    robot_name = config.robots[1]
    keyframes_path = paths.stage_file("stage_02_keyframes", "keyframes.json", robot_name)
    keyframes = load_keyframes_from_json(keyframes_path)
    if len(keyframes) < 2:
        print("Not enough keyframes for local registration.")
        return

    source_keyframe = keyframes[0]
    target_keyframe = keyframes[1]
    source_cloud = build_pointcloud_result(source_keyframe, config, coordinate_frame="camera").pointcloud
    target_cloud = build_pointcloud_result(target_keyframe, config, coordinate_frame="camera").pointcloud
    initial_transformation = np.linalg.inv(target_keyframe.frame.pose.matrix) @ source_keyframe.frame.pose.matrix

    icp_result = run_point_to_plane_icp(source_cloud, target_cloud, initial_transformation, config)
    colored_result = run_colored_icp(source_cloud, target_cloud, initial_transformation, config)

    print("Robot:", robot_name)
    print("ICP:", round(icp_result.fitness, 4), round(icp_result.inlier_rmse, 4), icp_result.is_valid)
    print("Colored ICP:", round(colored_result.fitness, 4), round(colored_result.inlier_rmse, 4), colored_result.is_valid)


if __name__ == "__main__":
    main()
