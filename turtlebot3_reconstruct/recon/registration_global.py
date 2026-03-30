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
from recon.io_utils import ProjectPaths, load_pcd


@dataclass
class GlobalRegistrationPreprocessResult:
    """Preprocessed point cloud and its FPFH features."""

    pointcloud: o3d.geometry.PointCloud
    fpfh: o3d.pipelines.registration.Feature
    voxel_size: float


@dataclass
class GlobalRegistrationResult:
    """Coarse registration output used to initialize local refinement."""

    transformation: np.ndarray
    fitness: float
    inlier_rmse: float
    correspondence_set_size: int
    is_valid: bool
    method: str

    @property
    def t_init(self) -> np.ndarray:
        """Alias used by downstream local registration."""

        return self.transformation


def _copy_pointcloud(pointcloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Create a detached point cloud copy."""

    copied = o3d.geometry.PointCloud()
    copied.points = o3d.utility.Vector3dVector(np.asarray(pointcloud.points))
    if pointcloud.has_colors():
        copied.colors = o3d.utility.Vector3dVector(np.asarray(pointcloud.colors))
    if pointcloud.has_normals():
        copied.normals = o3d.utility.Vector3dVector(np.asarray(pointcloud.normals))
    return copied


def preprocess_pointcloud_for_global_registration(
    pointcloud: o3d.geometry.PointCloud,
    config: Config,
    voxel_size: float | None = None,
) -> GlobalRegistrationPreprocessResult:
    """Downsample a point cloud, estimate normals, and compute FPFH features."""

    resolved_voxel_size = config.global_registration.voxel_size if voxel_size is None else voxel_size
    downsampled = _copy_pointcloud(pointcloud)
    if resolved_voxel_size > 0.0:
        downsampled = downsampled.voxel_down_sample(resolved_voxel_size)

    if len(downsampled.points) > 0:
        normal_radius = resolved_voxel_size * config.global_registration.normal_radius_factor
        downsampled.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=max(normal_radius, 0.05),
                max_nn=30,
            )
        )

    fpfh_radius = resolved_voxel_size * config.global_registration.fpfh_radius_factor
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        downsampled,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=max(fpfh_radius, 0.1),
            max_nn=100,
        ),
    )
    return GlobalRegistrationPreprocessResult(
        pointcloud=downsampled,
        fpfh=fpfh,
        voxel_size=resolved_voxel_size,
    )


def compute_fpfh_feature(
    pointcloud: o3d.geometry.PointCloud,
    config: Config,
    voxel_size: float | None = None,
) -> o3d.pipelines.registration.Feature:
    """Compute the 33D FPFH descriptor after standard coarse-registration preprocessing."""

    return preprocess_pointcloud_for_global_registration(pointcloud, config, voxel_size=voxel_size).fpfh


def _build_result(
    result: o3d.pipelines.registration.RegistrationResult,
    config: Config,
    method: str,
) -> GlobalRegistrationResult:
    """Convert Open3D registration output into a project dataclass."""

    registration_result = GlobalRegistrationResult(
        transformation=np.asarray(result.transformation, dtype=np.float64),
        fitness=float(result.fitness),
        inlier_rmse=float(result.inlier_rmse),
        correspondence_set_size=len(result.correspondence_set),
        is_valid=False,
        method=method,
    )
    registration_result.is_valid = is_global_registration_acceptable(registration_result, config)
    return registration_result


def is_global_registration_acceptable(
    result: GlobalRegistrationResult,
    config: Config,
    fitness_threshold: float | None = None,
    rmse_threshold: float | None = None,
) -> bool:
    """Filter weak coarse alignment results before passing them to ICP."""

    resolved_fitness_threshold = (
        config.global_registration.fitness_threshold if fitness_threshold is None else fitness_threshold
    )
    resolved_rmse_threshold = (
        config.global_registration.rmse_threshold if rmse_threshold is None else rmse_threshold
    )
    return result.fitness >= resolved_fitness_threshold and result.inlier_rmse <= resolved_rmse_threshold


def run_ransac_global_registration(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    config: Config,
    voxel_size: float | None = None,
) -> GlobalRegistrationResult:
    """Run FPFH + RANSAC global registration to estimate a coarse initial pose."""

    source_prepared = preprocess_pointcloud_for_global_registration(source, config, voxel_size=voxel_size)
    target_prepared = preprocess_pointcloud_for_global_registration(target, config, voxel_size=voxel_size)
    distance_threshold = source_prepared.voxel_size * config.global_registration.ransac_distance_factor

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_prepared.pointcloud,
        target_prepared.pointcloud,
        source_prepared.fpfh,
        target_prepared.fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        config.global_registration.ransac_n,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(
            config.global_registration.ransac_max_iteration,
            config.global_registration.ransac_confidence,
        ),
    )
    return _build_result(result, config, method="ransac_fpfh")


def run_fgr_global_registration(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    config: Config,
    voxel_size: float | None = None,
) -> GlobalRegistrationResult:
    """Run fast global registration to estimate a coarse initial pose."""

    source_prepared = preprocess_pointcloud_for_global_registration(source, config, voxel_size=voxel_size)
    target_prepared = preprocess_pointcloud_for_global_registration(target, config, voxel_size=voxel_size)
    distance_threshold = source_prepared.voxel_size * config.global_registration.fgr_distance_factor

    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        source_prepared.pointcloud,
        target_prepared.pointcloud,
        source_prepared.fpfh,
        target_prepared.fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold
        ),
    )
    return _build_result(result, config, method="fast_global_registration")


def main() -> None:
    """Simple example using one fragment and a synthetically perturbed clone."""

    config_path = PROJECT_ROOT / "config" / "default.yaml"
    config = load_config(config_path)
    paths = ProjectPaths(config)
    fragment_path = paths.stage_file("stage_03_fragments", "fragment_000.ply", config.robots[1])
    if not fragment_path.exists():
        print("Fragment point cloud not found:", fragment_path)
        return

    source = load_pcd(fragment_path)
    target = _copy_pointcloud(source)
    perturbation = np.eye(4, dtype=np.float64)
    perturbation[:3, 3] = np.array([0.05, -0.02, 0.0], dtype=np.float64)
    target.transform(perturbation)

    ransac_result = run_ransac_global_registration(source, target, config)
    fgr_result = run_fgr_global_registration(source, target, config)

    print("RANSAC:", round(ransac_result.fitness, 4), round(ransac_result.inlier_rmse, 4), ransac_result.is_valid)
    print("FGR:", round(fgr_result.fitness, 4), round(fgr_result.inlier_rmse, 4), fgr_result.is_valid)


if __name__ == "__main__":
    main()
