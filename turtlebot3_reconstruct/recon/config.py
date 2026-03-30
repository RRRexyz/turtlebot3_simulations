from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsics used by RGB-D reconstruction."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class DepthConfig:
    """Depth image conversion settings."""

    scale: float
    truncation_m: float


@dataclass
class KeyframeConfig:
    """Thresholds for deciding whether a frame becomes a keyframe."""

    translation_m: float
    rotation_deg: float
    time_sec: float


@dataclass
class ICPConfig:
    """Generic ICP refinement settings."""

    max_correspondence_distance: float
    max_iteration: int
    fitness_threshold: float
    rmse_threshold: float


@dataclass
class ColoredICPConfig:
    """Colored ICP refinement settings."""

    max_correspondence_distance: float
    max_iteration: int
    lambda_geometric: float


@dataclass
class GlobalRegistrationConfig:
    """Coarse registration settings for FPFH-based initialization."""

    voxel_size: float
    normal_radius_factor: float
    fpfh_radius_factor: float
    ransac_distance_factor: float
    ransac_n: int
    ransac_max_iteration: int
    ransac_confidence: float
    fgr_distance_factor: float
    fitness_threshold: float
    rmse_threshold: float


@dataclass
class CrossRobotCandidateConfig:
    """Filtering thresholds for cross-robot loop closure candidates."""

    max_translation_m: float
    max_rotation_deg: float
    min_overlap_ratio: float


@dataclass
class TSDFConfig:
    """TSDF fusion parameters."""

    voxel_length: float
    sdf_trunc: float
    color_type: str


@dataclass
class SyncConfig:
    """Nearest-neighbor synchronization thresholds."""

    max_time_diff_sec: float


@dataclass
class FragmentConfig:
    """Fragment construction settings."""

    keyframes_per_fragment: int


@dataclass
class PoseGraphOptimizationConfig:
    """Pose graph global optimization settings."""

    max_correspondence_distance: float
    edge_prune_threshold: float
    preference_loop_closure: float
    reference_node: int


@dataclass
class ExportTopicConfig:
    """Topic templates used by the rosbag export stage."""

    rgb_image: str
    depth_image: str
    odom: str
    rgb_camera_info: str
    depth_camera_info: str


@dataclass
class ResolvedExportTopicConfig:
    """Robot-specific topic names produced from templates."""

    rgb_image: str
    depth_image: str
    odom: str
    rgb_camera_info: str
    depth_camera_info: str


@dataclass
class ExportConfig:
    """Settings for exporting rosbag data into the offline dataset layout."""

    dataset_root: Path
    bag_files: dict[str, Path]
    image_format: str
    topics: ExportTopicConfig

    def bag_path(self, robot_name: str) -> Path:
        return Path(self.bag_files[robot_name]).expanduser()

    def resolve_topics(self, robot_name: str) -> ResolvedExportTopicConfig:
        return ResolvedExportTopicConfig(
            rgb_image=self.topics.rgb_image.format(robot_name=robot_name),
            depth_image=self.topics.depth_image.format(robot_name=robot_name),
            odom=self.topics.odom.format(robot_name=robot_name),
            rgb_camera_info=self.topics.rgb_camera_info.format(robot_name=robot_name),
            depth_camera_info=self.topics.depth_camera_info.format(robot_name=robot_name),
        )


@dataclass
class Config:
    """Top-level configuration shared across the project."""

    data_root: Path
    robots: list[str]
    camera: CameraIntrinsics
    depth: DepthConfig
    keyframe: KeyframeConfig
    pointcloud_voxel_size: float
    icp: ICPConfig
    colored_icp: ColoredICPConfig
    global_registration: GlobalRegistrationConfig
    cross_robot_candidate: CrossRobotCandidateConfig
    tsdf: TSDFConfig
    sync: SyncConfig
    fragment: FragmentConfig
    pose_graph_optimization: PoseGraphOptimizationConfig
    export: ExportConfig


def _require_mapping(data: Any, section_name: str) -> dict[str, Any]:
    """Validate that a YAML section is a mapping."""

    if not isinstance(data, dict):
        raise TypeError(f"Section '{section_name}' must be a mapping, got {type(data).__name__}.")
    return data


def _build_camera_intrinsics(data: dict[str, Any]) -> CameraIntrinsics:
    return CameraIntrinsics(
        width=int(data["width"]),
        height=int(data["height"]),
        fx=float(data["fx"]),
        fy=float(data["fy"]),
        cx=float(data["cx"]),
        cy=float(data["cy"]),
    )


def _build_depth_config(data: dict[str, Any]) -> DepthConfig:
    return DepthConfig(
        scale=float(data["scale"]),
        truncation_m=float(data["truncation_m"]),
    )


def _build_keyframe_config(data: dict[str, Any]) -> KeyframeConfig:
    return KeyframeConfig(
        translation_m=float(data["translation_m"]),
        rotation_deg=float(data["rotation_deg"]),
        time_sec=float(data["time_sec"]),
    )


def _build_icp_config(data: dict[str, Any]) -> ICPConfig:
    return ICPConfig(
        max_correspondence_distance=float(data["max_correspondence_distance"]),
        max_iteration=int(data["max_iteration"]),
        fitness_threshold=float(data["fitness_threshold"]),
        rmse_threshold=float(data["rmse_threshold"]),
    )


def _build_colored_icp_config(data: dict[str, Any]) -> ColoredICPConfig:
    return ColoredICPConfig(
        max_correspondence_distance=float(data["max_correspondence_distance"]),
        max_iteration=int(data["max_iteration"]),
        lambda_geometric=float(data["lambda_geometric"]),
    )


def _build_global_registration_config(data: dict[str, Any]) -> GlobalRegistrationConfig:
    return GlobalRegistrationConfig(
        voxel_size=float(data["voxel_size"]),
        normal_radius_factor=float(data["normal_radius_factor"]),
        fpfh_radius_factor=float(data["fpfh_radius_factor"]),
        ransac_distance_factor=float(data["ransac_distance_factor"]),
        ransac_n=int(data["ransac_n"]),
        ransac_max_iteration=int(data["ransac_max_iteration"]),
        ransac_confidence=float(data["ransac_confidence"]),
        fgr_distance_factor=float(data["fgr_distance_factor"]),
        fitness_threshold=float(data["fitness_threshold"]),
        rmse_threshold=float(data["rmse_threshold"]),
    )


def _build_cross_robot_candidate_config(data: dict[str, Any]) -> CrossRobotCandidateConfig:
    return CrossRobotCandidateConfig(
        max_translation_m=float(data["max_translation_m"]),
        max_rotation_deg=float(data["max_rotation_deg"]),
        min_overlap_ratio=float(data["min_overlap_ratio"]),
    )


def _build_tsdf_config(data: dict[str, Any]) -> TSDFConfig:
    return TSDFConfig(
        voxel_length=float(data["voxel_length"]),
        sdf_trunc=float(data["sdf_trunc"]),
        color_type=str(data["color_type"]),
    )


def _build_sync_config(data: dict[str, Any]) -> SyncConfig:
    return SyncConfig(max_time_diff_sec=float(data["max_time_diff_sec"]))


def _build_fragment_config(data: dict[str, Any]) -> FragmentConfig:
    return FragmentConfig(keyframes_per_fragment=int(data["keyframes_per_fragment"]))


def _build_pose_graph_optimization_config(data: dict[str, Any]) -> PoseGraphOptimizationConfig:
    return PoseGraphOptimizationConfig(
        max_correspondence_distance=float(data["max_correspondence_distance"]),
        edge_prune_threshold=float(data["edge_prune_threshold"]),
        preference_loop_closure=float(data["preference_loop_closure"]),
        reference_node=int(data["reference_node"]),
    )


def _build_export_topic_config(data: dict[str, Any]) -> ExportTopicConfig:
    return ExportTopicConfig(
        rgb_image=str(data["rgb_image"]),
        depth_image=str(data["depth_image"]),
        odom=str(data["odom"]),
        rgb_camera_info=str(data["rgb_camera_info"]),
        depth_camera_info=str(data["depth_camera_info"]),
    )


def _build_export_config(data: dict[str, Any]) -> ExportConfig:
    raw_bag_files = _require_mapping(data["bag_files"], "export.bag_files")
    bag_files = {
        str(robot_name): Path(str(bag_path)).expanduser()
        for robot_name, bag_path in raw_bag_files.items()
    }
    return ExportConfig(
        dataset_root=Path(str(data["dataset_root"])).expanduser(),
        bag_files=bag_files,
        image_format=str(data.get("image_format", "png")),
        topics=_build_export_topic_config(_require_mapping(data["topics"], "export.topics")),
    )


def load_config(path: str | Path) -> Config:
    """Load the project configuration from a YAML file."""

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    root = _require_mapping(raw, "root")

    robots = [str(robot_name) for robot_name in root["robots"]]
    return Config(
        data_root=Path(root["data_root"]).expanduser(),
        robots=robots,
        camera=_build_camera_intrinsics(_require_mapping(root["camera"], "camera")),
        depth=_build_depth_config(_require_mapping(root["depth"], "depth")),
        keyframe=_build_keyframe_config(_require_mapping(root["keyframe"], "keyframe")),
        pointcloud_voxel_size=float(root["pointcloud_voxel_size"]),
        icp=_build_icp_config(_require_mapping(root["icp"], "icp")),
        colored_icp=_build_colored_icp_config(_require_mapping(root["colored_icp"], "colored_icp")),
        global_registration=_build_global_registration_config(
            _require_mapping(root["global_registration"], "global_registration")
        ),
        cross_robot_candidate=_build_cross_robot_candidate_config(
            _require_mapping(root["cross_robot_candidate"], "cross_robot_candidate")
        ),
        tsdf=_build_tsdf_config(_require_mapping(root["tsdf"], "tsdf")),
        sync=_build_sync_config(_require_mapping(root["sync"], "sync")),
        fragment=_build_fragment_config(_require_mapping(root["fragment"], "fragment")),
        pose_graph_optimization=_build_pose_graph_optimization_config(
            _require_mapping(root["pose_graph_optimization"], "pose_graph_optimization")
        ),
        export=_build_export_config(_require_mapping(root["export"], "export")),
    )


def main() -> None:
    """Simple example for manual verification."""

    default_path = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
    config = load_config(default_path)
    print("Loaded robots:", config.robots)
    print("Data root:", config.data_root)
    print("Dataset root:", config.export.dataset_root)
    print("Camera size:", config.camera.width, "x", config.camera.height)
    print("Point cloud voxel size:", config.pointcloud_voxel_size)


if __name__ == "__main__":
    main()
