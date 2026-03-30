from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

import cv2
import numpy as np
import open3d as o3d

try:
    from recon.config import Config, load_config
except ModuleNotFoundError:
    from config import Config, load_config


JsonLike = Union[Dict[str, Any], List[Any], str, int, float, bool, None]


@dataclass
class ProjectPaths:
    """Centralized path resolver shared by all reconstruction stages."""

    config: Config
    output_root: Path | None = None

    def __post_init__(self) -> None:
        self.config.data_root = Path(self.config.data_root).expanduser()
        if self.output_root is not None:
            self.output_root = Path(self.output_root).expanduser()

    @property
    def data_root(self) -> Path:
        return self.config.data_root

    @property
    def results_root(self) -> Path:
        if self.output_root is not None:
            return self.output_root
        return self.data_root.parent / "outputs"

    @property
    def dataset_root(self) -> Path:
        return Path(self.config.export.dataset_root).expanduser()

    def robot_root(self, robot_name: str) -> Path:
        return self.data_root / robot_name

    def rgb_dir(self, robot_name: str) -> Path:
        return self.robot_root(robot_name) / "rgb"

    def depth_dir(self, robot_name: str) -> Path:
        return self.robot_root(robot_name) / "depth"

    def pose_dir(self, robot_name: str) -> Path:
        return self.robot_root(robot_name) / "pose"

    def rgb_file(self, robot_name: str, file_name: str) -> Path:
        return self.rgb_dir(robot_name) / file_name

    def depth_file(self, robot_name: str, file_name: str) -> Path:
        return self.depth_dir(robot_name) / file_name

    def pose_file(self, robot_name: str, file_name: str = "poses.csv") -> Path:
        return self.pose_dir(robot_name) / file_name

    def exported_robot_root(self, robot_name: str) -> Path:
        return self.dataset_root / robot_name

    def exported_rgb_dir(self, robot_name: str) -> Path:
        return self.exported_robot_root(robot_name) / "rgb"

    def exported_depth_dir(self, robot_name: str) -> Path:
        return self.exported_robot_root(robot_name) / "depth"

    def exported_rgb_file(self, robot_name: str, file_name: str) -> Path:
        return self.exported_rgb_dir(robot_name) / file_name

    def exported_depth_file(self, robot_name: str, file_name: str) -> Path:
        return self.exported_depth_dir(robot_name) / file_name

    def exported_pose_csv(self, robot_name: str) -> Path:
        return self.exported_robot_root(robot_name) / "poses.csv"

    def exported_camera_intrinsic_json(self, robot_name: str) -> Path:
        return self.exported_robot_root(robot_name) / "camera_intrinsic.json"

    def stage_dir(self, stage_name: str, robot_name: str | None = None) -> Path:
        base_dir = self.results_root / stage_name
        if robot_name is None:
            return base_dir / "global"
        return base_dir / robot_name

    def stage_file(self, stage_name: str, file_name: str, robot_name: str | None = None) -> Path:
        return self.stage_dir(stage_name, robot_name) / file_name

    def log_file(self, stage_name: str, robot_name: str | None = None, file_name: str = "run.log") -> Path:
        return self.stage_file(stage_name, file_name, robot_name)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return its resolved Path object."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_parent_dir(path: str | Path) -> Path:
    """Create the parent directory for a file path."""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


def load_rgb_image(path: str | Path) -> o3d.geometry.Image:
    """Load an RGB image from disk using Open3D."""

    return o3d.io.read_image(str(Path(path)))


def load_depth_image(path: str | Path) -> o3d.geometry.Image:
    """Load a depth image from disk using Open3D."""

    return o3d.io.read_image(str(Path(path)))


def save_image(path: str | Path, image: np.ndarray) -> Path:
    """Save an OpenCV image array to disk."""

    file_path = ensure_parent_dir(path)
    cv2.imwrite(str(file_path), image)
    return file_path


def load_rgbd_image(
    rgb_path: str | Path,
    depth_path: str | Path,
    config: Config,
    convert_rgb_to_intensity: bool = False,
) -> o3d.geometry.RGBDImage:
    """Load and combine RGB and depth images using the shared depth configuration."""

    color = load_rgb_image(rgb_path)
    depth = load_depth_image(depth_path)
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=color,
        depth=depth,
        depth_scale=config.depth.scale,
        depth_trunc=config.depth.truncation_m,
        convert_rgb_to_intensity=convert_rgb_to_intensity,
    )


def _json_default(value: Any) -> Any:
    """Convert project objects into JSON-serializable values."""

    if is_dataclass(value):
        return _json_default(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_default(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_default(item) for item in value]
    return value


def save_json(path: str | Path, payload: JsonLike | Any, indent: int = 2) -> Path:
    """Write JSON content to disk with dataclass and numpy support."""

    file_path = ensure_parent_dir(path)
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(_json_default(payload), file, indent=indent, ensure_ascii=False)
    return file_path


def load_json(path: str | Path) -> Any:
    """Load a JSON file from disk."""

    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_csv(
    path: str | Path,
    rows: Sequence[dict[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> Path:
    """Write a list of row dictionaries to CSV."""

    file_path = ensure_parent_dir(path)
    normalized_rows = [dict(row) for row in rows]
    resolved_fieldnames = list(fieldnames) if fieldnames is not None else []
    if not resolved_fieldnames and normalized_rows:
        resolved_fieldnames = list(normalized_rows[0].keys())

    with file_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=resolved_fieldnames)
        if resolved_fieldnames:
            writer.writeheader()
        for row in normalized_rows:
            writer.writerow({key: _json_default(value) for key, value in row.items()})
    return file_path


def load_csv(path: str | Path) -> list[dict[str, str]]:
    """Load CSV rows as dictionaries."""

    with Path(path).open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return [dict(row) for row in reader]


def save_pcd(path: str | Path, pointcloud: o3d.geometry.PointCloud, write_ascii: bool = False) -> Path:
    """Save a point cloud to PLY or another Open3D-supported format."""

    file_path = ensure_parent_dir(path)
    o3d.io.write_point_cloud(str(file_path), pointcloud, write_ascii=write_ascii)
    return file_path


def load_pcd(path: str | Path) -> o3d.geometry.PointCloud:
    """Load a point cloud from disk."""

    return o3d.io.read_point_cloud(str(Path(path)))


def save_mesh(path: str | Path, mesh: o3d.geometry.TriangleMesh, write_ascii: bool = False) -> Path:
    """Save a triangle mesh to PLY or another Open3D-supported format."""

    file_path = ensure_parent_dir(path)
    o3d.io.write_triangle_mesh(str(file_path), mesh, write_ascii=write_ascii)
    return file_path


def load_mesh(path: str | Path) -> o3d.geometry.TriangleMesh:
    """Load a triangle mesh from disk."""

    return o3d.io.read_triangle_mesh(str(Path(path)))


def save_stage_json(
    paths: ProjectPaths,
    stage_name: str,
    file_name: str,
    payload: JsonLike | Any,
    robot_name: str | None = None,
) -> Path:
    """Save intermediate JSON output under the centralized stage layout."""

    return save_json(paths.stage_file(stage_name, file_name, robot_name), payload)


def save_stage_csv(
    paths: ProjectPaths,
    stage_name: str,
    file_name: str,
    rows: Sequence[dict[str, Any]],
    robot_name: str | None = None,
    fieldnames: Sequence[str] | None = None,
) -> Path:
    """Save intermediate CSV output under the centralized stage layout."""

    return save_csv(paths.stage_file(stage_name, file_name, robot_name), rows, fieldnames=fieldnames)


def save_stage_pcd(
    paths: ProjectPaths,
    stage_name: str,
    file_name: str,
    pointcloud: o3d.geometry.PointCloud,
    robot_name: str | None = None,
    write_ascii: bool = False,
) -> Path:
    """Save intermediate point cloud output under the centralized stage layout."""

    return save_pcd(paths.stage_file(stage_name, file_name, robot_name), pointcloud, write_ascii=write_ascii)


def save_stage_mesh(
    paths: ProjectPaths,
    stage_name: str,
    file_name: str,
    mesh: o3d.geometry.TriangleMesh,
    robot_name: str | None = None,
    write_ascii: bool = False,
) -> Path:
    """Save intermediate mesh output under the centralized stage layout."""

    return save_mesh(paths.stage_file(stage_name, file_name, robot_name), mesh, write_ascii=write_ascii)


def setup_logger(
    name: str,
    log_path: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create a reusable logger with console and optional file handlers."""

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if log_path is not None:
        resolved_log_path = ensure_parent_dir(log_path)
        log_file_str = str(resolved_log_path)
        has_file_handler = any(
            isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", None) == log_file_str
            for handler in logger.handlers
        )
        if not has_file_handler:
            file_handler = logging.FileHandler(log_file_str, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
            logger.addHandler(file_handler)

    return logger


def main() -> None:
    """Simple example showing centralized path management and I/O helpers."""

    config_path = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
    config = load_config(config_path)
    paths = ProjectPaths(config=config)

    logger = setup_logger("recon.io_utils", paths.log_file(stage_name="example"))
    logger.info("Saving example intermediate files.")

    metadata = {
        "robots": config.robots,
        "camera_width": config.camera.width,
        "camera_height": config.camera.height,
    }
    save_stage_json(paths, "example", "metadata.json", metadata)
    save_stage_csv(
        paths,
        "example",
        "robots.csv",
        rows=[{"robot_name": robot_name, "index": index} for index, robot_name in enumerate(config.robots)],
    )

    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
            ],
            dtype=np.float64,
        )
    )
    save_stage_pcd(paths, "example", "sample_cloud.ply", pointcloud)

    mesh = o3d.geometry.TriangleMesh.create_box(width=0.1, height=0.1, depth=0.1)
    mesh.compute_vertex_normals()
    save_stage_mesh(paths, "example", "sample_mesh.ply", mesh)

    print("Example files saved under:", paths.stage_dir("example"))


if __name__ == "__main__":
    main()
