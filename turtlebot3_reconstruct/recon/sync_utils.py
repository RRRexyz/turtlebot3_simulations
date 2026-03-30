from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recon.config import Config, load_config
from recon.io_utils import ProjectPaths, load_csv
from recon.types import FrameRecord, PoseRecord


@dataclass
class TimestampedPath:
    """File path paired with its capture timestamp."""

    timestamp: float
    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)


@dataclass
class SyncMatch:
    """Nearest-neighbor match result for one auxiliary stream."""

    index: int
    delta_sec: float


def _parse_timestamp_from_stem(path: Path) -> float:
    """Parse timestamp from file stem, supporting nanosecond integer naming."""

    stem = path.stem
    if stem.isdigit():
        return int(stem) / 1e9
    return float(stem)


def list_timestamped_files(directory: str | Path) -> list[TimestampedPath]:
    """Load and sort timestamped image files from a directory."""

    folder = Path(directory)
    records = [TimestampedPath(timestamp=_parse_timestamp_from_stem(path), path=path) for path in folder.iterdir() if path.is_file()]
    return sorted(records, key=lambda record: record.timestamp)


def _quaternion_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert quaternion into a 3x3 rotation matrix."""

    norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm == 0.0:
        return np.eye(3, dtype=np.float64)

    x = qx / norm
    y = qy / norm
    z = qz / norm
    w = qw / norm

    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _pose_matrix_from_row(row: dict[str, str]) -> np.ndarray:
    """Build a homogeneous transform from one pose CSV row."""

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _quaternion_to_matrix(
        qx=float(row["qx"]),
        qy=float(row["qy"]),
        qz=float(row["qz"]),
        qw=float(row["qw"]),
    )
    matrix[0, 3] = float(row["tx"])
    matrix[1, 3] = float(row["ty"])
    matrix[2, 3] = float(row["tz"])
    return matrix


def load_pose_records(robot_name: str, poses_csv_path: str | Path) -> list[PoseRecord]:
    """Load pose samples from the exported poses.csv file."""

    rows = load_csv(poses_csv_path)
    pose_records: list[PoseRecord] = []
    for row_index, row in enumerate(rows):
        pose_timestamp = float(row.get("pose_timestamp", row["timestamp"]))
        pose_records.append(
            PoseRecord(
                robot_name=robot_name,
                frame_index=int(row.get("frame_index", row_index)),
                timestamp=pose_timestamp,
                matrix=_pose_matrix_from_row(row),
            )
        )
    return pose_records


def _nearest_match(target_timestamp: float, candidate_timestamps: Sequence[float]) -> SyncMatch | None:
    """Find the nearest timestamp index and distance."""

    if not candidate_timestamps:
        return None

    insert_index = bisect_left(candidate_timestamps, target_timestamp)
    candidate_indices: list[int] = []
    if insert_index < len(candidate_timestamps):
        candidate_indices.append(insert_index)
    if insert_index > 0:
        candidate_indices.append(insert_index - 1)

    best_index = candidate_indices[0]
    best_delta = abs(candidate_timestamps[best_index] - target_timestamp)
    for candidate_index in candidate_indices[1:]:
        candidate_delta = abs(candidate_timestamps[candidate_index] - target_timestamp)
        if candidate_delta < best_delta:
            best_index = candidate_index
            best_delta = candidate_delta

    return SyncMatch(index=best_index, delta_sec=best_delta)


def sync_frame_records(
    config: Config,
    robot_name: str,
    rgb_records: Sequence[TimestampedPath],
    depth_records: Sequence[TimestampedPath],
    pose_records: Sequence[PoseRecord],
    max_time_diff_sec: float | None = None,
) -> list[FrameRecord]:
    """Synchronize RGB, depth, and pose sequences using RGB as the master clock."""

    threshold = config.sync.max_time_diff_sec if max_time_diff_sec is None else max_time_diff_sec
    sorted_rgb_records = sorted(rgb_records, key=lambda record: record.timestamp)
    sorted_depth_records = sorted(depth_records, key=lambda record: record.timestamp)
    sorted_pose_records = sorted(pose_records, key=lambda record: record.timestamp)

    depth_timestamps = [record.timestamp for record in sorted_depth_records]
    pose_timestamps = [record.timestamp for record in sorted_pose_records]

    synced_frames: list[FrameRecord] = []
    for frame_index, rgb_record in enumerate(sorted_rgb_records):
        depth_match = _nearest_match(rgb_record.timestamp, depth_timestamps)
        pose_match = _nearest_match(rgb_record.timestamp, pose_timestamps)

        if depth_match is None or pose_match is None:
            continue
        if depth_match.delta_sec > threshold or pose_match.delta_sec > threshold:
            continue

        matched_depth = sorted_depth_records[depth_match.index]
        matched_pose = sorted_pose_records[pose_match.index]
        pose = PoseRecord(
            robot_name=robot_name,
            frame_index=frame_index,
            timestamp=matched_pose.timestamp,
            matrix=matched_pose.matrix.copy(),
        )
        synced_frames.append(
            FrameRecord(
                robot_name=robot_name,
                frame_index=frame_index,
                timestamp=rgb_record.timestamp,
                rgb_path=rgb_record.path,
                depth_path=matched_depth.path,
                pose=pose,
            )
        )

    return synced_frames


def load_synced_frames(config: Config, robot_name: str, max_time_diff_sec: float | None = None) -> list[FrameRecord]:
    """Load exported files for one robot and return synchronized FrameRecord objects."""

    paths = ProjectPaths(config=config)
    rgb_records = list_timestamped_files(paths.exported_rgb_dir(robot_name))
    depth_records = list_timestamped_files(paths.exported_depth_dir(robot_name))
    pose_records = load_pose_records(robot_name, paths.exported_pose_csv(robot_name))
    return sync_frame_records(
        config=config,
        robot_name=robot_name,
        rgb_records=rgb_records,
        depth_records=depth_records,
        pose_records=pose_records,
        max_time_diff_sec=max_time_diff_sec,
    )


def main() -> None:
    """Simple example for manual verification against exported test data."""

    config_path = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
    config = load_config(config_path)
    robot_name = config.robots[0]
    frames = load_synced_frames(config=config, robot_name=robot_name)

    print("Robot:", robot_name)
    print("Synced frame count:", len(frames))
    if frames:
        first_frame = frames[0]
        print("First RGB:", first_frame.rgb_path.name)
        print("First depth:", first_frame.depth_path.name)
        print("First pose timestamp:", first_frame.pose.timestamp)


if __name__ == "__main__":
    main()
