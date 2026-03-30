from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv_bridge
import numpy as np
import rosbag

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recon.config import Config, ResolvedExportTopicConfig, load_config
from recon.io_utils import (
    ProjectPaths,
    ensure_dir,
    save_csv,
    save_image,
    save_json,
    setup_logger,
)


@dataclass
class CameraIntrinsicRecord:
    """Serialized camera intrinsics saved next to exported frames."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    intrinsic_matrix: list[float]


@dataclass
class PoseCsvRecord:
    """Single pose row stored in poses.csv."""

    frame_index: int
    timestamp: float
    timestamp_ns: int
    rgb_file: str
    depth_file: str
    depth_timestamp: float
    depth_timestamp_ns: int
    pose_timestamp: float
    pose_timestamp_ns: int
    tx: float
    ty: float
    tz: float
    qx: float
    qy: float
    qz: float
    qw: float


@dataclass
class RobotExportSummary:
    """Lightweight export statistics for logging."""

    robot_name: str
    bag_path: Path
    rgb_count: int = 0
    depth_count: int = 0
    pose_count: int = 0


@dataclass
class TimestampRecord:
    """Normalized timestamp used for file naming and CSV indexing."""

    seconds: float
    nanoseconds: int


@dataclass
class DepthFrameRecord:
    """Latest depth frame kept in memory for RGB anchoring."""

    timestamp: TimestampRecord
    image: np.ndarray


@dataclass
class PoseSampleRecord:
    """Latest odometry sample kept in memory for RGB anchoring."""

    timestamp: TimestampRecord
    tx: float
    ty: float
    tz: float
    qx: float
    qy: float
    qz: float
    qw: float


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the export stage."""

    parser = argparse.ArgumentParser(description="Export RGB, depth, and odometry from rosbag files.")
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
        help="Optional subset of robots to export. Defaults to all robots in config.",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=None,
        help="Optional debug limit per topic for faster verification.",
    )
    return parser.parse_args()


def _timestamp_file_name(timestamp_ns: int, image_format: str) -> str:
    """Generate a stable file name from a timestamp key."""

    return f"{timestamp_ns}.{image_format}"


def _resolve_timestamp(message: object, bag_time: object) -> TimestampRecord:
    """Use message header time when available; otherwise fall back to rosbag time."""

    header = getattr(message, "header", None)
    if header is not None and hasattr(header, "stamp"):
        stamp = header.stamp
        if hasattr(stamp, "to_nsec"):
            timestamp_ns = int(stamp.to_nsec())
            if timestamp_ns > 0:
                return TimestampRecord(seconds=timestamp_ns / 1e9, nanoseconds=timestamp_ns)

        timestamp_seconds = float(stamp.to_sec())
        if timestamp_seconds > 0.0:
            return TimestampRecord(
                seconds=timestamp_seconds,
                nanoseconds=int(round(timestamp_seconds * 1e9)),
            )

    fallback_seconds = float(bag_time.to_sec())
    return TimestampRecord(
        seconds=fallback_seconds,
        nanoseconds=int(round(fallback_seconds * 1e9)),
    )


def _depth_to_uint16(depth_image: np.ndarray, depth_scale: float) -> np.ndarray:
    """Normalize depth images into uint16 millimeter-like storage."""

    if depth_image.dtype == np.uint16:
        return depth_image

    depth_float = np.asarray(depth_image, dtype=np.float32)
    depth_float = np.nan_to_num(depth_float, nan=0.0, posinf=0.0, neginf=0.0)
    scaled_depth = np.clip(depth_float * depth_scale, 0.0, float(np.iinfo(np.uint16).max))
    return scaled_depth.astype(np.uint16)


def _camera_info_to_record(message: object) -> CameraIntrinsicRecord:
    """Convert ROS CameraInfo to a JSON-friendly dataclass."""

    return CameraIntrinsicRecord(
        width=int(message.width),
        height=int(message.height),
        fx=float(message.K[0]),
        fy=float(message.K[4]),
        cx=float(message.K[2]),
        cy=float(message.K[5]),
        intrinsic_matrix=[float(value) for value in message.K],
    )


def _camera_record_from_config(config: Config) -> CameraIntrinsicRecord:
    """Fallback intrinsic record when the bag lacks CameraInfo."""

    return CameraIntrinsicRecord(
        width=config.camera.width,
        height=config.camera.height,
        fx=config.camera.fx,
        fy=config.camera.fy,
        cx=config.camera.cx,
        cy=config.camera.cy,
        intrinsic_matrix=[
            config.camera.fx,
            0.0,
            config.camera.cx,
            0.0,
            config.camera.fy,
            config.camera.cy,
            0.0,
            0.0,
            1.0,
        ],
    )


def _odom_to_pose_sample(message: object, bag_time: object) -> PoseSampleRecord:
    """Convert nav_msgs/Odometry into an in-memory pose sample."""

    pose = message.pose.pose
    timestamp = _resolve_timestamp(message, bag_time)
    return PoseSampleRecord(
        timestamp=timestamp,
        tx=float(pose.position.x),
        ty=float(pose.position.y),
        tz=float(pose.position.z),
        qx=float(pose.orientation.x),
        qy=float(pose.orientation.y),
        qz=float(pose.orientation.z),
        qw=float(pose.orientation.w),
    )


def _should_stop(exported_count: int, max_messages: int | None) -> bool:
    """Check whether the aligned export limit has been reached."""

    if max_messages is None:
        return False
    return exported_count >= max_messages


def _all_limits_reached(
    summary: RobotExportSummary,
    max_messages: int | None,
    camera_record: CameraIntrinsicRecord | None,
) -> bool:
    """Stop early in debug mode once enough aligned frames have been exported."""

    if max_messages is None:
        return False
    return summary.rgb_count >= max_messages and camera_record is not None


def _topic_list(topics: ResolvedExportTopicConfig) -> list[str]:
    """Build the bag topic filter list once."""

    return [
        topics.rgb_image,
        topics.depth_image,
        topics.odom,
        topics.rgb_camera_info,
        topics.depth_camera_info,
    ]


def _message_type(message: object) -> str:
    """Return the ROS message type string from a bag message."""

    return str(getattr(message, "_type", ""))


def export_robot_bag(
    config: Config,
    paths: ProjectPaths,
    robot_name: str,
    max_messages: int | None,
    logger_name: str,
) -> RobotExportSummary:
    """Export one robot bag into the standardized offline dataset layout."""

    bridge = cv_bridge.CvBridge()
    topics = config.export.resolve_topics(robot_name)
    summary = RobotExportSummary(robot_name=robot_name, bag_path=config.export.bag_path(robot_name))
    logger = setup_logger(logger_name)

    ensure_dir(paths.exported_robot_root(robot_name))
    ensure_dir(paths.exported_rgb_dir(robot_name))
    ensure_dir(paths.exported_depth_dir(robot_name))

    camera_record: CameraIntrinsicRecord | None = None
    pose_rows: list[PoseCsvRecord] = []
    latest_depth: DepthFrameRecord | None = None
    latest_pose: PoseSampleRecord | None = None

    with rosbag.Bag(str(summary.bag_path), "r") as bag:
        for topic, message, bag_time in bag.read_messages(topics=_topic_list(topics)):
            if topic == topics.rgb_image and _message_type(message) == "sensor_msgs/Image":
                if _should_stop(summary.rgb_count, max_messages):
                    continue

                if latest_depth is None or latest_pose is None:
                    continue

                rgb_timestamp = _resolve_timestamp(message, bag_time)
                file_name = _timestamp_file_name(rgb_timestamp.nanoseconds, config.export.image_format)
                color_image = bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
                save_image(
                    paths.exported_rgb_file(robot_name, file_name),
                    color_image,
                )
                save_image(paths.exported_depth_file(robot_name, file_name), latest_depth.image)

                pose_rows.append(
                    PoseCsvRecord(
                        frame_index=summary.rgb_count,
                        timestamp=rgb_timestamp.seconds,
                        timestamp_ns=rgb_timestamp.nanoseconds,
                        rgb_file=file_name,
                        depth_file=file_name,
                        depth_timestamp=latest_depth.timestamp.seconds,
                        depth_timestamp_ns=latest_depth.timestamp.nanoseconds,
                        pose_timestamp=latest_pose.timestamp.seconds,
                        pose_timestamp_ns=latest_pose.timestamp.nanoseconds,
                        tx=latest_pose.tx,
                        ty=latest_pose.ty,
                        tz=latest_pose.tz,
                        qx=latest_pose.qx,
                        qy=latest_pose.qy,
                        qz=latest_pose.qz,
                        qw=latest_pose.qw,
                    )
                )
                summary.rgb_count += 1
                summary.depth_count += 1
                summary.pose_count += 1
                continue

            if topic == topics.depth_image and _message_type(message) == "sensor_msgs/Image":
                depth_timestamp = _resolve_timestamp(message, bag_time)
                depth_image = bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")
                depth_uint16 = _depth_to_uint16(depth_image, config.depth.scale)
                latest_depth = DepthFrameRecord(timestamp=depth_timestamp, image=depth_uint16)
                continue

            if topic == topics.odom and _message_type(message) == "nav_msgs/Odometry":
                latest_pose = _odom_to_pose_sample(message, bag_time)
                continue

            if topic in (topics.rgb_camera_info, topics.depth_camera_info) and _message_type(message) == "sensor_msgs/CameraInfo":
                if camera_record is None:
                    camera_record = _camera_info_to_record(message)

            if _all_limits_reached(summary, max_messages, camera_record):
                break

    if camera_record is None:
        camera_record = _camera_record_from_config(config)

    save_json(paths.exported_camera_intrinsic_json(robot_name), camera_record)
    save_csv(
        paths.exported_pose_csv(robot_name),
        rows=[asdict(row) for row in pose_rows],
        fieldnames=[
            "frame_index",
            "timestamp",
            "timestamp_ns",
            "rgb_file",
            "depth_file",
            "depth_timestamp",
            "depth_timestamp_ns",
            "pose_timestamp",
            "pose_timestamp_ns",
            "tx",
            "ty",
            "tz",
            "qx",
            "qy",
            "qz",
            "qw",
        ],
    )

    logger.info(
        "Exported %s from %s | rgb=%d depth=%d poses=%d",
        robot_name,
        summary.bag_path,
        summary.rgb_count,
        summary.depth_count,
        summary.pose_count,
    )
    return summary


def export_all(
    config: Config,
    robot_names: Iterable[str] | None = None,
    max_messages: int | None = None,
) -> list[RobotExportSummary]:
    """Export all configured robots or a selected subset."""

    paths = ProjectPaths(config=config)
    logger = setup_logger("stage_01_export", paths.results_root / "stage_01_export.log")
    selected_robots = list(robot_names) if robot_names is not None else list(config.robots)

    logger.info("Export target dataset root: %s", paths.dataset_root)
    summaries: list[RobotExportSummary] = []
    for robot_name in selected_robots:
        logger.info("Starting export for %s", robot_name)
        summary = export_robot_bag(
            config=config,
            paths=paths,
            robot_name=robot_name,
            max_messages=max_messages,
            logger_name="stage_01_export",
        )
        summaries.append(summary)

    return summaries


def main() -> None:
    """CLI entry point with a small summary printout."""

    args = parse_args()
    config = load_config(args.config)
    summaries = export_all(config=config, robot_names=args.robots, max_messages=args.max_messages)

    for summary in summaries:
        print(
            f"{summary.robot_name}: rgb={summary.rgb_count}, "
            f"depth={summary.depth_count}, poses={summary.pose_count}"
        )


if __name__ == "__main__":
    main()
