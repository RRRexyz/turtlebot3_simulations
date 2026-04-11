"""Shared topic/frame helpers for offline RGB-D reconstruction."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RGBDTopicSet:
    robot_name: str
    rgb_topic: str
    rgb_camera_info_topic: str
    depth_topic: str
    depth_camera_info_topic: str
    odom_topic: str
    base_frame: str
    odom_frame: str
    optical_frame: str


def build_rgbd_topic_set(robot_name: str) -> RGBDTopicSet:
    prefix = f"/{robot_name}"
    return RGBDTopicSet(
        robot_name=robot_name,
        rgb_topic=f"{prefix}/camera/rgb/image_raw",
        rgb_camera_info_topic=f"{prefix}/camera/rgb/camera_info",
        depth_topic=f"{prefix}/camera/depth/image_raw",
        depth_camera_info_topic=f"{prefix}/camera/depth/camera_info",
        odom_topic=f"{prefix}/odom",
        base_frame=f"{robot_name}/base_footprint",
        odom_frame=f"{robot_name}/odom",
        optical_frame="camera_rgb_optical_frame",
    )


def camera_static_transform(robot_name: str):
    """Return the fixed camera transform used when replaying the recorded bags."""
    return {
        "parent_frame": f"{robot_name}/base_footprint",
        "child_frame": "camera_rgb_optical_frame",
        "translation_xyz": (0.069, -0.047, 0.107),
        "rotation_rpy": (-1.57079632679, 0.0, -1.57079632679),
    }
