from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


def _as_matrix4x4(matrix: np.ndarray | Iterable[Iterable[float]], name: str) -> np.ndarray:
    """Convert arbitrary matrix-like input to a validated 4x4 float64 pose matrix."""

    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {array.shape}.")
    return array


def _as_information6x6(matrix: np.ndarray | Iterable[Iterable[float]], name: str) -> np.ndarray:
    """Convert arbitrary matrix-like input to a validated 6x6 information matrix."""

    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != (6, 6):
        raise ValueError(f"{name} must have shape (6, 6), got {array.shape}.")
    return array


@dataclass
class PoseRecord:
    """Rigid pose associated with a frame or fragment."""

    robot_name: str
    frame_index: int
    timestamp: float
    matrix: np.ndarray

    def __post_init__(self) -> None:
        self.matrix = _as_matrix4x4(self.matrix, "PoseRecord.matrix")


@dataclass
class FrameRecord:
    """Raw RGB-D frame and its pose metadata."""

    robot_name: str
    frame_index: int
    timestamp: float
    rgb_path: Path
    depth_path: Path
    pose: PoseRecord

    def __post_init__(self) -> None:
        self.rgb_path = Path(self.rgb_path)
        self.depth_path = Path(self.depth_path)

        if self.pose.robot_name != self.robot_name:
            raise ValueError("FrameRecord.robot_name must match pose.robot_name.")
        if self.pose.frame_index != self.frame_index:
            raise ValueError("FrameRecord.frame_index must match pose.frame_index.")


@dataclass
class KeyframeRecord:
    """Selected keyframe used for registration and fragment construction."""

    keyframe_index: int
    frame: FrameRecord
    downsampled_pose: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    pointcloud_path: Path | None = None

    def __post_init__(self) -> None:
        self.downsampled_pose = _as_matrix4x4(self.downsampled_pose, "KeyframeRecord.downsampled_pose")
        if self.pointcloud_path is not None:
            self.pointcloud_path = Path(self.pointcloud_path)

    @property
    def robot_name(self) -> str:
        return self.frame.robot_name

    @property
    def frame_index(self) -> int:
        return self.frame.frame_index

    @property
    def timestamp(self) -> float:
        return self.frame.timestamp

    @property
    def key(self) -> str:
        return f"{self.robot_name}:{self.keyframe_index}"


@dataclass
class RegistrationEdge:
    """Pose-graph edge between two keyframes or fragments."""

    source_key: str
    target_key: str
    transformation: np.ndarray
    information: np.ndarray = field(default_factory=lambda: np.eye(6, dtype=np.float64))
    fitness: float = 0.0
    rmse: float = 0.0
    is_loop_closure: bool = False
    is_cross_robot: bool = False

    def __post_init__(self) -> None:
        self.transformation = _as_matrix4x4(self.transformation, "RegistrationEdge.transformation")
        self.information = _as_information6x6(self.information, "RegistrationEdge.information")


@dataclass
class FragmentRecord:
    """Intermediate local reconstruction result before global fusion."""

    robot_name: str
    fragment_index: int
    keyframes: list[KeyframeRecord]
    pose_graph_edges: list[RegistrationEdge] = field(default_factory=list)
    global_pose: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    pointcloud_path: Path | None = None

    def __post_init__(self) -> None:
        self.global_pose = _as_matrix4x4(self.global_pose, "FragmentRecord.global_pose")
        if self.pointcloud_path is not None:
            self.pointcloud_path = Path(self.pointcloud_path)

        for keyframe in self.keyframes:
            if keyframe.robot_name != self.robot_name:
                raise ValueError("All keyframes in FragmentRecord must belong to the same robot.")

    @property
    def key(self) -> str:
        return f"{self.robot_name}:fragment:{self.fragment_index}"


def build_identity_pose(robot_name: str, frame_index: int, timestamp: float) -> PoseRecord:
    """Helper for tests and small examples."""

    return PoseRecord(
        robot_name=robot_name,
        frame_index=frame_index,
        timestamp=timestamp,
        matrix=np.eye(4, dtype=np.float64),
    )


def main() -> None:
    """Simple example showing how downstream modules should instantiate records."""

    pose = build_identity_pose(robot_name="tb3_1", frame_index=0, timestamp=0.0)
    frame = FrameRecord(
        robot_name="tb3_1",
        frame_index=0,
        timestamp=0.0,
        rgb_path=Path("data/tb3_1/rgb/000000.png"),
        depth_path=Path("data/tb3_1/depth/000000.png"),
        pose=pose,
    )
    keyframe = KeyframeRecord(keyframe_index=0, frame=frame)
    fragment = FragmentRecord(robot_name="tb3_1", fragment_index=0, keyframes=[keyframe])

    print("Frame:", frame.rgb_path.name, frame.depth_path.name)
    print("Keyframe key:", keyframe.key)
    print("Fragment key:", fragment.key)


if __name__ == "__main__":
    main()
