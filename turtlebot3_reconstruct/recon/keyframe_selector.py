from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recon.config import Config, load_config
from recon.sync_utils import load_synced_frames
from recon.types import FrameRecord, KeyframeRecord


@dataclass
class FrameDelta:
    """Motion delta between two synchronized frames."""

    translation_m: float
    rotation_deg: float
    time_sec: float


@dataclass
class KeyframeSelectionStats:
    """Summary statistics for keyframe selection."""

    raw_frame_count: int
    keyframe_count: int
    compression_ratio: float
    selected_frame_indices: list[int]
    translation_threshold_m: float
    rotation_threshold_deg: float
    time_threshold_sec: float


def _rotation_angle_deg(reference_rotation: np.ndarray, current_rotation: np.ndarray) -> float:
    """Compute relative rotation angle in degrees."""

    relative_rotation = reference_rotation.T @ current_rotation
    trace_value = np.trace(relative_rotation)
    cosine = np.clip((trace_value - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def compute_frame_delta(reference_frame: FrameRecord, current_frame: FrameRecord) -> FrameDelta:
    """Measure translation, rotation, and time difference between two frames."""

    reference_translation = reference_frame.pose.matrix[:3, 3]
    current_translation = current_frame.pose.matrix[:3, 3]
    translation_m = float(np.linalg.norm(current_translation - reference_translation))

    rotation_deg = _rotation_angle_deg(
        reference_frame.pose.matrix[:3, :3],
        current_frame.pose.matrix[:3, :3],
    )
    time_sec = float(current_frame.timestamp - reference_frame.timestamp)
    return FrameDelta(
        translation_m=translation_m,
        rotation_deg=rotation_deg,
        time_sec=time_sec,
    )


def should_select_keyframe(delta: FrameDelta, config: Config) -> bool:
    """Apply threshold-based keyframe rules."""

    return (
        delta.translation_m > config.keyframe.translation_m
        or delta.rotation_deg > config.keyframe.rotation_deg
        or delta.time_sec > config.keyframe.time_sec
    )


def select_keyframes(frames: Sequence[FrameRecord], config: Config) -> list[KeyframeRecord]:
    """Select keyframes relative to the last accepted keyframe."""

    if not frames:
        return []

    keyframes: list[KeyframeRecord] = [
        KeyframeRecord(
            keyframe_index=0,
            frame=frames[0],
            downsampled_pose=frames[0].pose.matrix.copy(),
        )
    ]
    last_keyframe_frame = frames[0]

    for frame in frames[1:-1]:
        delta = compute_frame_delta(last_keyframe_frame, frame)
        if not should_select_keyframe(delta, config):
            continue

        keyframes.append(
            KeyframeRecord(
                keyframe_index=len(keyframes),
                frame=frame,
                downsampled_pose=frame.pose.matrix.copy(),
            )
        )
        last_keyframe_frame = frame

    last_frame = frames[-1]
    if last_frame.frame_index != keyframes[-1].frame_index:
        keyframes.append(
            KeyframeRecord(
                keyframe_index=len(keyframes),
                frame=last_frame,
                downsampled_pose=last_frame.pose.matrix.copy(),
            )
        )

    return keyframes


def summarize_keyframes(frames: Sequence[FrameRecord], keyframes: Sequence[KeyframeRecord], config: Config) -> KeyframeSelectionStats:
    """Build summary statistics for saved reports and plots."""

    raw_frame_count = len(frames)
    keyframe_count = len(keyframes)
    compression_ratio = 0.0
    if raw_frame_count > 0:
        compression_ratio = float(keyframe_count) / float(raw_frame_count)

    return KeyframeSelectionStats(
        raw_frame_count=raw_frame_count,
        keyframe_count=keyframe_count,
        compression_ratio=compression_ratio,
        selected_frame_indices=[keyframe.frame_index for keyframe in keyframes],
        translation_threshold_m=config.keyframe.translation_m,
        rotation_threshold_deg=config.keyframe.rotation_deg,
        time_threshold_sec=config.keyframe.time_sec,
    )


def main() -> None:
    """Simple example for manual verification."""

    config_path = PROJECT_ROOT / "config" / "default.yaml"
    config = load_config(config_path)
    robot_name = config.robots[0]
    frames = load_synced_frames(config, robot_name)
    keyframes = select_keyframes(frames, config)
    stats = summarize_keyframes(frames, keyframes, config)

    print("Robot:", robot_name)
    print("Frames:", stats.raw_frame_count)
    print("Keyframes:", stats.keyframe_count)
    print("Compression ratio:", round(stats.compression_ratio, 3))


if __name__ == "__main__":
    main()
