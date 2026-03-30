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
from recon.fragment_builder import load_keyframes_from_json
from recon.io_utils import ProjectPaths
from recon.types import KeyframeRecord


@dataclass
class PairCandidate:
    """Registration candidate selected before running coarse or local alignment."""

    source_key: str
    target_key: str
    source_robot_name: str
    target_robot_name: str
    source_frame_index: int
    target_frame_index: int
    initial_transformation: np.ndarray
    translation_distance_m: float
    rotation_difference_deg: float
    overlap_ratio: float
    pair_type: str


@dataclass
class PairSelectionResult:
    """Selected registration candidates grouped by role."""

    odometry_candidates: list[PairCandidate]
    loop_candidates: list[PairCandidate]
    cross_robot_candidates: list[PairCandidate]

    @property
    def total_candidates(self) -> int:
        return len(self.odometry_candidates) + len(self.loop_candidates) + len(self.cross_robot_candidates)


def _rotation_angle_deg(source_rotation: np.ndarray, target_rotation: np.ndarray) -> float:
    """Compute the angular difference between two rotation matrices."""

    relative_rotation = source_rotation.T @ target_rotation
    trace_value = np.trace(relative_rotation)
    cosine = np.clip((trace_value - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _forward_axis(pose_matrix: np.ndarray) -> np.ndarray:
    """Use camera +Z as the viewing direction in world coordinates."""

    axis = pose_matrix[:3, 2].astype(np.float64)
    norm = float(np.linalg.norm(axis))
    if norm == 0.0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return axis / norm


def _translation_distance(source_keyframe: KeyframeRecord, target_keyframe: KeyframeRecord) -> float:
    """Compute world-space translation distance between keyframe poses."""

    source_position = source_keyframe.frame.pose.matrix[:3, 3]
    target_position = target_keyframe.frame.pose.matrix[:3, 3]
    return float(np.linalg.norm(target_position - source_position))


def _rotation_difference_deg(source_keyframe: KeyframeRecord, target_keyframe: KeyframeRecord) -> float:
    """Compute world-space orientation difference between keyframes."""

    return _rotation_angle_deg(
        source_keyframe.frame.pose.matrix[:3, :3],
        target_keyframe.frame.pose.matrix[:3, :3],
    )


def _overlap_ratio(source_keyframe: KeyframeRecord, target_keyframe: KeyframeRecord, config: Config) -> float:
    """Estimate a simple overlap score from distance and viewing alignment."""

    distance = _translation_distance(source_keyframe, target_keyframe)
    max_distance = max(config.cross_robot_candidate.max_translation_m, 1e-6)
    distance_score = max(0.0, 1.0 - (distance / max_distance))

    source_forward = _forward_axis(source_keyframe.frame.pose.matrix)
    target_forward = _forward_axis(target_keyframe.frame.pose.matrix)
    viewing_alignment = max(0.0, float(np.dot(source_forward, target_forward)))
    return distance_score * viewing_alignment


def _initial_transformation(source_keyframe: KeyframeRecord, target_keyframe: KeyframeRecord) -> np.ndarray:
    """Build a source-to-target initial transform from their world poses."""

    return np.linalg.inv(target_keyframe.frame.pose.matrix) @ source_keyframe.frame.pose.matrix


def _make_candidate(source_keyframe: KeyframeRecord, target_keyframe: KeyframeRecord, pair_type: str, overlap_ratio: float) -> PairCandidate:
    """Convert two keyframes into a registration candidate."""

    return PairCandidate(
        source_key=source_keyframe.key,
        target_key=target_keyframe.key,
        source_robot_name=source_keyframe.robot_name,
        target_robot_name=target_keyframe.robot_name,
        source_frame_index=source_keyframe.frame_index,
        target_frame_index=target_keyframe.frame_index,
        initial_transformation=_initial_transformation(source_keyframe, target_keyframe),
        translation_distance_m=_translation_distance(source_keyframe, target_keyframe),
        rotation_difference_deg=_rotation_difference_deg(source_keyframe, target_keyframe),
        overlap_ratio=overlap_ratio,
        pair_type=pair_type,
    )


def group_keyframes_by_robot(keyframes: Sequence[KeyframeRecord]) -> dict[str, list[KeyframeRecord]]:
    """Group and sort keyframes by robot name."""

    grouped: dict[str, list[KeyframeRecord]] = {}
    for keyframe in keyframes:
        grouped.setdefault(keyframe.robot_name, []).append(keyframe)
    for robot_name in grouped:
        grouped[robot_name] = sorted(grouped[robot_name], key=lambda item: item.frame_index)
    return grouped


def select_odometry_candidates(keyframes: Sequence[KeyframeRecord]) -> list[PairCandidate]:
    """Connect only adjacent keyframes within each robot trajectory."""

    grouped = group_keyframes_by_robot(keyframes)
    candidates: list[PairCandidate] = []
    for robot_keyframes in grouped.values():
        for source_keyframe, target_keyframe in zip(robot_keyframes[:-1], robot_keyframes[1:]):
            candidates.append(
                _make_candidate(
                    source_keyframe=source_keyframe,
                    target_keyframe=target_keyframe,
                    pair_type="odometry",
                    overlap_ratio=1.0,
                )
            )
    return candidates


def select_cross_robot_candidates(keyframes: Sequence[KeyframeRecord], config: Config) -> list[PairCandidate]:
    """Filter cross-robot pairs by distance, heading difference, and overlap heuristic."""

    grouped = group_keyframes_by_robot(keyframes)
    robot_names = sorted(grouped.keys())
    candidates: list[PairCandidate] = []

    for source_robot_index, source_robot_name in enumerate(robot_names):
        source_keyframes = grouped[source_robot_name]
        for target_robot_name in robot_names[source_robot_index + 1 :]:
            target_keyframes = grouped[target_robot_name]
            for source_keyframe in source_keyframes:
                for target_keyframe in target_keyframes:
                    translation_distance = _translation_distance(source_keyframe, target_keyframe)
                    if translation_distance > config.cross_robot_candidate.max_translation_m:
                        continue

                    rotation_difference = _rotation_difference_deg(source_keyframe, target_keyframe)
                    if rotation_difference > config.cross_robot_candidate.max_rotation_deg:
                        continue

                    overlap_ratio = _overlap_ratio(source_keyframe, target_keyframe, config)
                    if overlap_ratio < config.cross_robot_candidate.min_overlap_ratio:
                        continue

                    candidates.append(
                        _make_candidate(
                            source_keyframe=source_keyframe,
                            target_keyframe=target_keyframe,
                            pair_type="cross_robot",
                            overlap_ratio=overlap_ratio,
                        )
                    )

    candidates.sort(key=lambda item: (-item.overlap_ratio, item.translation_distance_m, item.rotation_difference_deg))
    return candidates


def select_pair_candidates(keyframes: Sequence[KeyframeRecord], config: Config) -> PairSelectionResult:
    """Select sparse registration candidates from all keyframes."""

    odometry_candidates = select_odometry_candidates(keyframes)
    cross_robot_candidates = select_cross_robot_candidates(keyframes, config)
    return PairSelectionResult(
        odometry_candidates=odometry_candidates,
        loop_candidates=[],
        cross_robot_candidates=cross_robot_candidates,
    )


def load_all_keyframes(config: Config, robot_names: Sequence[str] | None = None) -> list[KeyframeRecord]:
    """Load keyframes for all requested robots from stage_02 output."""

    paths = ProjectPaths(config)
    selected_robots = list(robot_names) if robot_names is not None else list(config.robots)
    keyframes: list[KeyframeRecord] = []
    for robot_name in selected_robots:
        keyframe_json_path = paths.stage_file("stage_02_keyframes", "keyframes.json", robot_name)
        if not keyframe_json_path.exists():
            continue
        keyframes.extend(load_keyframes_from_json(keyframe_json_path))
    return keyframes


def main() -> None:
    """Simple example for manual verification."""

    config_path = PROJECT_ROOT / "config" / "default.yaml"
    config = load_config(config_path)
    keyframes = load_all_keyframes(config)
    result = select_pair_candidates(keyframes, config)

    print("Robots with keyframes:", sorted({keyframe.robot_name for keyframe in keyframes}))
    print("Odometry candidates:", len(result.odometry_candidates))
    print("Loop candidates:", len(result.loop_candidates))
    print("Cross-robot candidates:", len(result.cross_robot_candidates))


if __name__ == "__main__":
    main()
