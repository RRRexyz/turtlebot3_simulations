#!/usr/bin/env python3

from dataclasses import dataclass, field
from collections import deque
import math
from typing import Deque, List, Optional, Set, Tuple


@dataclass
class Pose2D:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0

    def distance_xy(self, other: "Pose2D") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def copy(self) -> "Pose2D":
        return Pose2D(self.x, self.y, self.yaw)


@dataclass
class RobotConfig:
    robot_id: str
    namespace: str
    map_frame: str
    odom_frame: str
    base_frame: str
    map_topic: str
    move_base_action: str
    move_base_status_topic: str
    move_base_make_plan_service: str
    spawn_pose: Pose2D


@dataclass
class FrontierLocal:
    robot_id: str
    frontier_id: str
    centroid: Pose2D
    size: int
    gain: float
    cells: List[Pose2D] = field(default_factory=list)


@dataclass
class FrontierGlobal:
    frontier_id: str
    centroid: Pose2D
    size: int
    gain: float
    source_robot_ids: Set[str] = field(default_factory=set)
    member_local_ids: List[str] = field(default_factory=list)


@dataclass
class Assignment:
    robot_id: str
    frontier_id: str
    goal_global: Pose2D
    score: float
    created_at: float
    status: str = "PENDING"


@dataclass
class RobotState:
    robot_id: str
    config: RobotConfig
    status: str = "IDLE"
    goal_status: str = "IDLE"
    registration_initialized: bool = False
    pose_map_to_base: Optional[Pose2D] = None
    pose_global_to_base: Optional[Pose2D] = None
    local_frontiers: List[FrontierLocal] = field(default_factory=list)
    last_map_stamp: Optional[object] = None
    last_map_received_at: Optional[object] = None
    last_tf_update: Optional[object] = None
    active_assignment: Optional[Assignment] = None
    active_goal_sent_at: Optional[object] = None
    current_goal_local: Optional[Pose2D] = None
    last_status_code: Optional[int] = None
    last_status_text: str = ""
    goal_timeout_reported: bool = False
    failure_count: int = 0


@dataclass
class ExplorationStopState:
    valid_global_frontier_count: int = 0
    active_goal_count: int = 0
    idle_robot_count: int = 0
    terminal_robot_count: int = 0
    known_cell_count_current: int = 0
    known_cell_count_history: Deque[Tuple[float, int]] = field(default_factory=deque)
    known_cell_growth_in_window: int = 0
    map_growth_ratio_in_window: float = 0.0
    stop_counter: int = 0
    finished: bool = False
    finish_reason: str = ""
    no_valid_frontiers: bool = False
    all_robots_idle_or_terminal: bool = False
    map_growth_below_threshold: bool = False
    runtime_sec: float = 0.0
