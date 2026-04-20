#!/usr/bin/env python3
"""Automatic metrics recorder for experiment 1."""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import rospy
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import String


class MetricsRecorder:
    def __init__(self):
        rospy.init_node("metrics_recorder")

        self.group_name = rospy.get_param("~group_name", "unknown_group")
        self.robot_namespaces = list(rospy.get_param("~robot_namespaces", ["tb3_1", "tb3_2", "tb3_3"]))
        self.result_root = Path(rospy.get_param("~result_root", self._default_result_root())).expanduser().resolve()
        requested_run_name = str(rospy.get_param("~run_name", "")).strip()
        self.run_name = requested_run_name or time.strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.result_root / self.group_name / self.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.map_growth_samples: List[Dict] = []
        self.frontier_samples: List[Dict] = []
        self.events: List[Dict] = []
        self.robot_trace_samples: List[Dict] = []

        self.path_lengths = {robot: 0.0 for robot in self.robot_namespaces}
        self.last_positions: Dict[str, tuple] = {}

        self.goal_sent_count = Counter()
        self.goal_done_status_count = defaultdict(Counter)
        self.goal_timeout_count = Counter()
        self.blacklist_reason_count = Counter()

        self.start_sim_time: Optional[float] = None
        self.finish_sim_time: Optional[float] = None
        self.finish_reason = ""
        self.assignment_mode = ""
        self.latest_known_cells = 0
        self.latest_resolution = 0.05
        self.latest_known_area = 0.0
        self.latest_raw_frontiers = 0
        self.latest_valid_frontiers = 0
        self.finalize_after_wall_time: Optional[float] = None
        self.summary_written = False

        rospy.Subscriber("/map", OccupancyGrid, self.map_callback, queue_size=1)
        rospy.Subscriber("/experiment1/state", String, self.state_callback, queue_size=100)
        rospy.Subscriber("/experiment1/event", String, self.event_callback, queue_size=200)
        for robot_name in self.robot_namespaces:
            rospy.Subscriber(f"/{robot_name}/odom", Odometry, self.make_odom_callback(robot_name), queue_size=50)

        rospy.Timer(rospy.Duration(0.5), self.timer_callback)
        rospy.on_shutdown(self.on_shutdown)
        rospy.loginfo("Metrics recorder output directory: %s", self.output_dir)

    @staticmethod
    def _default_result_root() -> str:
        try:
            import rospkg  # type: ignore

            return str(Path(rospkg.RosPack().get_path("experiment1")) / "results")
        except Exception:
            return str(Path(__file__).resolve().parents[1] / "results")

    def make_odom_callback(self, robot_name: str):
        def _callback(message: Odometry):
            position = message.pose.pose.position
            current = (float(position.x), float(position.y))
            sim_time = float(message.header.stamp.to_sec()) if message.header.stamp else float(rospy.Time.now().to_sec())
            if self.start_sim_time is None:
                self.start_sim_time = sim_time

            if robot_name in self.last_positions:
                previous = self.last_positions[robot_name]
                self.path_lengths[robot_name] += math.hypot(current[0] - previous[0], current[1] - previous[1])
            self.last_positions[robot_name] = current
            self.robot_trace_samples.append(
                {
                    "sim_time": sim_time,
                    "robot_name": robot_name,
                    "x": current[0],
                    "y": current[1],
                    "path_length_m": self.path_lengths[robot_name],
                }
            )

        return _callback

    def map_callback(self, message: OccupancyGrid):
        data = message.data
        known_cells = sum(1 for value in data if value >= 0)
        resolution = float(message.info.resolution)
        known_area = known_cells * resolution * resolution
        sim_time = float(message.header.stamp.to_sec()) if message.header.stamp else float(rospy.Time.now().to_sec())

        if self.start_sim_time is None:
            self.start_sim_time = sim_time

        if self.map_growth_samples and abs(self.map_growth_samples[-1]["sim_time"] - sim_time) < 1e-6:
            self.map_growth_samples[-1] = {
                "sim_time": sim_time,
                "known_cells": known_cells,
                "known_area_m2": known_area,
                "resolution": resolution,
            }
        else:
            self.map_growth_samples.append(
                {
                    "sim_time": sim_time,
                    "known_cells": known_cells,
                    "known_area_m2": known_area,
                    "resolution": resolution,
                }
            )

        self.latest_known_cells = known_cells
        self.latest_resolution = resolution
        self.latest_known_area = known_area

    def state_callback(self, message: String):
        payload = json.loads(message.data)
        sim_time = float(payload.get("sim_time", rospy.Time.now().to_sec()))
        if self.start_sim_time is None:
            self.start_sim_time = sim_time

        self.assignment_mode = payload.get("assignment_mode", self.assignment_mode)
        self.latest_raw_frontiers = int(payload.get("raw_frontier_count", 0))
        self.latest_valid_frontiers = int(payload.get("valid_frontier_count", 0))
        self.frontier_samples.append(
            {
                "sim_time": sim_time,
                "raw_frontier_count": self.latest_raw_frontiers,
                "valid_frontier_count": self.latest_valid_frontiers,
                "idle_robot_count": int(payload.get("idle_robot_count", 0)),
                "known_cells": int(payload.get("known_cells", self.latest_known_cells)),
                "finished": bool(payload.get("finished", False)),
                "finish_reason": payload.get("finish_reason", ""),
            }
        )

        if payload.get("finished", False) and self.finish_sim_time is None:
            self.finish_sim_time = sim_time
            self.finish_reason = payload.get("finish_reason", "")
            self.finalize_after_wall_time = time.monotonic() + 2.0

    def event_callback(self, message: String):
        payload = json.loads(message.data)
        self.events.append(payload)
        event_type = payload.get("event_type", "")
        robot_name = payload.get("robot_name")

        if event_type == "goal_sent" and robot_name:
            self.goal_sent_count[robot_name] += 1
        elif event_type == "goal_done" and robot_name:
            status = payload.get("status", "UNKNOWN")
            self.goal_done_status_count[robot_name][status] += 1
        elif event_type == "goal_timeout" and robot_name:
            self.goal_timeout_count[robot_name] += 1
        elif event_type == "goal_blacklisted":
            self.blacklist_reason_count[payload.get("reason", "unknown")] += 1
        elif event_type == "experiment_finished":
            self.finish_reason = payload.get("finish_reason", self.finish_reason)
            if self.finish_sim_time is None:
                self.finish_sim_time = float(payload.get("sim_time", rospy.Time.now().to_sec()))
            if self.finalize_after_wall_time is None:
                self.finalize_after_wall_time = time.monotonic() + 2.0

    def timer_callback(self, _event):
        if self.summary_written:
            return
        if self.finalize_after_wall_time is not None and time.monotonic() >= self.finalize_after_wall_time:
            self.write_outputs()
            rospy.signal_shutdown("Experiment finished and metrics saved.")

    def _compute_summary(self) -> Dict:
        if self.start_sim_time is None:
            self.start_sim_time = 0.0
        if self.finish_sim_time is None:
            self.finish_sim_time = self.map_growth_samples[-1]["sim_time"] if self.map_growth_samples else self.start_sim_time

        completion_time = max(0.0, self.finish_sim_time - self.start_sim_time)
        total_path_length = float(sum(self.path_lengths.values()))
        per_robot_goals_assigned = {robot: int(self.goal_sent_count[robot]) for robot in self.robot_namespaces}
        per_robot_goal_success = {
            robot: int(self.goal_done_status_count[robot]["SUCCEEDED"])
            for robot in self.robot_namespaces
        }
        per_robot_goal_failure = {
            robot: int(
                self.goal_done_status_count[robot]["ABORTED"]
                + self.goal_done_status_count[robot]["REJECTED"]
                + self.goal_done_status_count[robot]["LOST"]
                + self.goal_timeout_count[robot]
            )
            for robot in self.robot_namespaces
        }
        task_values = list(per_robot_goals_assigned.values()) or [0]
        path_values = list(self.path_lengths.values()) or [0.0]

        if self.map_growth_samples:
            initial_known_area = float(self.map_growth_samples[0]["known_area_m2"])
            initial_known_cells = int(self.map_growth_samples[0]["known_cells"])
        else:
            initial_known_area = 0.0
            initial_known_cells = 0

        map_growth_rate = (
            (self.latest_known_area - initial_known_area) / completion_time
            if completion_time > 1e-6
            else 0.0
        )

        return {
            "group_name": self.group_name,
            "run_name": self.run_name,
            "robot_namespaces": list(self.robot_namespaces),
            "assignment_mode": self.assignment_mode,
            "start_sim_time": float(self.start_sim_time),
            "finish_sim_time": float(self.finish_sim_time),
            "exploration_completion_time_sec": float(completion_time),
            "finish_reason": self.finish_reason or "shutdown_without_finish",
            "final_known_cells": int(self.latest_known_cells),
            "final_known_area_m2": float(self.latest_known_area),
            "initial_known_cells": int(initial_known_cells),
            "initial_known_area_m2": float(initial_known_area),
            "map_growth_rate_m2_per_sec": float(map_growth_rate),
            "latest_raw_frontier_count": int(self.latest_raw_frontiers),
            "latest_valid_frontier_count": int(self.latest_valid_frontiers),
            "total_path_length_m": total_path_length,
            "per_robot_path_length_m": {robot: float(value) for robot, value in self.path_lengths.items()},
            "path_length_std_m": float(statistics.pstdev(path_values) if len(path_values) > 1 else 0.0),
            "task_count_std": float(statistics.pstdev(task_values) if len(task_values) > 1 else 0.0),
            "per_robot_goals_assigned": per_robot_goals_assigned,
            "per_robot_goals_succeeded": per_robot_goal_success,
            "per_robot_goals_failed": per_robot_goal_failure,
            "per_robot_goal_timeouts": {robot: int(self.goal_timeout_count[robot]) for robot in self.robot_namespaces},
            "per_robot_goal_status_counts": {
                robot: {status: int(count) for status, count in self.goal_done_status_count[robot].items()}
                for robot in self.robot_namespaces
            },
            "total_goals_assigned": int(sum(per_robot_goals_assigned.values())),
            "total_goals_succeeded": int(sum(per_robot_goal_success.values())),
            "total_goals_failed": int(sum(per_robot_goal_failure.values())),
            "blacklist_reason_counts": {reason: int(count) for reason, count in self.blacklist_reason_count.items()},
            "artifacts": {
                "map_growth_csv": str(self.output_dir / "map_growth.csv"),
                "frontier_series_csv": str(self.output_dir / "frontier_series.csv"),
                "robot_paths_csv": str(self.output_dir / "robot_paths.csv"),
                "events_jsonl": str(self.output_dir / "events.jsonl"),
            },
        }

    def write_outputs(self):
        if self.summary_written:
            return

        summary = self._compute_summary()
        summary_path = self.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

        with (self.output_dir / "map_growth.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sim_time", "known_cells", "known_area_m2", "resolution"])
            writer.writeheader()
            writer.writerows(self.map_growth_samples)

        with (self.output_dir / "frontier_series.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "sim_time",
                    "raw_frontier_count",
                    "valid_frontier_count",
                    "idle_robot_count",
                    "known_cells",
                    "finished",
                    "finish_reason",
                ],
            )
            writer.writeheader()
            writer.writerows(self.frontier_samples)

        with (self.output_dir / "robot_paths.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["sim_time", "robot_name", "x", "y", "path_length_m"],
            )
            writer.writeheader()
            writer.writerows(self.robot_trace_samples)

        with (self.output_dir / "events.jsonl").open("w") as handle:
            for event in self.events:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")

        self.summary_written = True
        rospy.loginfo("Metrics written to %s", summary_path)

    def on_shutdown(self):
        self.write_outputs()


def main() -> int:
    MetricsRecorder()
    rospy.spin()
    return 0


if __name__ == "__main__":
    sys.exit(main())
