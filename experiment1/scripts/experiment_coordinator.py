#!/usr/bin/env python3
"""Experiment-specific coordinator for multi-robot exploration comparisons."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import rospy
import tf
from actionlib_msgs.msg import GoalStatus
from std_msgs.msg import String


def _ensure_central_import_path() -> None:
    try:
        import rospkg  # type: ignore

        central_script_dir = Path(rospkg.RosPack().get_path("turtlebot3_central")) / "script"
    except Exception:
        central_script_dir = (
            Path(__file__).resolve().parents[2] / "turtlebot3_central" / "script"
        )
    if str(central_script_dir) not in sys.path:
        sys.path.insert(0, str(central_script_dir))


_ensure_central_import_path()

from central import FrontierExplorer, StoppingConditionEvaluator, TaskAssigner  # noqa: E402


EVENT_TOPIC = "/experiment1/event"
STATE_TOPIC = "/experiment1/state"


def _json_publish(publisher: rospy.Publisher, payload: Dict) -> None:
    message = String()
    message.data = json.dumps(payload, ensure_ascii=False)
    publisher.publish(message)


def _serialize_goal(goal: Optional[Tuple[float, float, float]]) -> Optional[Dict[str, float]]:
    if goal is None:
        return None
    payload = {"x": float(goal[0]), "y": float(goal[1])}
    if len(goal) > 2:
        payload["size"] = float(goal[2])
    return payload


class ExperimentTaskAssigner(TaskAssigner):
    def __init__(self, event_publisher: rospy.Publisher):
        self.event_publisher = event_publisher
        super().__init__()

    def _publish_event(self, event_type: str, **payload) -> None:
        event = {
            "event_type": event_type,
            "sim_time": float(rospy.Time.now().to_sec()),
        }
        event.update(payload)
        _json_publish(self.event_publisher, event)

    def blacklist_goal(self, goal_coords, reason="failed_goal"):
        super().blacklist_goal(goal_coords, reason=reason)
        self._publish_event(
            "goal_blacklisted",
            reason=reason,
            goal=_serialize_goal(goal_coords),
        )

    def send_goal(self, robot_name: str, goal_coords: tuple, known_area=None):
        super().send_goal(robot_name, goal_coords, known_area=known_area)
        self._publish_event(
            "goal_sent",
            robot_name=robot_name,
            goal=_serialize_goal(goal_coords),
            known_area_at_send=int(self.goal_start_known_areas.get(robot_name, self.latest_known_area)),
        )

    def goal_done_callback(self, robot_name, status, result):
        goal_coords = self.current_goals.get(robot_name)
        start_known_area = self.goal_start_known_areas.get(robot_name, self.latest_known_area)
        status_names = {
            GoalStatus.PENDING: "PENDING",
            GoalStatus.ACTIVE: "ACTIVE",
            GoalStatus.PREEMPTED: "PREEMPTED",
            GoalStatus.SUCCEEDED: "SUCCEEDED",
            GoalStatus.ABORTED: "ABORTED",
            GoalStatus.REJECTED: "REJECTED",
            GoalStatus.PREEMPTING: "PREEMPTING",
            GoalStatus.RECALLING: "RECALLING",
            GoalStatus.RECALLED: "RECALLED",
            GoalStatus.LOST: "LOST",
        }
        status_name = status_names.get(status, f"UNKNOWN({status})")
        known_area_gain = max(0, self.latest_known_area - start_known_area)
        super().goal_done_callback(robot_name, status, result)
        self._publish_event(
            "goal_done",
            robot_name=robot_name,
            status=status_name,
            goal=_serialize_goal(goal_coords),
            known_area_gain=int(known_area_gain),
        )

    def check_timeouts(self, timeout_sec=40.0):
        current_time = rospy.Time.now()
        for robot_name, status in list(self.robot_status.items()):
            if status == "BUSY" and robot_name in self.goal_start_times:
                elapsed = (current_time - self.goal_start_times[robot_name]).to_sec()
                if elapsed > timeout_sec:
                    goal_coords = self.current_goals.get(robot_name)
                    rospy.logwarn(
                        "Goal for %s timed out (%.1fs). Cancelling...", robot_name, timeout_sec
                    )
                    self._publish_event(
                        "goal_timeout",
                        robot_name=robot_name,
                        elapsed_sec=float(elapsed),
                        goal=_serialize_goal(goal_coords),
                    )
                    self.blacklist_goal(goal_coords, reason="timeout")
                    self.clients[robot_name].cancel_goal()
                    self.robot_status[robot_name] = "IDLE"
                    self.current_goals.pop(robot_name, None)
                    self.goal_start_times.pop(robot_name, None)
                    self.goal_start_known_areas.pop(robot_name, None)

    def cancel_all_goals(self):
        busy_before_cancel = list(self.current_goals.keys())
        super().cancel_all_goals()
        self._publish_event("cancel_all_goals", robot_names=busy_before_cancel)


def ordered_nearest_assignments(
    frontiers: Sequence[tuple],
    robot_positions: Dict[str, Tuple[float, float]],
    idle_robots: Sequence[str],
    task_assigner: ExperimentTaskAssigner,
) -> Dict[str, tuple]:
    assignments = {}
    available_frontiers = [
        frontier for frontier in frontiers
        if not task_assigner.is_blacklisted(frontier)
    ]

    ordered_idle = [name for name in task_assigner.robot_names if name in idle_robots]
    for robot_name in ordered_idle:
        robot_position = robot_positions.get(robot_name)
        if robot_position is None:
            continue

        best_frontier = None
        best_distance = None
        for frontier in available_frontiers:
            if not task_assigner.is_goal_reachable(robot_name, frontier, robot_position):
                continue
            distance = np.hypot(frontier[0] - robot_position[0], frontier[1] - robot_position[1])
            if best_frontier is None or distance < best_distance:
                best_frontier = frontier
                best_distance = distance

        if best_frontier is not None:
            assignments[robot_name] = best_frontier
            available_frontiers = [item for item in available_frontiers if item != best_frontier]

    return assignments


def publish_state(
    state_publisher: rospy.Publisher,
    group_name: str,
    assignment_mode: str,
    task_assigner: ExperimentTaskAssigner,
    raw_frontiers: Sequence[tuple],
    valid_frontiers: Sequence[tuple],
    idle_robots: Sequence[str],
    known_cells: int,
    finished: bool,
    finish_reason: str,
) -> None:
    payload = {
        "group_name": group_name,
        "assignment_mode": assignment_mode,
        "sim_time": float(rospy.Time.now().to_sec()),
        "robot_namespaces": list(task_assigner.robot_names),
        "raw_frontier_count": int(len(raw_frontiers)),
        "valid_frontier_count": int(len(valid_frontiers)),
        "idle_robot_count": int(len(idle_robots)),
        "idle_robots": list(idle_robots),
        "known_cells": int(known_cells),
        "robot_status": dict(task_assigner.robot_status),
        "current_goals": {
            robot_name: _serialize_goal(goal)
            for robot_name, goal in task_assigner.current_goals.items()
        },
        "finished": bool(finished),
        "finish_reason": finish_reason or "",
    }
    _json_publish(state_publisher, payload)


def main() -> int:
    rospy.init_node("experiment_coordinator")
    rospy.set_param("~manual_stop_requested", False)

    group_name = rospy.get_param("~group_name", "unnamed_group")
    assignment_mode = rospy.get_param("~assignment_mode", "scored")
    state_publisher = rospy.Publisher(STATE_TOPIC, String, queue_size=10, latch=True)
    event_publisher = rospy.Publisher(EVENT_TOPIC, String, queue_size=50)

    explorer = FrontierExplorer()
    task_assigner = ExperimentTaskAssigner(event_publisher)
    stopping_evaluator = StoppingConditionEvaluator()
    tf_listener = tf.TransformListener()

    rospy.sleep(2.0)
    rospy.loginfo("Starting experiment coordinator for %s (%s)", group_name, assignment_mode)
    _json_publish(
        event_publisher,
        {
            "event_type": "experiment_started",
            "group_name": group_name,
            "assignment_mode": assignment_mode,
            "sim_time": float(rospy.Time.now().to_sec()),
            "robot_namespaces": list(task_assigner.robot_names),
        },
    )

    rate = rospy.Rate(1.0)
    exploration_completed = False

    while not rospy.is_shutdown():
        try:
            task_assigner.prune_blacklist()
            task_assigner.check_timeouts(40.0)

            idle_robots = task_assigner.get_idle_robots()
            all_idle = task_assigner.all_robots_idle()
            robot_positions = explorer.get_robot_positions(tf_listener)

            frontiers = explorer.detect_frontiers()
            valid_frontiers = task_assigner.filter_valid_frontiers(frontiers, robot_positions)
            current_time_sec = rospy.Time.now().to_sec()
            current_known_area = explorer.get_known_area()
            task_assigner.update_known_area(current_known_area)

            manual_stop_requested = bool(rospy.get_param("~manual_stop_requested", False))
            if manual_stop_requested and not stopping_evaluator.finished:
                rospy.logwarn("Manual stop requested. Cancelling exploration and all active goals.")
                stopping_evaluator.request_manual_stop()

            finished, finish_reason, stop_debug = stopping_evaluator.update(
                valid_frontiers=valid_frontiers,
                all_idle=all_idle,
                known_cells=current_known_area,
                current_time_sec=current_time_sec,
                raw_frontier_count=len(frontiers),
            )
            rospy.loginfo_throttle(5.0, "Stopping check: %s", stop_debug)
            explorer.publish_markers(frontiers)

            if exploration_completed or finished:
                if not exploration_completed:
                    rospy.loginfo("=" * 50)
                    rospy.loginfo("Exploration completed! reason=%s", finish_reason)
                    rospy.loginfo("=" * 50)
                    exploration_completed = True
                    task_assigner.cancel_all_goals()
                    _json_publish(
                        event_publisher,
                        {
                            "event_type": "experiment_finished",
                            "group_name": group_name,
                            "assignment_mode": assignment_mode,
                            "sim_time": float(rospy.Time.now().to_sec()),
                            "finish_reason": finish_reason,
                        },
                    )

                publish_state(
                    state_publisher=state_publisher,
                    group_name=group_name,
                    assignment_mode=assignment_mode,
                    task_assigner=task_assigner,
                    raw_frontiers=frontiers,
                    valid_frontiers=valid_frontiers,
                    idle_robots=idle_robots,
                    known_cells=current_known_area,
                    finished=True,
                    finish_reason=finish_reason,
                )
                rate.sleep()
                continue

            assignments = {}
            if idle_robots and valid_frontiers:
                if assignment_mode == "ordered_nearest":
                    assignments = ordered_nearest_assignments(
                        frontiers=valid_frontiers,
                        robot_positions=robot_positions,
                        idle_robots=idle_robots,
                        task_assigner=task_assigner,
                    )
                else:
                    assignments = explorer.assign_tasks(
                        valid_frontiers,
                        robot_positions,
                        idle_robots,
                        active_goals=task_assigner.current_goals.copy(),
                    )

                if assignments:
                    rospy.loginfo("Assigning %d tasks for group %s", len(assignments), group_name)
                    for robot_name, frontier in assignments.items():
                        task_assigner.send_goal(robot_name, frontier, known_area=current_known_area)
                else:
                    rospy.loginfo_throttle(
                        5.0,
                        "No assignments made for %s (idle robots may be too far or no suitable frontiers)",
                        group_name,
                    )

            explorer.publish_assignments(task_assigner.current_goals)
            publish_state(
                state_publisher=state_publisher,
                group_name=group_name,
                assignment_mode=assignment_mode,
                task_assigner=task_assigner,
                raw_frontiers=frontiers,
                valid_frontiers=valid_frontiers,
                idle_robots=idle_robots,
                known_cells=current_known_area,
                finished=False,
                finish_reason="",
            )
            rate.sleep()

        except KeyboardInterrupt:
            rospy.loginfo("Keyboard interrupt, cancelling all goals...")
            task_assigner.cancel_all_goals()
            break
        except Exception as exc:
            rospy.logerr("Error in experiment coordinator loop: %s", exc)
            import traceback

            traceback.print_exc()
            rate.sleep()

    rospy.loginfo("Experiment coordinator shutting down...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
