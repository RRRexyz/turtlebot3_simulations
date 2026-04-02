#!/usr/bin/env python3

import rospy

from .data_types import Assignment


class TaskAllocator(object):
    def __init__(
        self,
        distance_weight=1.0,
        information_gain_weight=2.0,
        conflict_penalty_weight=3.0,
        conflict_distance=1.0,
    ):
        self.distance_weight = float(distance_weight)
        self.information_gain_weight = float(information_gain_weight)
        self.conflict_penalty_weight = float(conflict_penalty_weight)
        self.conflict_distance = float(conflict_distance)

    def allocate(self, robot_states, global_frontiers):
        idle_robots = [state for state in robot_states if self._is_idle(state)]
        remaining_frontiers = list(global_frontiers)
        assignments = []

        while idle_robots and remaining_frontiers:
            best_candidate = None
            reserved_goals = [assignment.goal_global for assignment in assignments]
            reserved_goals.extend(
                state.active_assignment.goal_global
                for state in robot_states
                if state.active_assignment is not None
            )

            for robot_state in idle_robots:
                if robot_state.pose_global_to_base is None:
                    continue
                for frontier in remaining_frontiers:
                    score = self._score(robot_state, frontier, reserved_goals)
                    if best_candidate is None or score < best_candidate[0]:
                        best_candidate = (score, robot_state, frontier)

            if best_candidate is None:
                break

            score, robot_state, frontier = best_candidate
            assignments.append(
                Assignment(
                    robot_id=robot_state.robot_id,
                    frontier_id=frontier.frontier_id,
                    goal_global=frontier.centroid.copy(),
                    score=score,
                    created_at=rospy.Time.now().to_sec(),
                    status="ASSIGNED",
                )
            )
            idle_robots = [state for state in idle_robots if state.robot_id != robot_state.robot_id]
            remaining_frontiers = [
                frontier_item
                for frontier_item in remaining_frontiers
                if frontier_item.frontier_id != frontier.frontier_id
            ]

        return assignments

    def _is_idle(self, robot_state):
        return robot_state.active_assignment is None and robot_state.status in ("IDLE", "READY", "FAILED", "REACHED")

    def _score(self, robot_state, frontier, reserved_goals):
        distance = robot_state.pose_global_to_base.distance_xy(frontier.centroid)
        conflict_penalty = 0.0
        for goal in reserved_goals:
            distance_to_reserved = frontier.centroid.distance_xy(goal)
            if distance_to_reserved < self.conflict_distance:
                conflict_penalty += (self.conflict_distance - distance_to_reserved) / max(self.conflict_distance, 1e-6)

        return (
            self.distance_weight * distance
            - self.information_gain_weight * frontier.gain
            + self.conflict_penalty_weight * conflict_penalty
        )
