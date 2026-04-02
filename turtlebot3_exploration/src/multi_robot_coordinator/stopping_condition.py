#!/usr/bin/env python3

"""Exploration stopping condition evaluation for the multi-robot coordinator."""

import math

from tf.transformations import euler_from_quaternion

from .data_types import ExplorationStopState, Pose2D


class StoppingConditionEvaluator(object):
    """Evaluate composite exploration stopping conditions with confirmation cycles."""

    ACTIVE_GOAL_STATUSES = set(["ACTIVE", "PENDING"])
    IDLE_GOAL_STATUSES = set(["IDLE", "READY"])
    TERMINAL_GOAL_STATUSES = set(["SUCCEEDED", "ABORTED", "FAILED", "CANCELED", "REACHED"])

    def __init__(self, config, pose_registration=None):
        self.enabled = bool(config.get("enabled", True))
        self.stop_confirm_cycles = int(config.get("stop_confirm_cycles", 8))
        self.map_growth_window_sec = float(config.get("map_growth_window_sec", 15.0))
        self.known_cells_growth_abs_threshold = int(
            config.get("known_cells_growth_abs_threshold", 80)
        )
        self.known_cells_growth_ratio_threshold = float(
            config.get("known_cells_growth_ratio_threshold", 0.005)
        )
        self.max_exploration_runtime_sec = float(
            config.get("max_exploration_runtime_sec", 1800.0)
        )
        self.max_failed_reassign_rounds = int(config.get("max_failed_reassign_rounds", 10))
        self.coverage_resolution = float(config.get("coverage_resolution", 0.1))
        self.cancel_goals_on_finish = bool(config.get("cancel_goals_on_finish", True))
        self.publish_finished_topic = config.get("publish_finished_topic", "/exploration_finished")
        self.publish_finish_reason_topic = config.get(
            "publish_finish_reason_topic",
            "/exploration_finish_reason",
        )
        self.publish_stop_debug_topic = config.get(
            "publish_stop_debug_topic",
            "/exploration_stop_debug",
        )
        self.pose_registration = pose_registration
        self.state = ExplorationStopState()
        self.start_time_sec = None

    def update(
        self,
        filtered_global_frontiers,
        robot_states,
        global_map_msg,
        current_time_sec,
        failed_reassign_rounds=0,
        manual_stop=False,
        local_map_msgs=None,
    ):
        if self.start_time_sec is None:
            self.start_time_sec = float(current_time_sec)

        if self.state.finished:
            self.state.runtime_sec = max(0.0, float(current_time_sec) - self.start_time_sec)
            return self.state

        self.state.valid_global_frontier_count = self._count_valid_frontiers(filtered_global_frontiers)
        active_count, idle_count, terminal_count = self._count_robot_states(robot_states)
        self.state.active_goal_count = active_count
        self.state.idle_robot_count = idle_count
        self.state.terminal_robot_count = terminal_count
        self.state.no_valid_frontiers = self.has_no_valid_frontiers(filtered_global_frontiers)
        self.state.all_robots_idle_or_terminal = active_count == 0
        self.state.runtime_sec = max(0.0, float(current_time_sec) - self.start_time_sec)

        current_known_cells = self._count_known_cells(
            global_map_msg=global_map_msg,
            local_map_msgs=local_map_msgs or {},
        )
        self.state.known_cell_count_current = current_known_cells
        self._update_known_cell_history(current_time_sec, current_known_cells)
        self._update_growth_metrics(current_time_sec)

        self.state.map_growth_below_threshold = self._is_map_growth_small(current_time_sec)

        forced_reason = self._check_forced_stop_conditions(
            failed_reassign_rounds=failed_reassign_rounds,
            manual_stop=manual_stop,
        )
        if forced_reason:
            self.state.finished = True
            self.state.finish_reason = forced_reason
            return self.state

        composite_reached = (
            self.enabled
            and self.state.no_valid_frontiers
            and self.state.all_robots_idle_or_terminal
            and self.state.map_growth_below_threshold
        )
        if composite_reached:
            self.state.stop_counter += 1
        else:
            self.state.stop_counter = 0

        if self.state.stop_counter >= self.stop_confirm_cycles:
            self.state.finished = True
            self.state.finish_reason = "frontier_exhausted_and_map_stagnant"

        return self.state

    def has_no_valid_frontiers(self, frontiers):
        return self._count_valid_frontiers(frontiers) == 0

    def is_finished(self):
        return self.state.finished

    def get_finish_reason(self):
        return self.state.finish_reason

    def reset(self):
        self.state = ExplorationStopState()
        self.start_time_sec = None

    def _count_valid_frontiers(self, frontiers):
        return len(frontiers or [])

    def _count_robot_states(self, robot_states):
        active_count = 0
        idle_count = 0
        terminal_count = 0

        for state in robot_states or []:
            goal_status = getattr(state, "goal_status", getattr(state, "status", "IDLE")) or "IDLE"
            if goal_status in self.ACTIVE_GOAL_STATUSES:
                active_count += 1
            elif goal_status in self.TERMINAL_GOAL_STATUSES:
                terminal_count += 1
            else:
                idle_count += 1

        return active_count, idle_count, terminal_count

    def _count_known_cells(self, global_map_msg=None, local_map_msgs=None):
        if self._is_map_msg_valid(global_map_msg):
            return sum(1 for value in global_map_msg.data if value >= 0)

        if not local_map_msgs or self.pose_registration is None:
            return 0

        coverage_cells = set()
        for robot_id, map_msg in local_map_msgs.items():
            if not self._is_map_msg_valid(map_msg):
                continue
            if self.pose_registration.get_registration(robot_id) is None:
                continue

            origin_yaw = euler_from_quaternion(
                [
                    map_msg.info.origin.orientation.x,
                    map_msg.info.origin.orientation.y,
                    map_msg.info.origin.orientation.z,
                    map_msg.info.origin.orientation.w,
                ]
            )[2]
            cos_yaw = math.cos(origin_yaw)
            sin_yaw = math.sin(origin_yaw)
            resolution = map_msg.info.resolution
            origin_x = map_msg.info.origin.position.x
            origin_y = map_msg.info.origin.position.y
            width = map_msg.info.width

            for index, value in enumerate(map_msg.data):
                if value < 0:
                    continue
                ix = index % width
                iy = index // width
                local_x = origin_x + ((ix + 0.5) * resolution * cos_yaw) - ((iy + 0.5) * resolution * sin_yaw)
                local_y = origin_y + ((ix + 0.5) * resolution * sin_yaw) + ((iy + 0.5) * resolution * cos_yaw)
                pose_global = self.pose_registration.local_to_global(
                    robot_id,
                    Pose2D(x=local_x, y=local_y, yaw=0.0),
                )
                if pose_global is None:
                    continue
                coverage_cells.add(self._to_coverage_cell(pose_global.x, pose_global.y))

        return len(coverage_cells)

    def _update_known_cell_history(self, current_time_sec, current_known_cells):
        if self.state.known_cell_count_history and self.state.known_cell_count_history[-1][0] == current_time_sec:
            self.state.known_cell_count_history[-1] = (current_time_sec, current_known_cells)
        else:
            self.state.known_cell_count_history.append((current_time_sec, current_known_cells))

        while len(self.state.known_cell_count_history) > 1:
            next_entry_time = self.state.known_cell_count_history[1][0]
            if next_entry_time < (current_time_sec - self.map_growth_window_sec):
                self.state.known_cell_count_history.popleft()
            else:
                break

    def _update_growth_metrics(self, current_time_sec):
        if not self.state.known_cell_count_history:
            self.state.known_cell_growth_in_window = 0
            self.state.map_growth_ratio_in_window = 0.0
            return

        oldest_time, oldest_known_cells = self.state.known_cell_count_history[0]
        current_known_cells = self.state.known_cell_count_current
        self.state.known_cell_growth_in_window = current_known_cells - oldest_known_cells
        self.state.map_growth_ratio_in_window = float(self.state.known_cell_growth_in_window) / float(
            max(oldest_known_cells, 1)
        )

    def _is_map_growth_small(self, current_time_sec):
        if not self.state.known_cell_count_history:
            return False
        oldest_time = self.state.known_cell_count_history[0][0]
        if (current_time_sec - oldest_time) < self.map_growth_window_sec:
            return False

        abs_small = self.state.known_cell_growth_in_window < self.known_cells_growth_abs_threshold
        ratio_small = (
            self.state.map_growth_ratio_in_window < self.known_cells_growth_ratio_threshold
        )
        return abs_small or ratio_small

    def _check_forced_stop_conditions(self, failed_reassign_rounds, manual_stop):
        if manual_stop:
            return "manual_stop"
        if self.max_exploration_runtime_sec > 0.0 and self.state.runtime_sec >= self.max_exploration_runtime_sec:
            return "timeout"
        if self.max_failed_reassign_rounds >= 0 and failed_reassign_rounds >= self.max_failed_reassign_rounds:
            return "too_many_failed_reassignments"
        return ""

    def _is_map_msg_valid(self, map_msg):
        return (
            map_msg is not None
            and getattr(map_msg.info, "width", 0) > 0
            and getattr(map_msg.info, "height", 0) > 0
            and len(getattr(map_msg, "data", [])) > 0
        )

    def _to_coverage_cell(self, x_coord, y_coord):
        resolution = max(self.coverage_resolution, 1e-6)
        return (
            int(math.floor(x_coord / resolution)),
            int(math.floor(y_coord / resolution)),
        )

    def build_debug_string(self):
        return (
            "valid_global_frontier_count={0} active_goal_count={1} idle_robot_count={2} "
            "terminal_robot_count={3} known_cell_count_current={4} "
            "known_cell_growth_in_window={5} map_growth_ratio_in_window={6:.6f} "
            "stop_counter={7} finished={8} finish_reason={9}"
        ).format(
            self.state.valid_global_frontier_count,
            self.state.active_goal_count,
            self.state.idle_robot_count,
            self.state.terminal_robot_count,
            self.state.known_cell_count_current,
            self.state.known_cell_growth_in_window,
            self.state.map_growth_ratio_in_window,
            self.state.stop_counter,
            str(self.state.finished).lower(),
            self.state.finish_reason or "none",
        )
