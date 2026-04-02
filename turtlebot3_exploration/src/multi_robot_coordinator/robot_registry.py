#!/usr/bin/env python3

import os

import yaml

from .data_types import Pose2D, RobotConfig, RobotState


def _normalize_namespace(namespace):
    return namespace.strip("/")


def _resolve_topic(namespace, topic_name):
    if topic_name.startswith("/"):
        return topic_name
    namespace = _normalize_namespace(namespace)
    if not namespace:
        return "/" + topic_name.lstrip("/")
    return "/" + namespace + "/" + topic_name.lstrip("/")


def _resolve_frame(namespace, frame_name, default_suffix):
    if frame_name:
        return frame_name
    namespace = _normalize_namespace(namespace)
    if not namespace:
        return default_suffix
    return namespace + "/" + default_suffix


def _parse_pose2d(value):
    if isinstance(value, dict):
        return Pose2D(
            x=float(value.get("x", 0.0)),
            y=float(value.get("y", 0.0)),
            yaw=float(value.get("yaw", 0.0)),
        )
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return Pose2D(x=float(value[0]), y=float(value[1]), yaw=float(value[2]))
    return Pose2D()


class RobotRegistry(object):
    def __init__(self, config_path):
        self.config_path = os.path.abspath(config_path)
        self._states = {}
        self._configs = {}
        self._load_from_yaml(self.config_path)

    def _load_from_yaml(self, config_path):
        with open(config_path, "r") as stream:
            content = yaml.safe_load(stream) or {}

        robot_entries = content.get("robots", [])
        if isinstance(robot_entries, dict):
            robot_entries = [dict({"robot_id": key}, **value) for key, value in robot_entries.items()]

        for entry in robot_entries:
            robot_id = str(entry.get("robot_id") or entry.get("id"))
            namespace = _normalize_namespace(entry.get("namespace", robot_id))
            config = RobotConfig(
                robot_id=robot_id,
                namespace=namespace,
                map_frame=_resolve_frame(namespace, entry.get("map_frame"), "map"),
                odom_frame=_resolve_frame(namespace, entry.get("odom_frame"), "odom"),
                base_frame=_resolve_frame(namespace, entry.get("base_frame"), "base_link"),
                map_topic=_resolve_topic(namespace, entry.get("map_topic", "map")),
                move_base_action=_resolve_topic(namespace, entry.get("move_base_action", "move_base")),
                move_base_status_topic=_resolve_topic(
                    namespace,
                    entry.get("move_base_status_topic", "move_base/status"),
                ),
                move_base_make_plan_service=_resolve_topic(
                    namespace,
                    entry.get("move_base_make_plan_service", "move_base/make_plan"),
                ),
                spawn_pose=_parse_pose2d(entry.get("spawn", entry.get("spawn_pose", {}))),
            )
            self._configs[robot_id] = config
            self._states[robot_id] = RobotState(robot_id=robot_id, config=config)

    def robot_ids(self):
        return list(self._states.keys())

    def all_states(self):
        return list(self._states.values())

    def all_configs(self):
        return list(self._configs.values())

    def get_state(self, robot_id):
        return self._states.get(robot_id)

    def get_config(self, robot_id):
        return self._configs.get(robot_id)

    def get_active_assignments(self):
        assignments = []
        for state in self.all_states():
            if state.active_assignment is not None:
                assignments.append(state.active_assignment)
        return assignments

    def set_idle(self, robot_id, status="IDLE"):
        state = self.get_state(robot_id)
        if state is None:
            return
        state.status = status
        state.goal_status = status
        state.active_assignment = None
        state.active_goal_sent_at = None
        state.current_goal_local = None
        state.goal_timeout_reported = False

    def set_busy(self, robot_id, assignment, local_goal):
        state = self.get_state(robot_id)
        if state is None:
            return
        state.status = "ACTIVE"
        state.goal_status = "ACTIVE"
        state.active_assignment = assignment
        state.active_goal_sent_at = assignment.created_at
        state.current_goal_local = local_goal
        state.goal_timeout_reported = False
