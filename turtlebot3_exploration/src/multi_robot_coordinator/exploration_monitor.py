#!/usr/bin/env python3

import rospy
from actionlib_msgs.msg import GoalStatus, GoalStatusArray


class ExplorationMonitor(object):
    FAILED_CODES = set(
        [
            GoalStatus.ABORTED,
            GoalStatus.REJECTED,
            GoalStatus.RECALLED,
            GoalStatus.LOST,
        ]
    )

    def __init__(self, robot_registry, goal_timeout=120.0):
        self.robot_registry = robot_registry
        self.goal_timeout = float(goal_timeout)
        self._latest_status = {}
        self._pending_events = []
        self._subscribers = []

        for state in self.robot_registry.all_states():
            subscriber = rospy.Subscriber(
                state.config.move_base_status_topic,
                GoalStatusArray,
                self._status_callback,
                callback_args=state.robot_id,
                queue_size=1,
            )
            self._subscribers.append(subscriber)

    def _status_callback(self, msg, robot_id):
        if not msg.status_list:
            return

        status = msg.status_list[-1]
        previous_code = self._latest_status.get(robot_id, {}).get("code")
        self._latest_status[robot_id] = {
            "code": status.status,
            "text": status.text,
            "stamp": rospy.Time.now(),
        }

        state = self.robot_registry.get_state(robot_id)
        if state is not None:
            state.last_status_code = status.status
            state.last_status_text = status.text
            if status.status == GoalStatus.ACTIVE:
                state.goal_status = "ACTIVE"
            elif status.status == GoalStatus.PENDING:
                state.goal_status = "PENDING"
            elif status.status == GoalStatus.SUCCEEDED:
                state.goal_status = "SUCCEEDED"
            elif status.status == GoalStatus.ABORTED:
                state.goal_status = "ABORTED"
            elif status.status == GoalStatus.REJECTED:
                state.goal_status = "FAILED"
            elif status.status == GoalStatus.RECALLED:
                state.goal_status = "CANCELED"
            elif status.status == GoalStatus.LOST:
                state.goal_status = "FAILED"

        if status.status == previous_code:
            return

        if status.status == GoalStatus.SUCCEEDED:
            self._pending_events.append((robot_id, "reached", status.text or "goal reached"))
        elif status.status in self.FAILED_CODES:
            self._pending_events.append((robot_id, "failed", status.text or "goal failed"))

    def poll_events(self):
        now = rospy.Time.now()
        events = list(self._pending_events)
        self._pending_events = []

        for state in self.robot_registry.all_states():
            if state.active_assignment is None or state.active_goal_sent_at is None:
                continue
            if state.goal_timeout_reported:
                continue

            if hasattr(state.active_goal_sent_at, "to_sec"):
                started_at = state.active_goal_sent_at
            else:
                started_at = rospy.Time.from_sec(float(state.active_goal_sent_at))

            if (now - started_at).to_sec() > self.goal_timeout:
                state.goal_timeout_reported = True
                state.goal_status = "FAILED"
                events.append((state.robot_id, "timeout", "goal timeout"))

        return events
