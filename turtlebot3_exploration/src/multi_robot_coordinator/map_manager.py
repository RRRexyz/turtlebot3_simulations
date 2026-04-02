#!/usr/bin/env python3

import rospy
from nav_msgs.msg import OccupancyGrid


class MapManager(object):
    def __init__(self, robot_registry, stale_timeout=5.0):
        self.robot_registry = robot_registry
        self.stale_timeout = float(stale_timeout)
        self._maps = {}
        self._received_at = {}
        self._subscribers = []

        for state in self.robot_registry.all_states():
            subscriber = rospy.Subscriber(
                state.config.map_topic,
                OccupancyGrid,
                self._map_callback,
                callback_args=state.robot_id,
                queue_size=1,
            )
            self._subscribers.append(subscriber)

    def _map_callback(self, msg, robot_id):
        self._maps[robot_id] = msg
        self._received_at[robot_id] = rospy.Time.now()
        state = self.robot_registry.get_state(robot_id)
        if state is not None:
            state.last_map_stamp = msg.header.stamp
            state.last_map_received_at = self._received_at[robot_id]

    def get_latest_map(self, robot_id):
        return self._maps.get(robot_id)

    def get_latest_maps(self):
        return dict(self._maps)

    def is_map_valid(self, robot_id):
        msg = self.get_latest_map(robot_id)
        return msg is not None and msg.info.width > 0 and msg.info.height > 0 and len(msg.data) > 0

    def is_map_stale(self, robot_id):
        received_at = self._received_at.get(robot_id)
        if received_at is None:
            return True
        return (rospy.Time.now() - received_at).to_sec() > self.stale_timeout
