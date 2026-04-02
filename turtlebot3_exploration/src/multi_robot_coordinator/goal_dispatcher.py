#!/usr/bin/env python3

import actionlib
import rospy
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from tf.transformations import quaternion_from_euler


class GoalDispatcher(object):
    def __init__(self, robot_registry, pose_registration, server_wait_timeout=1.0):
        self.robot_registry = robot_registry
        self.pose_registration = pose_registration
        self.server_wait_timeout = float(server_wait_timeout)
        self._clients = {}
        self._server_ready = {}

        for state in self.robot_registry.all_states():
            client = actionlib.SimpleActionClient(state.config.move_base_action, MoveBaseAction)
            self._clients[state.robot_id] = client
            self._server_ready[state.robot_id] = False

    def wait_for_servers(self):
        for robot_id, client in self._clients.items():
            self._server_ready[robot_id] = client.wait_for_server(rospy.Duration(self.server_wait_timeout))
            if not self._server_ready[robot_id]:
                rospy.logwarn("move_base action server not ready for %s", robot_id)

    def is_server_ready(self, robot_id):
        client = self._clients.get(robot_id)
        if client is None:
            return False
        if self._server_ready.get(robot_id):
            return True
        self._server_ready[robot_id] = client.wait_for_server(rospy.Duration(0.05))
        return self._server_ready[robot_id]

    def send_goal(self, robot_id, goal_global):
        state = self.robot_registry.get_state(robot_id)
        client = self._clients.get(robot_id)
        if state is None or client is None:
            return None
        if not self.is_server_ready(robot_id):
            return None

        goal_local = self.pose_registration.global_to_local(robot_id, goal_global)
        if goal_local is None:
            return None

        goal = MoveBaseGoal()
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.header.frame_id = state.config.map_frame
        goal.target_pose.pose.position.x = goal_local.x
        goal.target_pose.pose.position.y = goal_local.y
        quaternion = quaternion_from_euler(0.0, 0.0, goal_local.yaw)
        goal.target_pose.pose.orientation.x = quaternion[0]
        goal.target_pose.pose.orientation.y = quaternion[1]
        goal.target_pose.pose.orientation.z = quaternion[2]
        goal.target_pose.pose.orientation.w = quaternion[3]
        client.send_goal(goal)
        return goal_local

    def cancel_goal(self, robot_id):
        client = self._clients.get(robot_id)
        if client is None:
            return
        client.cancel_goal()

    def get_client_state(self, robot_id):
        client = self._clients.get(robot_id)
        if client is None:
            return None
        return client.get_state()
