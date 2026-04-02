#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.srv import GetPlan, GetPlanRequest
from tf.transformations import quaternion_from_euler


class FrontierFilter(object):
    def __init__(
        self,
        robot_registry=None,
        pose_registration=None,
        min_size=5,
        min_gain=0.1,
        min_robot_distance=0.5,
        min_assignment_distance=0.75,
        min_frontier_separation=0.75,
        blacklist_radius=0.75,
        use_reachability_filter=True,
        min_plan_poses=2,
        plan_tolerance=0.2,
        make_plan_wait_timeout=0.2,
    ):
        self.robot_registry = robot_registry
        self.pose_registration = pose_registration
        self.min_size = int(min_size)
        self.min_gain = float(min_gain)
        self.min_robot_distance = float(min_robot_distance)
        self.min_assignment_distance = float(min_assignment_distance)
        self.min_frontier_separation = float(min_frontier_separation)
        self.blacklist_radius = float(blacklist_radius)
        self.use_reachability_filter = bool(use_reachability_filter)
        self.min_plan_poses = int(min_plan_poses)
        self.plan_tolerance = float(plan_tolerance)
        self.make_plan_wait_timeout = float(make_plan_wait_timeout)
        self._make_plan_clients = {}
        self._make_plan_ready = {}
        self._last_reachability_result = {}

    def filter(self, frontiers, robot_states, active_assignments=None, blacklisted_goals=None):
        active_assignments = active_assignments or []
        blacklisted_goals = blacklisted_goals or []
        self._last_reachability_result = {}

        filtered = []
        for frontier in sorted(frontiers, key=lambda item: item.gain, reverse=True):
            if frontier.size < self.min_size or frontier.gain < self.min_gain:
                continue
            if self._is_too_close_to_robot(frontier, robot_states):
                continue
            if self._is_too_close_to_assignment(frontier, active_assignments):
                continue
            if self._is_blacklisted(frontier, blacklisted_goals):
                continue
            if self._is_too_close_to_selected(frontier, filtered):
                continue
            if self._is_unreachable(frontier, robot_states):
                continue
            filtered.append(frontier)

        return filtered

    def _is_too_close_to_robot(self, frontier, robot_states):
        for state in robot_states:
            if state.pose_global_to_base is None:
                continue
            if frontier.centroid.distance_xy(state.pose_global_to_base) < self.min_robot_distance:
                return True
        return False

    def _is_too_close_to_assignment(self, frontier, active_assignments):
        for assignment in active_assignments:
            if frontier.centroid.distance_xy(assignment.goal_global) < self.min_assignment_distance:
                return True
        return False

    def _is_blacklisted(self, frontier, blacklisted_goals):
        for blacklisted_pose in blacklisted_goals:
            if frontier.centroid.distance_xy(blacklisted_pose) < self.blacklist_radius:
                return True
        return False

    def _is_too_close_to_selected(self, frontier, selected):
        for kept in selected:
            if frontier.centroid.distance_xy(kept.centroid) < self.min_frontier_separation:
                return True
        return False

    def _is_unreachable(self, frontier, robot_states):
        if not self.use_reachability_filter:
            return False
        if self.robot_registry is None or self.pose_registration is None:
            return False

        candidate_states = [
            state for state in robot_states
            if state.registration_initialized and state.pose_map_to_base is not None
        ]
        if not candidate_states:
            return False

        service_checked = False
        for state in candidate_states:
            if self._has_valid_plan(state, frontier):
                self._last_reachability_result[frontier.frontier_id] = state.robot_id
                return False
            if self._make_plan_ready.get(state.robot_id, False):
                service_checked = True

        if not service_checked:
            rospy.logwarn_throttle(
                5.0,
                "Frontier reachability filter skipped because no move_base/make_plan service is ready.",
            )
            return False

        return True

    def _has_valid_plan(self, robot_state, frontier):
        client = self._get_make_plan_client(robot_state)
        if client is None:
            return False

        start_pose = robot_state.pose_map_to_base
        goal_pose = self.pose_registration.global_to_local(robot_state.robot_id, frontier.centroid)
        if start_pose is None or goal_pose is None:
            return False

        request_start = self._pose2d_to_pose_stamped(start_pose, robot_state.config.map_frame)
        request_goal = self._pose2d_to_pose_stamped(goal_pose, robot_state.config.map_frame)
        request = GetPlanRequest(
            start=request_start,
            goal=request_goal,
            tolerance=self.plan_tolerance,
        )

        try:
            response = client(request)
        except rospy.ServiceException as exc:
            self._make_plan_ready[robot_state.robot_id] = False
            rospy.logwarn_throttle(
                5.0,
                "make_plan call failed for %s: %s",
                robot_state.robot_id,
                str(exc),
            )
            return False

        return len(response.plan.poses) >= self.min_plan_poses

    def _get_make_plan_client(self, robot_state):
        robot_id = robot_state.robot_id
        if robot_id not in self._make_plan_clients:
            self._make_plan_clients[robot_id] = rospy.ServiceProxy(
                robot_state.config.move_base_make_plan_service,
                GetPlan,
            )
            self._make_plan_ready[robot_id] = False

        if self._make_plan_ready.get(robot_id, False):
            return self._make_plan_clients[robot_id]

        try:
            rospy.wait_for_service(
                robot_state.config.move_base_make_plan_service,
                timeout=self.make_plan_wait_timeout,
            )
            self._make_plan_ready[robot_id] = True
            return self._make_plan_clients[robot_id]
        except rospy.ROSException:
            return None

    def _pose2d_to_pose_stamped(self, pose, frame_id):
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = frame_id
        msg.pose.position.x = pose.x
        msg.pose.position.y = pose.y
        quaternion = quaternion_from_euler(0.0, 0.0, pose.yaw)
        msg.pose.orientation.x = quaternion[0]
        msg.pose.orientation.y = quaternion[1]
        msg.pose.orientation.z = quaternion[2]
        msg.pose.orientation.w = quaternion[3]
        return msg
