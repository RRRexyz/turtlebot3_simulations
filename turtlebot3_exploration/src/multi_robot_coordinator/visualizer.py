#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Point
from tf.transformations import quaternion_from_euler
from visualization_msgs.msg import Marker, MarkerArray


class Visualizer(object):
    def __init__(self, global_frame, robot_topic, frontier_topic, goal_topic):
        self.global_frame = global_frame
        self.robot_pub = rospy.Publisher(robot_topic, MarkerArray, queue_size=1)
        self.frontier_pub = rospy.Publisher(frontier_topic, MarkerArray, queue_size=1)
        self.goal_pub = rospy.Publisher(goal_topic, MarkerArray, queue_size=1)

    def publish(self, robot_states, frontiers, assignments):
        self.robot_pub.publish(self._robot_markers(robot_states))
        self.frontier_pub.publish(self._frontier_markers(frontiers))
        self.goal_pub.publish(self._goal_markers(robot_states, assignments))

    def _robot_markers(self, robot_states):
        marker_array = MarkerArray()
        marker_array.markers.append(self._delete_all_marker("robots"))
        for index, state in enumerate(robot_states):
            if state.pose_global_to_base is None:
                continue
            marker = Marker()
            marker.header.frame_id = self.global_frame
            marker.header.stamp = rospy.Time.now()
            marker.ns = "robots"
            marker.id = index
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose.position.x = state.pose_global_to_base.x
            marker.pose.position.y = state.pose_global_to_base.y
            quaternion = quaternion_from_euler(0.0, 0.0, state.pose_global_to_base.yaw)
            marker.pose.orientation.x = quaternion[0]
            marker.pose.orientation.y = quaternion[1]
            marker.pose.orientation.z = quaternion[2]
            marker.pose.orientation.w = quaternion[3]
            marker.scale.x = 0.35
            marker.scale.y = 0.12
            marker.scale.z = 0.12
            marker.color.r = 0.1
            marker.color.g = 0.8
            marker.color.b = 0.2
            marker.color.a = 0.9
            marker_array.markers.append(marker)
        return marker_array

    def _frontier_markers(self, frontiers):
        marker_array = MarkerArray()
        marker_array.markers.append(self._delete_all_marker("frontiers"))
        for index, frontier in enumerate(frontiers):
            marker = Marker()
            marker.header.frame_id = self.global_frame
            marker.header.stamp = rospy.Time.now()
            marker.ns = "frontiers"
            marker.id = index
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = frontier.centroid.x
            marker.pose.position.y = frontier.centroid.y
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.22
            marker.scale.y = 0.22
            marker.scale.z = 0.22
            marker.color.r = 0.0
            marker.color.g = 0.6
            marker.color.b = 1.0
            marker.color.a = 0.85
            marker_array.markers.append(marker)
        return marker_array

    def _goal_markers(self, robot_states, assignments):
        marker_array = MarkerArray()
        marker_array.markers.append(self._delete_all_marker("goals"))
        robot_lookup = {state.robot_id: state for state in robot_states}

        for index, assignment in enumerate(assignments):
            robot_state = robot_lookup.get(assignment.robot_id)
            if robot_state is None or robot_state.pose_global_to_base is None:
                continue

            marker = Marker()
            marker.header.frame_id = self.global_frame
            marker.header.stamp = rospy.Time.now()
            marker.ns = "goals"
            marker.id = index
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.05
            marker.color.r = 1.0
            marker.color.g = 0.55
            marker.color.b = 0.0
            marker.color.a = 0.9
            marker.points = [
                Point(robot_state.pose_global_to_base.x, robot_state.pose_global_to_base.y, 0.0),
                Point(assignment.goal_global.x, assignment.goal_global.y, 0.0),
            ]
            marker_array.markers.append(marker)
        return marker_array

    def _delete_all_marker(self, namespace):
        marker = Marker()
        marker.header.frame_id = self.global_frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = namespace
        marker.action = Marker.DELETEALL
        return marker
