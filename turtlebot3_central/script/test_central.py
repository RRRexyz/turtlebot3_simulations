#!/bin/python3
import math
from dataclasses import dataclass
from nav_msgs.msg import OccupancyGrid
from typing import Optional
import rospy
from geometry_msgs.msg import Pose2D
from tf import TransformListener
from tf.transformations import euler_from_quaternion

@dataclass
class RobotState:
    """负责保存所有机器人基本信息。"""
    ns: str = "tb3_1"
    active: bool = False
    local_map: Optional[OccupancyGrid] = None
    map_stamp: Optional[rospy.Time] = None
    pose_map_to_base: Optional[Pose2D] = None
    pose_global_to_map: Optional[Pose2D] = None
    pose_global_to_base: Optional[Pose2D] = None
    current_goal_global: Optional[Pose2D] = None
    current_goal_local: Optional[Pose2D] = None
    move_base_status: str = ""
    last_goal_time: float = 0.0
    last_seen_time: float = 0.0

class PoseRegistrationManager:
    """负责维护每台机器人局部地图到全局地图的注册关系。"""
    def __init__(self):
        self.tf_listener = TransformListener()

    @staticmethod
    def _normalize_angle(theta: float) -> float:
        return math.atan2(math.sin(theta), math.cos(theta))

    @classmethod
    def _compose_pose2d(cls, lhs: Pose2D, rhs: Pose2D) -> Pose2D:
        result = Pose2D()
        lhs_cos = math.cos(lhs.theta)
        lhs_sin = math.sin(lhs.theta)
        result.x = lhs.x + lhs_cos * rhs.x - lhs_sin * rhs.y
        result.y = lhs.y + lhs_sin * rhs.x + lhs_cos * rhs.y
        result.theta = cls._normalize_angle(lhs.theta + rhs.theta)
        return result

    @classmethod
    def _inverse_pose2d(cls, pose: Pose2D) -> Pose2D:
        result = Pose2D()
        pose_cos = math.cos(pose.theta)
        pose_sin = math.sin(pose.theta)
        result.x = -pose_cos * pose.x - pose_sin * pose.y
        result.y = pose_sin * pose.x - pose_cos * pose.y
        result.theta = cls._normalize_angle(-pose.theta)
        return result

    @classmethod
    def _transform_to_pose2d(cls, trans, rot) -> Pose2D:
        result = Pose2D()
        result.x = trans[0]
        result.y = trans[1]
        _, _, yaw = euler_from_quaternion(rot)
        result.theta = cls._normalize_angle(yaw)
        return result

    def lookup_pose2d(self, target_frame: str, source_frame: str, timeout: float = 3.0) -> Pose2D:
        self.tf_listener.waitForTransform(
            target_frame,
            source_frame,
            rospy.Time(0),
            rospy.Duration(timeout),
        )
        trans, rot = self.tf_listener.lookupTransform(
            target_frame,
            source_frame,
            rospy.Time(0),
        )
        return self._transform_to_pose2d(trans, rot)

    def initialize_registration(self, robot_state: RobotState, pose_global_base_init: Pose2D, pose_local_base_init: Pose2D):
        T_global_map = self._compose_pose2d(
            pose_global_base_init,
            self._inverse_pose2d(pose_local_base_init),
        )
        robot_state.pose_global_to_map = T_global_map

    def get_local_to_global_tf(self, robot_state: RobotState, pose_in_local: Pose2D) -> Pose2D:
        if robot_state.pose_global_to_map is None:
            raise ValueError("pose_global_to_map is not initialized")
        return self._compose_pose2d(robot_state.pose_global_to_map, pose_in_local)

    def get_global_to_local_tf(self, robot_state: RobotState, pose_in_global: Pose2D) -> Pose2D:
        if robot_state.pose_global_to_map is None:
            raise ValueError("pose_global_to_map is not initialized")
        return self._compose_pose2d(
            self._inverse_pose2d(robot_state.pose_global_to_map),
            pose_in_global,
        )


if __name__ == "__main__":
    rospy.init_node("central_node")
    pose_registration_manager = PoseRegistrationManager()
    robot1_state = RobotState(ns="tb3_1")

    global_frame = "global_map"
    local_map_frame = f"{robot1_state.ns}/map"
    base_frame = f"{robot1_state.ns}/base_link"

    pose_global_base_init = pose_registration_manager.lookup_pose2d(global_frame, base_frame)
    pose_local_base_init = pose_registration_manager.lookup_pose2d(local_map_frame, base_frame)
    pose_registration_manager.initialize_registration(robot1_state, pose_global_base_init, pose_local_base_init)
    print(f"Robot {robot1_state.ns} global to map pose: {robot1_state.pose_global_to_map}")
