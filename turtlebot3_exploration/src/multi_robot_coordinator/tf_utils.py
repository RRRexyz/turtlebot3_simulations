#!/usr/bin/env python3

import math

import rospy
import tf2_ros
from tf.transformations import euler_from_quaternion

from .data_types import Pose2D


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def pose2d_to_matrix(pose):
    cos_yaw = math.cos(pose.yaw)
    sin_yaw = math.sin(pose.yaw)
    return [
        [cos_yaw, -sin_yaw, pose.x],
        [sin_yaw, cos_yaw, pose.y],
        [0.0, 0.0, 1.0],
    ]


def matrix_to_pose2d(matrix):
    return Pose2D(
        x=matrix[0][2],
        y=matrix[1][2],
        yaw=normalize_angle(math.atan2(matrix[1][0], matrix[0][0])),
    )


def compose_pose2d(lhs, rhs):
    cos_yaw = math.cos(lhs.yaw)
    sin_yaw = math.sin(lhs.yaw)
    return Pose2D(
        x=lhs.x + cos_yaw * rhs.x - sin_yaw * rhs.y,
        y=lhs.y + sin_yaw * rhs.x + cos_yaw * rhs.y,
        yaw=normalize_angle(lhs.yaw + rhs.yaw),
    )


def inverse_pose2d(pose):
    cos_yaw = math.cos(pose.yaw)
    sin_yaw = math.sin(pose.yaw)
    return Pose2D(
        x=-(cos_yaw * pose.x + sin_yaw * pose.y),
        y=(sin_yaw * pose.x - cos_yaw * pose.y),
        yaw=normalize_angle(-pose.yaw),
    )


def quaternion_to_yaw(quaternion):
    return euler_from_quaternion(
        [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
    )[2]


def transform_to_pose2d(transform):
    return Pose2D(
        x=transform.transform.translation.x,
        y=transform.transform.translation.y,
        yaw=quaternion_to_yaw(transform.transform.rotation),
    )


class TFUtils(object):
    def __init__(self, lookup_timeout=0.2):
        self.buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
        self.listener = tf2_ros.TransformListener(self.buffer)
        self.lookup_timeout = rospy.Duration(lookup_timeout)

    def lookup_pose2d(self, target_frame, source_frame):
        try:
            transform = self.buffer.lookup_transform(
                target_frame,
                source_frame,
                rospy.Time(0),
                self.lookup_timeout,
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
            tf2_ros.TransformException,
        ):
            return None

        return transform_to_pose2d(transform)
