#!/usr/bin/env python3

from .tf_utils import compose_pose2d, inverse_pose2d


class PoseRegistrationManager(object):
    def __init__(self):
        self._registrations = {}

    def initialize_registration(self, robot_id, global_base_init, local_base_init):
        registration = compose_pose2d(global_base_init, inverse_pose2d(local_base_init))
        self._registrations[robot_id] = registration
        return registration

    def local_to_global(self, robot_id, pose_local):
        registration = self.get_registration(robot_id)
        if registration is None or pose_local is None:
            return None
        return compose_pose2d(registration, pose_local)

    def global_to_local(self, robot_id, pose_global):
        registration = self.get_registration(robot_id)
        if registration is None or pose_global is None:
            return None
        return compose_pose2d(inverse_pose2d(registration), pose_global)

    def get_registration(self, robot_id):
        return self._registrations.get(robot_id)
