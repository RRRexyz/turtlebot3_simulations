#!/usr/bin/env python3

import rospy


class RobotPoseUpdater(object):
    def __init__(self, robot_registry, tf_utils, pose_registration):
        self.robot_registry = robot_registry
        self.tf_utils = tf_utils
        self.pose_registration = pose_registration

    def update_all(self):
        updated_robot_ids = []
        for state in self.robot_registry.all_states():
            pose_map_to_base = self.tf_utils.lookup_pose2d(
                state.config.map_frame,
                state.config.base_frame,
            )
            if pose_map_to_base is None:
                continue

            state.pose_map_to_base = pose_map_to_base
            state.last_tf_update = rospy.Time.now()

            if not state.registration_initialized:
                self.pose_registration.initialize_registration(
                    state.robot_id,
                    state.config.spawn_pose,
                    pose_map_to_base,
                )
                state.registration_initialized = True

            state.pose_global_to_base = self.pose_registration.local_to_global(
                state.robot_id,
                pose_map_to_base,
            )
            updated_robot_ids.append(state.robot_id)

        return updated_robot_ids
