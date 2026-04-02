#!/usr/bin/env python3

import os

import rospy
from std_msgs.msg import Bool, String

from multi_robot_coordinator import (
    FrontierExtractor,
    FrontierFilter,
    FrontierFusion,
    GoalDispatcher,
    MapManager,
    Pose2D,
    PoseRegistrationManager,
    RobotPoseUpdater,
    RobotRegistry,
    StoppingConditionEvaluator,
    TFUtils,
    TaskAllocator,
    Visualizer,
    ExplorationMonitor,
)


class MultiRobotCoordinatorNode(object):
    def __init__(self):
        package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        robots_config = rospy.get_param(
            "~robots_config",
            os.path.join(package_root, "config", "robots.yaml"),
        )

        self.global_frame = rospy.get_param("~global_frame", "global_map")
        self.loop_hz = float(rospy.get_param("~loop_hz", 2.0))
        self.blacklist_timeout = float(rospy.get_param("~blacklist_timeout", 45.0))
        self.finished_logged = False
        self.failed_reassign_rounds = 0

        frontier_params = rospy.get_param("~frontier", {})
        assignment_params = rospy.get_param("~assignment", {})
        visualization_params = rospy.get_param("~visualization", {})
        stopping_params = rospy.get_param("~stopping_condition", {})

        self.robot_registry = RobotRegistry(robots_config)
        self.tf_utils = TFUtils(rospy.get_param("~tf_lookup_timeout", 0.2))
        self.pose_registration = PoseRegistrationManager()
        self.map_manager = MapManager(
            self.robot_registry,
            stale_timeout=rospy.get_param("~map_stale_timeout", 5.0),
        )
        self.frontier_extractor = FrontierExtractor(
            clustering_radius=frontier_params.get("local_clustering_radius", 0.45),
            min_cluster_size=frontier_params.get("min_cluster_size", 6),
        )
        self.frontier_fusion = FrontierFusion(
            self.pose_registration,
            clustering_radius=frontier_params.get("global_clustering_radius", 0.9),
        )
        self.frontier_filter = FrontierFilter(
            robot_registry=self.robot_registry,
            pose_registration=self.pose_registration,
            min_size=frontier_params.get("min_cluster_size", 6),
            min_gain=frontier_params.get("min_gain", 0.2),
            min_robot_distance=frontier_params.get("min_robot_distance", 0.5),
            min_assignment_distance=frontier_params.get("min_assignment_distance", 0.8),
            min_frontier_separation=frontier_params.get("min_frontier_separation", 0.8),
            blacklist_radius=frontier_params.get("blacklist_radius", 0.8),
            use_reachability_filter=frontier_params.get("use_reachability_filter", True),
            min_plan_poses=frontier_params.get("min_plan_poses", 2),
            plan_tolerance=frontier_params.get("plan_tolerance", 0.2),
            make_plan_wait_timeout=frontier_params.get("make_plan_wait_timeout", 0.2),
        )
        self.robot_pose_updater = RobotPoseUpdater(
            self.robot_registry,
            self.tf_utils,
            self.pose_registration,
        )
        self.task_allocator = TaskAllocator(
            distance_weight=assignment_params.get("distance_weight", 1.0),
            information_gain_weight=assignment_params.get("information_gain_weight", 2.5),
            conflict_penalty_weight=assignment_params.get("conflict_penalty_weight", 3.0),
            conflict_distance=assignment_params.get("conflict_distance", 1.0),
        )
        self.goal_dispatcher = GoalDispatcher(
            self.robot_registry,
            self.pose_registration,
            server_wait_timeout=rospy.get_param("~move_base_server_wait_timeout", 1.0),
        )
        self.exploration_monitor = ExplorationMonitor(
            self.robot_registry,
            goal_timeout=rospy.get_param("~goal_timeout", 120.0),
        )
        self.visualizer = Visualizer(
            self.global_frame,
            visualization_params.get("robot_markers_topic", "/coordinator/robots"),
            visualization_params.get("frontier_markers_topic", "/coordinator/frontiers"),
            visualization_params.get("goal_markers_topic", "/coordinator/assignments"),
        )
        self.stopping_evaluator = StoppingConditionEvaluator(
            stopping_params,
            pose_registration=self.pose_registration,
        )
        self.finished_pub = rospy.Publisher(
            self.stopping_evaluator.publish_finished_topic,
            Bool,
            queue_size=1,
            latch=True,
        )
        self.finish_reason_pub = rospy.Publisher(
            self.stopping_evaluator.publish_finish_reason_topic,
            String,
            queue_size=1,
            latch=True,
        )
        self.stop_debug_pub = rospy.Publisher(
            self.stopping_evaluator.publish_stop_debug_topic,
            String,
            queue_size=1,
            latch=True,
        )

        self.blacklisted_goals = []
        self.latest_global_frontiers = []

        self.goal_dispatcher.wait_for_servers()
        self._publish_finish_state()

    def spin(self):
        rate = rospy.Rate(self.loop_hz)
        while not rospy.is_shutdown():
            self.step()
            rate.sleep()

    def step(self):
        self.robot_pose_updater.update_all()
        self._prune_blacklist()
        self._handle_monitor_events(self.exploration_monitor.poll_events())
        self._update_local_frontiers()
        fused_frontiers = self.frontier_fusion.fuse(self.robot_registry.all_states())
        self.latest_global_frontiers = self.frontier_filter.filter(
            fused_frontiers,
            self.robot_registry.all_states(),
            active_assignments=self.robot_registry.get_active_assignments(),
            blacklisted_goals=[entry["pose"] for entry in self.blacklisted_goals],
        )
        stop_state = self.stopping_evaluator.update(
            filtered_global_frontiers=self.latest_global_frontiers,
            robot_states=self.robot_registry.all_states(),
            global_map_msg=None,
            current_time_sec=rospy.Time.now().to_sec(),
            failed_reassign_rounds=self.failed_reassign_rounds,
            manual_stop=bool(rospy.get_param("~manual_stop", False)),
            local_map_msgs=self._get_valid_local_maps(),
        )
        self._publish_finish_state()

        if stop_state.finished:
            self._handle_finished_state(stop_state.finish_reason)
        else:
            successful_dispatches = self._dispatch_assignments()
            self._update_failed_reassign_rounds(successful_dispatches)

        self.visualizer.publish(
            self.robot_registry.all_states(),
            self.latest_global_frontiers,
            self.robot_registry.get_active_assignments(),
        )

    def _update_local_frontiers(self):
        for state in self.robot_registry.all_states():
            if not state.registration_initialized:
                state.local_frontiers = []
                continue
            if not self.map_manager.is_map_valid(state.robot_id):
                state.local_frontiers = []
                continue
            if self.map_manager.is_map_stale(state.robot_id):
                state.local_frontiers = []
                continue

            map_msg = self.map_manager.get_latest_map(state.robot_id)
            state.local_frontiers = self.frontier_extractor.extract_frontiers(state.robot_id, map_msg)

    def _dispatch_assignments(self):
        assignments = self.task_allocator.allocate(
            self.robot_registry.all_states(),
            self.latest_global_frontiers,
        )
        successful_dispatches = 0
        for assignment in assignments:
            assignment.status = "PENDING"
            local_goal = self.goal_dispatcher.send_goal(assignment.robot_id, assignment.goal_global)
            if local_goal is None:
                continue
            assignment.status = "ACTIVE"
            self.robot_registry.set_busy(assignment.robot_id, assignment, local_goal)
            successful_dispatches += 1
        return successful_dispatches

    def _handle_monitor_events(self, events):
        for robot_id, event_type, message in events:
            state = self.robot_registry.get_state(robot_id)
            if state is None or state.active_assignment is None:
                continue

            if event_type in ("failed", "timeout"):
                self.goal_dispatcher.cancel_goal(robot_id)
                state.failure_count += 1
                self._blacklist_pose(state.active_assignment.goal_global)
                terminal_status = "FAILED"
            elif event_type == "reached":
                self._blacklist_pose(state.active_assignment.goal_global)
                terminal_status = "REACHED"
            else:
                terminal_status = "IDLE"

            rospy.loginfo("robot=%s event=%s detail=%s", robot_id, event_type, message)
            self.robot_registry.set_idle(robot_id, status=terminal_status)

    def _blacklist_pose(self, pose):
        self.blacklisted_goals.append(
            {
                "pose": Pose2D(pose.x, pose.y, pose.yaw),
                "expires_at": rospy.Time.now().to_sec() + self.blacklist_timeout,
            }
        )

    def _prune_blacklist(self):
        now = rospy.Time.now().to_sec()
        self.blacklisted_goals = [
            entry for entry in self.blacklisted_goals if entry["expires_at"] > now
        ]

    def _get_valid_local_maps(self):
        valid_maps = {}
        for state in self.robot_registry.all_states():
            if not self.map_manager.is_map_valid(state.robot_id):
                continue
            if self.map_manager.is_map_stale(state.robot_id):
                continue
            valid_maps[state.robot_id] = self.map_manager.get_latest_map(state.robot_id)
        return valid_maps

    def _publish_finish_state(self):
        self.finished_pub.publish(Bool(data=self.stopping_evaluator.is_finished()))
        self.finish_reason_pub.publish(String(data=self.stopping_evaluator.get_finish_reason()))
        self.stop_debug_pub.publish(String(data=self.stopping_evaluator.build_debug_string()))

    def _handle_finished_state(self, finish_reason):
        if self.stopping_evaluator.cancel_goals_on_finish:
            for state in self.robot_registry.all_states():
                if state.active_assignment is not None:
                    self.goal_dispatcher.cancel_goal(state.robot_id)
                    self.robot_registry.set_idle(state.robot_id, status="CANCELED")
        if not self.finished_logged:
            rospy.loginfo("exploration finished: %s", finish_reason)
            self.finished_logged = True

    def _update_failed_reassign_rounds(self, successful_dispatches):
        if successful_dispatches > 0:
            self.failed_reassign_rounds = 0
            return

        has_assignable_robot = any(
            state.active_assignment is None and state.goal_status not in ("ACTIVE", "PENDING")
            for state in self.robot_registry.all_states()
        )
        if self.latest_global_frontiers and has_assignable_robot:
            self.failed_reassign_rounds += 1
        else:
            self.failed_reassign_rounds = 0


def main():
    rospy.init_node("coordinator")
    node = MultiRobotCoordinatorNode()
    node.spin()


if __name__ == "__main__":
    main()
