#!/usr/bin/env python3
import math
from dataclasses import dataclass

import rospy
from nav_msgs.msg import OccupancyGrid
from tf import TransformBroadcaster, TransformListener
from tf.transformations import euler_from_quaternion, quaternion_from_euler


@dataclass
class Pose2D:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0


class GlobalMapFuser:
    def __init__(self):
        self.robot_namespaces = rospy.get_param("~robot_namespaces", ["tb3_1", "tb3_2", "tb3_3"])
        self.map_topic = rospy.get_param("~robot_map_topic", "map")
        self.base_frame_suffix = rospy.get_param("~base_frame_suffix", "base_footprint")
        self.map_frame_suffix = rospy.get_param("~map_frame_suffix", "map")
        self.odom_frame_suffix = rospy.get_param("~odom_frame_suffix", "odom")
        self.global_frame = rospy.get_param("~global_frame", "map")
        self.publish_rate = float(rospy.get_param("~publish_rate", 1.0))
        self.tf_publish_rate = float(rospy.get_param("~tf_publish_rate", 30.0))
        self.tf_time_offset = float(rospy.get_param("~tf_time_offset", 0.1))
        self.output_resolution = float(rospy.get_param("~resolution", 0.05))
        self.occupied_threshold = int(rospy.get_param("~occupied_threshold", 50))
        self.publish_tf = bool(rospy.get_param("~publish_tf", True))
        self.canvas_padding = float(rospy.get_param("~canvas_padding", 1.0))
        self.canvas_expand_trigger = float(rospy.get_param("~canvas_expand_trigger", 0.5))

        self.tf_listener = TransformListener()
        self.tf_broadcaster = TransformBroadcaster() if self.publish_tf else None
        self.latest_maps = {}
        self.subscribers = []
        self.canvas_bounds = None

        for namespace in self.robot_namespaces:
            topic = f"/{namespace}/{self.map_topic.lstrip('/')}"
            subscriber = rospy.Subscriber(topic, OccupancyGrid, self._make_map_callback(namespace), queue_size=1)
            self.subscribers.append(subscriber)

        self.publisher = rospy.Publisher("/map", OccupancyGrid, queue_size=1, latch=True)
        if self.publish_tf:
            rospy.Timer(rospy.Duration(max(1.0 / self.tf_publish_rate, 0.02)), self._tf_timer_callback)
        rospy.Timer(rospy.Duration(max(1.0 / self.publish_rate, 0.1)), self._timer_callback)

    @staticmethod
    def _normalize_angle(theta):
        return math.atan2(math.sin(theta), math.cos(theta))

    @classmethod
    def _compose_pose2d(cls, lhs, rhs):
        lhs_cos = math.cos(lhs.theta)
        lhs_sin = math.sin(lhs.theta)
        return Pose2D(
            x=lhs.x + lhs_cos * rhs.x - lhs_sin * rhs.y,
            y=lhs.y + lhs_sin * rhs.x + lhs_cos * rhs.y,
            theta=cls._normalize_angle(lhs.theta + rhs.theta),
        )

    @classmethod
    def _inverse_pose2d(cls, pose):
        pose_cos = math.cos(pose.theta)
        pose_sin = math.sin(pose.theta)
        return Pose2D(
            x=-pose_cos * pose.x - pose_sin * pose.y,
            y=pose_sin * pose.x - pose_cos * pose.y,
            theta=cls._normalize_angle(-pose.theta),
        )

    @staticmethod
    def _quaternion_to_yaw(quaternion):
        _, _, yaw = euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
        return yaw

    def _make_map_callback(self, namespace):
        def _callback(message):
            self.latest_maps[namespace] = message
        return _callback

    def _lookup_pose2d(self, target_frame, source_frame):
        self.tf_listener.waitForTransform(target_frame, source_frame, rospy.Time(0), rospy.Duration(0.5))
        trans, rot = self.tf_listener.lookupTransform(target_frame, source_frame, rospy.Time(0))
        _, _, yaw = euler_from_quaternion(rot)
        return Pose2D(x=trans[0], y=trans[1], theta=yaw)

    def _lookup_global_to_map(self, namespace):
        map_frame = f"{namespace}/{self.map_frame_suffix}"
        odom_frame = f"{namespace}/{self.odom_frame_suffix}"
        base_frame = f"{namespace}/{self.base_frame_suffix}"
        global_to_base = self._lookup_pose2d(odom_frame, base_frame)
        local_map_to_base = self._lookup_pose2d(map_frame, base_frame)
        return self._compose_pose2d(global_to_base, self._inverse_pose2d(local_map_to_base))

    def _map_origin_in_global(self, global_to_map, map_message):
        origin = map_message.info.origin
        origin_pose = Pose2D(
            x=origin.position.x,
            y=origin.position.y,
            theta=self._quaternion_to_yaw(origin.orientation),
        )
        return self._compose_pose2d(global_to_map, origin_pose)

    def _publish_map_registration_tf(self, namespace, global_to_map):
        if self.tf_broadcaster is None:
            return

        quaternion = quaternion_from_euler(0.0, 0.0, global_to_map.theta)
        self.tf_broadcaster.sendTransform(
            (global_to_map.x, global_to_map.y, 0.0),
            quaternion,
            rospy.Time.now() + rospy.Duration(self.tf_time_offset),
            f"{namespace}/{self.map_frame_suffix}",
            self.global_frame,
        )

    def _transform_cell_to_global(self, origin_pose, resolution, cell_x, cell_y):
        local_x = (cell_x + 0.5) * resolution
        local_y = (cell_y + 0.5) * resolution
        origin_cos = math.cos(origin_pose.theta)
        origin_sin = math.sin(origin_pose.theta)
        return (
            origin_pose.x + origin_cos * local_x - origin_sin * local_y,
            origin_pose.y + origin_sin * local_x + origin_cos * local_y,
        )

    def _compute_bounds(self, map_states):
        min_x = float("inf")
        min_y = float("inf")
        max_x = float("-inf")
        max_y = float("-inf")

        for _, map_message, origin_pose in map_states:
            width = map_message.info.width
            height = map_message.info.height
            resolution = map_message.info.resolution
            corners = [
                self._transform_cell_to_global(origin_pose, resolution, 0, 0),
                self._transform_cell_to_global(origin_pose, resolution, width, 0),
                self._transform_cell_to_global(origin_pose, resolution, 0, height),
                self._transform_cell_to_global(origin_pose, resolution, width, height),
            ]
            for x_coord, y_coord in corners:
                min_x = min(min_x, x_coord)
                min_y = min(min_y, y_coord)
                max_x = max(max_x, x_coord)
                max_y = max(max_y, y_coord)

        return min_x, min_y, max_x, max_y

    def _snap_floor(self, value, resolution):
        return math.floor(value / resolution) * resolution

    def _snap_ceil(self, value, resolution):
        return math.ceil(value / resolution) * resolution

    def _build_canvas_bounds(self, min_x, min_y, max_x, max_y, resolution):
        padding = max(0.0, self.canvas_padding)
        return (
            self._snap_floor(min_x - padding, resolution),
            self._snap_floor(min_y - padding, resolution),
            self._snap_ceil(max_x + padding, resolution),
            self._snap_ceil(max_y + padding, resolution),
        )

    def _update_canvas_bounds(self, observed_bounds, resolution):
        observed_min_x, observed_min_y, observed_max_x, observed_max_y = observed_bounds

        if self.canvas_bounds is None:
            self.canvas_bounds = self._build_canvas_bounds(
                observed_min_x, observed_min_y, observed_max_x, observed_max_y, resolution
            )
            rospy.loginfo(
                "Initialized fused map canvas: min=(%.2f, %.2f) max=(%.2f, %.2f)",
                self.canvas_bounds[0], self.canvas_bounds[1], self.canvas_bounds[2], self.canvas_bounds[3]
            )
            return self.canvas_bounds

        min_x, min_y, max_x, max_y = self.canvas_bounds
        trigger = max(0.0, self.canvas_expand_trigger)
        expanded = False

        if observed_min_x < (min_x + trigger):
            min_x = self._snap_floor(observed_min_x - self.canvas_padding, resolution)
            expanded = True
        if observed_min_y < (min_y + trigger):
            min_y = self._snap_floor(observed_min_y - self.canvas_padding, resolution)
            expanded = True
        if observed_max_x > (max_x - trigger):
            max_x = self._snap_ceil(observed_max_x + self.canvas_padding, resolution)
            expanded = True
        if observed_max_y > (max_y - trigger):
            max_y = self._snap_ceil(observed_max_y + self.canvas_padding, resolution)
            expanded = True

        if expanded:
            self.canvas_bounds = (min_x, min_y, max_x, max_y)
            rospy.loginfo(
                "Expanded fused map canvas: min=(%.2f, %.2f) max=(%.2f, %.2f)",
                min_x, min_y, max_x, max_y
            )

        return self.canvas_bounds

    def _collect_map_states(self, publish_tf=False):
        map_states = []
        for namespace in self.robot_namespaces:
            map_message = self.latest_maps.get(namespace)
            if map_message is None:
                continue
            try:
                global_to_map = self._lookup_global_to_map(namespace)
            except Exception as error:
                rospy.logwarn_throttle(5.0, f"Failed to resolve registration for {namespace}: {error}")
                continue
            if publish_tf:
                self._publish_map_registration_tf(namespace, global_to_map)
            map_states.append((namespace, map_message, self._map_origin_in_global(global_to_map, map_message)))
        return map_states

    def _tf_timer_callback(self, _event):
        self._collect_map_states(publish_tf=True)

    def _timer_callback(self, _event):
        map_states = self._collect_map_states(publish_tf=False)

        if not map_states:
            return

        resolution = self.output_resolution if self.output_resolution > 0.0 else map_states[0][1].info.resolution
        observed_bounds = self._compute_bounds(map_states)
        min_x, min_y, max_x, max_y = self._update_canvas_bounds(observed_bounds, resolution)
        width = max(1, int(math.ceil((max_x - min_x) / resolution)))
        height = max(1, int(math.ceil((max_y - min_y) / resolution)))

        occupied = [False] * (width * height)
        free = [False] * (width * height)

        for _, map_message, origin_pose in map_states:
            source_resolution = map_message.info.resolution
            source_width = map_message.info.width
            source_height = map_message.info.height
            for y_index in range(source_height):
                row_offset = y_index * source_width
                for x_index in range(source_width):
                    value = map_message.data[row_offset + x_index]
                    if value < 0:
                        continue

                    world_x, world_y = self._transform_cell_to_global(origin_pose, source_resolution, x_index, y_index)
                    global_x = int(math.floor((world_x - min_x) / resolution))
                    global_y = int(math.floor((world_y - min_y) / resolution))
                    if global_x < 0 or global_x >= width or global_y < 0 or global_y >= height:
                        continue

                    merged_index = global_y * width + global_x
                    if value >= self.occupied_threshold:
                        occupied[merged_index] = True
                    else:
                        free[merged_index] = True

        merged_data = []
        for cell_index in range(width * height):
            if occupied[cell_index]:
                merged_data.append(100)
            elif free[cell_index]:
                merged_data.append(0)
            else:
                merged_data.append(-1)

        merged_map = OccupancyGrid()
        merged_map.header.stamp = rospy.Time.now()
        merged_map.header.frame_id = self.global_frame
        merged_map.info.map_load_time = merged_map.header.stamp
        merged_map.info.resolution = resolution
        merged_map.info.width = width
        merged_map.info.height = height
        merged_map.info.origin.position.x = min_x
        merged_map.info.origin.position.y = min_y
        merged_map.info.origin.position.z = 0.0
        merged_map.info.origin.orientation.w = 1.0
        merged_map.data = merged_data
        self.publisher.publish(merged_map)


if __name__ == "__main__":
    rospy.init_node("global_map_fuser")
    GlobalMapFuser()
    rospy.spin()
