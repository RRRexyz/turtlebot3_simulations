#!/bin/python3
import rospy
import tf
import math
from sensor_msgs.msg import LaserScan

class MultiRobotMaskNode:
    def __init__(self):
        rospy.init_node('multi_robot_mask_node')

        self.ns = rospy.get_param('~robot_namespace', 'tb3_1')
        other_robots = rospy.get_param('~other_robots', 'tb3_2,tb3_3')
        self.other_frames = [r + '/base_footprint' for r in other_robots.split(',') if r]
        # 放大掩码半径，提供宽裕余量防止边角鬼影
        self.robot_radius = rospy.get_param('~robot_radius', 0.28)

        self.tf_listener = tf.TransformListener()
        self.pub = rospy.Publisher('scan_filtered', LaserScan, queue_size=10)
        rospy.Subscriber('scan', LaserScan, self.scan_cb, queue_size=10)

    def get_robot_position_local(self, target_frame: str, source_frame: str):
        try:
            (trans, rot) = self.tf_listener.lookupTransform(target_frame, source_frame, rospy.Time(0))
            return (trans[0], trans[1])
        except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return None

    def scan_cb(self, msg: LaserScan):
        filtered = LaserScan()
        filtered.header = msg.header
        filtered.angle_min = msg.angle_min
        filtered.angle_increment = msg.angle_increment
        filtered.time_increment = msg.time_increment
        filtered.scan_time = msg.scan_time
        filtered.range_min = msg.range_min
        filtered.range_max = msg.range_max
        filtered.ranges = list(msg.ranges)
        filtered.intensities = list(msg.intensities)

        # Gazebo's full-circle scan can be off by one beam relative to the
        # count Karto expects. Normalize the outgoing scan for slam_toolbox.
        expected_count = int(round((msg.angle_max - msg.angle_min) / msg.angle_increment))
        if expected_count > 0 and len(filtered.ranges) == expected_count + 1:
            filtered.ranges = filtered.ranges[:expected_count]
            if filtered.intensities:
                filtered.intensities = filtered.intensities[:expected_count]

        if filtered.ranges:
            filtered.angle_max = filtered.angle_min + len(filtered.ranges) * filtered.angle_increment
        else:
            filtered.angle_max = msg.angle_max

        other_positions = []
        for frame in self.other_frames:
            pos = self.get_robot_position_local(msg.header.frame_id, frame)
            if pos is not None:
                other_positions.append(pos)

        if not other_positions:
            self.pub.publish(filtered)
            return

        masked_count = 0
        # 给半径加上 1.5 倍余量，防止非圆形碰撞箱(比如waffle四个角)逃过检测
        eff_radius = self.robot_radius * 1.5

        for i, r in enumerate(filtered.ranges):
            if math.isnan(r) or r < msg.range_min or r > msg.range_max:
                continue

            angle_i = msg.angle_min + i * msg.angle_increment
            mask_this_ray = False

            for rx, ry in other_positions:
                dist_to_robot = math.hypot(rx, ry)
                angle_to_robot = math.atan2(ry, rx)

                if dist_to_robot <= eff_radius:
                    mask_this_ray = True
                    break

                # 计算目标机器人在视角中占据的角度半宽
                angular_width = math.asin(eff_radius / dist_to_robot)

                diff = (angle_i - angle_to_robot)
                diff = (diff + math.pi) % (2 * math.pi) - math.pi

                if abs(diff) <= angular_width:
                    # 确保是真正受到机器人阻挡，而不是机器人视野背后很远的墙面
                    if r <= dist_to_robot + eff_radius:
                        mask_this_ray = True
                        break

            if mask_this_ray:
                # 使用 inf 代替 nan：
                # nan 会被 SLAM 完全忽略（不更新地图）；使用 inf 会使这束射线主动清空路径上的障碍，
                # 这种"透视光线"特性可以积极地消除由于网络或TF瞬时延迟画出的"历史鬼影"。
                filtered.ranges[i] = float('inf')
                masked_count += 1

        if masked_count > 0:
            rospy.loginfo_throttle(1.0, f"({self.ns}) Masked {masked_count} points corresponding to other robots.")

        self.pub.publish(filtered)

if __name__ == '__main__':
    node = MultiRobotMaskNode()
    rospy.spin()
