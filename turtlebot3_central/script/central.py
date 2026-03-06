#!/bin/python3
import rospy
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
from collections import deque
import tf

class FrontierExplorer:
    def __init__(self):
        self.map_data = None
        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        self.marker_pub = rospy.Publisher("/frontier_markers", MarkerArray, queue_size=10)
        self.assignment_pub = rospy.Publisher("/assignment_markers", MarkerArray, queue_size=10)
        
        # 为不同机器人定义颜色
        self.robot_colors = {
            'tb3_0': (1.0, 0.0, 0.0),  # 红色
            'tb3_1': (0.0, 0.0, 1.0),  # 蓝色
            'tb3_2': (1.0, 1.0, 0.0),  # 黄色
            'tb3_3': (1.0, 0.0, 1.0),  # 品红
            'tb3_4': (0.0, 1.0, 1.0),  # 青色
        }

    def map_callback(self, msg):
        self.map_data = msg

    def detect_frontiers(self):
        if self.map_data is None:
            return list()
        
        height = self.map_data.info.height
        width = self.map_data.info.width

        frontiers = list()
        data = np.array(self.map_data.data).reshape((height, width))

        # 定义邻居方向 (4邻域)
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        # 访问矩阵，标记已访问的栅格
        visited = np.zeros_like(data, dtype=bool)

        for y in range(height):
            for x in range(width):
                # 1. 初筛：必须是自由栅格 (值在 [0, 50) 范围)
                if not visited[y, x] and 0 <= data[y, x] < 50:
                    # 检查是否是 Frontier (邻居有未知区域 -1)
                    is_frontier = False
                    for dx, dy in neighbors:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            if data[ny, nx] == -1:
                                is_frontier = True
                                break
                
                    if is_frontier:
                        # 2. 开始 BFS 聚类
                        cluster = list()
                        queue = deque([(x, y)])
                        visited[y, x] = True

                        while queue:
                            curr_x, curr_y = queue.popleft()
                            cluster.append((curr_x, curr_y))

                            for dx, dy in neighbors:
                                nx, ny = curr_x + dx, curr_y + dy
                                if 0 <= nx < width and 0 <= ny < height:
                                    # 只加入自由的且未被访问的 Frontier 栅格
                                    if not visited[ny, nx] and 0 <= data[ny, nx] < 50:
                                        # 再次确认该邻居是否连接未知区域
                                        is_still_frontier = False
                                        for ddx, ddy in neighbors:
                                            nnx, nny = nx + ddx, ny + ddy
                                            if 0 <= nnx < width and 0 <= nny < height:
                                                if data[nny, nnx] == -1:
                                                    is_still_frontier = True
                                                    break

                                        if is_still_frontier:
                                            visited[ny, nx] = True
                                            queue.append((nx, ny))

                        # 3. 过滤并计算中心点
                        if len(cluster) > 5: # 阈值，忽略小簇
                            # 计算平均坐标
                            avg_x = np.mean([p[0] for p in cluster])
                            avg_y = np.mean([p[1] for p in cluster])

                            # 转换为世界坐标
                            world_x = self.map_data.info.origin.position.x + avg_x * self.map_data.info.resolution
                            world_y = self.map_data.info.origin.position.y + avg_y * self.map_data.info.resolution

                            frontiers.append((world_x, world_y, len(cluster))) # （x, y, cluster_size）

        return frontiers
    
    def publish_markers(self, frontiers):
        """发布frontier中心点的可视化标记"""
        marker_array = MarkerArray()
        
        for i, (x, y, size) in enumerate(frontiers):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "frontiers"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            # 设置位置
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            
            # 设置大小（根据cluster大小调整）
            scale = min(0.3 + size * 0.01, 1.0)
            marker.scale.x = scale
            marker.scale.y = scale
            marker.scale.z = scale
            
            # 设置颜色（绿色）
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.8
            
            marker.lifetime = rospy.Duration(2.0)
            
            marker_array.markers.append(marker)
        
        # 如果没有frontier，发布一个删除所有标记的消息
        if len(frontiers) == 0:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "frontiers"
            marker.id = 0
            marker.action = Marker.DELETEALL
            marker_array.markers.append(marker)
        
        self.marker_pub.publish(marker_array)

    def get_robot_positions(self, listener: tf.TransformListener):
        robot_positions = dict()
        for name in ['tb3_1', 'tb3_2', 'tb3_3']:
            try:
                # 查询 map -> tb3_X/base_footprint 的变换
                (trans, rot) = listener.lookupTransform("/map", f"/{name}/base_link", rospy.Time(0))
                x, y, _ = trans
                robot_positions[name] = (x, y)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rospy.logwarn(f"Cannot get position of {name}")
        return robot_positions

    def assign_tasks(self, frontiers, robot_positions):
        # frontiers: 列表 [(x, y, size), ...]
        # robot_positions: 字典 { 'tb3_1': (x, y), 'tb3_2': (x, y) }

        assignments = dict()
        # 复制一份用于删除已分配的
        available_frontiers = frontiers.copy()

        # 按机器人顺序分配
        for robot_name, pos in robot_positions.items():
            if not available_frontiers:
                break  # 没有frontier可分配

            best_frontier = None
            max_utility = -1
            for f in available_frontiers:
                fx, fy, fsize = f
                # 计算距离
                dist = np.sqrt((fx - pos[0])**2 + (fy - pos[1])**2)
                if dist < 0.1:
                    dist = 0.1  # 避免除以零

                # 收益函数：大小/距离
                utility = fsize / dist

                # 可以加惩罚项，比如如果这个目标太靠近另一个机器人正在去的目标

                if utility > max_utility:
                    max_utility = utility
                    best_frontier = f

            if best_frontier:
                assignments[robot_name] = best_frontier
                available_frontiers.remove(best_frontier)

        return assignments
    
    def publish_assignments(self, assignments):
        """发布为每个机器人分配的目标点标记
        
        Args:
            assignments: 字典 { 'robot_name': (x, y, size), ... }
        """
        marker_array = MarkerArray()
        
        for i, (robot_name, frontier) in enumerate(assignments.items()):
            x, y, size = frontier
            
            # 创建球形标记
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "assignments"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            # 设置位置
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.2  # 稍微抬高以便区分
            marker.pose.orientation.w = 1.0
            
            # 设置大小（稍大以便突出显示）
            marker.scale.x = 0.4
            marker.scale.y = 0.4
            marker.scale.z = 0.4
            
            # 根据机器人名称设置颜色
            if robot_name in self.robot_colors:
                r, g, b = self.robot_colors[robot_name]
            else:
                # 默认使用橙色
                r, g, b = 1.0, 0.5, 0.0
            
            marker.color.r = r
            marker.color.g = g
            marker.color.b = b
            marker.color.a = 1.0  # 完全不透明
            
            marker.lifetime = rospy.Duration(2.0)
            
            marker_array.markers.append(marker)
            
            # 为每个分配添加文本标签
            text_marker = Marker()
            text_marker.header.frame_id = "map"
            text_marker.header.stamp = rospy.Time.now()
            text_marker.ns = "assignment_labels"
            text_marker.id = i + 1000  # 避免ID冲突
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            
            text_marker.pose.position.x = x
            text_marker.pose.position.y = y
            text_marker.pose.position.z = 0.5  # 文字在球体上方
            text_marker.pose.orientation.w = 1.0
            
            text_marker.scale.z = 0.2  # 文字大小
            
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            
            text_marker.text = robot_name
            text_marker.lifetime = rospy.Duration(2.0)
            
            marker_array.markers.append(text_marker)
        
        # 如果没有分配，发布删除所有标记的消息
        if len(assignments) == 0:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "assignments"
            marker.id = 0
            marker.action = Marker.DELETEALL
            marker_array.markers.append(marker)
            
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "assignment_labels"
            marker.id = 0
            marker.action = Marker.DELETEALL
            marker_array.markers.append(marker)
        
        self.assignment_pub.publish(marker_array)


if __name__ == "__main__":
    rospy.init_node("frontier_explorer")
    explorer = FrontierExplorer()
    listener = tf.TransformListener()
    rate = rospy.Rate(1) # 1 Hz

    while not rospy.is_shutdown():
        frontiers = explorer.detect_frontiers()
        # rospy.loginfo(f"Detected {len(frontiers)} frontiers: {frontiers}")
        robot_positions = explorer.get_robot_positions(listener)
        # rospy.loginfo(f"Robot positions: {robot_positions}")
        assignments = explorer.assign_tasks(frontiers, robot_positions)
        rospy.loginfo(f"Assignments: {assignments}")
        explorer.publish_assignments(assignments)
        explorer.publish_markers(frontiers)
        rate.sleep()