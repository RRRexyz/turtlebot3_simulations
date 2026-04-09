#!/bin/python3
import rospy
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from nav_msgs.srv import GetPlan, GetPlanRequest
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import cv2
from collections import deque
import tf
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from tf.transformations import quaternion_from_euler


class FrontierExplorer:
    def __init__(self):
        self.global_map_topic = rospy.get_param("~global_map_topic", "/map")
        self.global_frame = rospy.get_param("~global_frame", "map")
        self.robot_names = rospy.get_param("~robot_namespaces", ['tb3_1', 'tb3_2', 'tb3_3'])
        self.base_frame_suffix = rospy.get_param("~base_frame_suffix", "base_footprint")
        self.distance_weight = float(rospy.get_param("~assignment_distance_weight", 1.0))
        self.information_gain_weight = float(rospy.get_param("~assignment_information_gain_weight", 1.5))
        self.conflict_penalty_weight = float(rospy.get_param("~assignment_conflict_penalty_weight", 3.0))
        self.conflict_distance = float(rospy.get_param("~assignment_conflict_distance", 1.5))
        self.map_data = None
        self.map_sub = rospy.Subscriber(self.global_map_topic, OccupancyGrid, self.map_callback)
        self.marker_pub = rospy.Publisher("/frontier_markers", MarkerArray, queue_size=10)
        self.assignment_pub = rospy.Publisher("/assignment_markers", MarkerArray, queue_size=10)
        
        # 为不同机器人定义颜色
        self.robot_colors = {
            'tb3_1': (0.0, 0.0, 1.0),  # 蓝色
            'tb3_2': (1.0, 1.0, 0.0),  # 黄色
            'tb3_3': (1.0, 0.0, 1.0),  # 品红
        }

    def map_callback(self, msg):
        """对订阅到的地图应用滤波，去除因为互相观测导致的虚假小障碍物连通域"""
        try:
            width = msg.info.width
            height = msg.info.height
            
            # 将占据栅格数据转为 numpy，方便图像处理
            data = np.array(msg.data, dtype=np.int8).reshape((height, width))
            
            # 创建仅包含障碍物的二值化图像 (假设障碍物 > 50 概率)
            obstacle_mask = (data > 50).astype(np.uint8) * 255
            
            # 寻找所有连通域
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(obstacle_mask, connectivity=8)
            
            # 背景色占据了label 0, 所以从1开始遍历
            # stats 中包含每个连通域的具体面积 (像素数)
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                # 设定阈值：根据 Turtlebot3 投影面积(约30个栅格)来过滤小面积噪点或动态机器人残影
                if area < 30:
                    # 将这一块小面积障碍物设回自由区域(0)
                    obstacle_mask[labels == i] = 0
            
            # 提取清理过的小连通域的掩膜，修改原地图数据
            cleaned_obstacles = (obstacle_mask == 255)
            # 原本是障碍物，但连通域被标记清理后，我们将其重写为自由空间 (0)
            data[(data > 50) & ~cleaned_obstacles] = 0
            
            # 重写回消息
            msg.data = tuple(data.flatten())
            self.map_data = msg
            
        except Exception as e:
            rospy.logwarn(f"Map filter error: {e}")
            self.map_data = msg

    def get_known_area(self):
        """计算当前地图中已知区域（值为 0-100 的栅格数）"""
        if self.map_data is None:
            return 0
        data = np.array(self.map_data.data)
        known_cells = np.sum((data >= 0) & (data <= 100))
        return known_cells

    def detect_frontiers(self):
        if self.map_data is None:
            return list()
        
        height = self.map_data.info.height
        width = self.map_data.info.width

        frontiers = list()
        data = np.array(self.map_data.data).reshape((height, width))

        # 定义邻居方向 (8邻域)
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]

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
                        if len(cluster) > 10: # 阈值，忽略小簇
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
            marker.header.frame_id = self.global_frame
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
            marker.header.frame_id = self.global_frame
            marker.header.stamp = rospy.Time.now()
            marker.ns = "frontiers"
            marker.id = 0
            marker.action = Marker.DELETEALL
            marker_array.markers.append(marker)
        
        self.marker_pub.publish(marker_array)

    def get_robot_positions(self, listener: tf.TransformListener):
        robot_positions = dict()
        for name in self.robot_names:
            try:
                target_frame = f"/{self.global_frame}"
                source_frame = f"/{name}/{self.base_frame_suffix}"
                (trans, rot) = listener.lookupTransform(target_frame, source_frame, rospy.Time(0))
                x, y, _ = trans
                robot_positions[name] = (x, y)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rospy.logwarn(f"Cannot get position of {name}")
        return robot_positions

    def _assignment_score(self, robot_position, frontier, reserved_goals):
        fx, fy, fsize = frontier
        distance = np.hypot(fx - robot_position[0], fy - robot_position[1])
        distance = max(distance, 0.1)
        information_gain = np.log1p(max(fsize, 0))

        conflict_penalty = 0.0
        for goal in reserved_goals:
            gx, gy, _ = goal
            distance_to_reserved = np.hypot(fx - gx, fy - gy)
            if distance_to_reserved < self.conflict_distance:
                conflict_penalty += (
                    (self.conflict_distance - distance_to_reserved)
                    / max(self.conflict_distance, 1e-6)
                )

        return (
            self.distance_weight * distance
            - self.information_gain_weight * information_gain
            + self.conflict_penalty_weight * conflict_penalty
        )

    def assign_tasks(self, frontiers, robot_positions, idle_robots, active_goals=None,
                     blacklist_checker=None, reachability_checker=None):
        """为空闲机器人分配frontier目标
        
        Args:
            frontiers: 列表 [(x, y, size), ...]
            robot_positions: 字典 { 'tb3_1': (x, y), 'tb3_2': (x, y) }
            idle_robots: 空闲机器人名称列表
            active_goals: 字典 { 'robot_name': (x, y, size), ... } 正在执行的目标
            
        Returns:
            assignments: 字典 { 'robot_name': (x, y, size), ... }
        """
        if active_goals is None:
            active_goals = {}
            
        assignments = dict()
        remaining_idle_robots = [name for name in idle_robots if name in robot_positions]
        available_frontiers = [
            frontier for frontier in frontiers
            if blacklist_checker is None or not blacklist_checker(frontier)
        ]

        while remaining_idle_robots and available_frontiers:
            best_candidate = None
            reserved_goals = list(active_goals.values()) + list(assignments.values())

            for robot_name in remaining_idle_robots:
                robot_position = robot_positions[robot_name]
                for frontier in available_frontiers:
                    if reachability_checker is not None and not reachability_checker(robot_name, frontier):
                        continue

                    score = self._assignment_score(robot_position, frontier, reserved_goals)
                    if best_candidate is None or score < best_candidate[0]:
                        best_candidate = (score, robot_name, frontier)

            if best_candidate is None:
                break

            _, robot_name, frontier = best_candidate
            assignments[robot_name] = frontier
            active_goals[robot_name] = frontier
            remaining_idle_robots = [name for name in remaining_idle_robots if name != robot_name]
            available_frontiers = [item for item in available_frontiers if item != frontier]

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
            marker.header.frame_id = self.global_frame
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
            
            marker.lifetime = rospy.Duration(0)  # 0表示永久显示，需要手动删除
            
            marker_array.markers.append(marker)
            
            # 为每个分配添加文本标签
            text_marker = Marker()
            text_marker.header.frame_id = self.global_frame
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
            text_marker.lifetime = rospy.Duration(0)  # 0表示永久显示
            
            marker_array.markers.append(text_marker)
        
        # 如果没有分配，发布删除所有标记的消息
        if len(assignments) == 0:
            marker = Marker()
            marker.header.frame_id = self.global_frame
            marker.header.stamp = rospy.Time.now()
            marker.ns = "assignments"
            marker.id = 0
            marker.action = Marker.DELETEALL
            marker_array.markers.append(marker)
            
            marker = Marker()
            marker.header.frame_id = self.global_frame
            marker.header.stamp = rospy.Time.now()
            marker.ns = "assignment_labels"
            marker.id = 0
            marker.action = Marker.DELETEALL
            marker_array.markers.append(marker)
        
        self.assignment_pub.publish(marker_array)


class TaskAssigner:
    def __init__(self):
        self.global_frame = rospy.get_param("~global_frame", "map")
        self.robot_names = rospy.get_param("~robot_namespaces", ['tb3_1', 'tb3_2', 'tb3_3'])
        self.base_frame_suffix = rospy.get_param("~base_frame_suffix", "base_footprint")
        self.make_plan_wait_timeout = float(rospy.get_param("~make_plan_wait_timeout", 0.2))
        self.plan_tolerance = float(rospy.get_param("~plan_tolerance", 0.2))
        self.min_plan_poses = int(rospy.get_param("~min_plan_poses", 2))
        self.validation_robot_limit = int(rospy.get_param("~validation_robot_limit", 2))
        self.goal_blacklist_timeout = float(rospy.get_param("~goal_blacklist_timeout", 45.0))
        self.goal_blacklist_radius = float(rospy.get_param("~goal_blacklist_radius", 0.8))
        self.no_gain_min_cells = int(rospy.get_param("~no_gain_min_cells", 20))
        self.max_no_gain_attempts = int(rospy.get_param("~max_no_gain_attempts", 3))
        self.frontier_retirement_radius = float(rospy.get_param("~frontier_retirement_radius", 0.8))
        self.clients = {
            name: actionlib.SimpleActionClient(f'/{name}/move_base', MoveBaseAction)
            for name in self.robot_names
        }
        self.make_plan_clients = {
            name: rospy.ServiceProxy(f'/{name}/move_base/make_plan', GetPlan)
            for name in self.robot_names
        }
        self.make_plan_ready = {name: False for name in self.robot_names}
        self.tf_listener = tf.TransformListener()
        
        # 机器人状态跟踪：IDLE 或 BUSY
        self.robot_status = {name: 'IDLE' for name in self.robot_names}
        
        # 记录当前分配给每个机器人的目标
        self.current_goals = {}
        
        # 记录每个机器人开始执行目标的时间
        self.goal_start_times = {}
        self.goal_start_known_areas = {}
        self.goal_blacklist = []
        self.frontier_no_gain_stats = {}
        self.retired_frontiers = {}
        self.latest_known_area = 0

        # 连接服务器
        for name, client in self.clients.items():
            rospy.loginfo(f"Waiting for {name} move_base server...")
            client.wait_for_server()
            rospy.loginfo(f"{name} move_base server connected.")

    def prune_blacklist(self):
        now_sec = rospy.Time.now().to_sec()
        self.goal_blacklist = [entry for entry in self.goal_blacklist if entry["expires_at"] > now_sec]

    def update_known_area(self, known_area):
        self.latest_known_area = known_area

    def _frontier_key(self, goal_coords):
        resolution = max(self.frontier_retirement_radius, 1e-3)
        return (
            int(np.floor(goal_coords[0] / resolution)),
            int(np.floor(goal_coords[1] / resolution)),
        )

    def blacklist_goal(self, goal_coords, reason="failed_goal"):
        if goal_coords is None:
            return
        self.prune_blacklist()
        self.goal_blacklist.append(
            {
                "goal": (goal_coords[0], goal_coords[1]),
                "expires_at": rospy.Time.now().to_sec() + self.goal_blacklist_timeout,
                "reason": reason,
            }
        )
        rospy.loginfo(
            "Blacklisted goal (%.2f, %.2f) for %.1fs due to %s",
            goal_coords[0], goal_coords[1], self.goal_blacklist_timeout, reason
        )

    def retire_frontier(self, goal_coords, reason="no_gain"):
        if goal_coords is None:
            return
        frontier_key = self._frontier_key(goal_coords)
        if frontier_key in self.retired_frontiers:
            return
        self.retired_frontiers[frontier_key] = {
            "goal": (goal_coords[0], goal_coords[1]),
            "reason": reason,
            "retired_at": rospy.Time.now().to_sec(),
        }
        rospy.loginfo(
            "Retired frontier (%.2f, %.2f) due to %s",
            goal_coords[0], goal_coords[1], reason
        )

    def is_retired(self, goal_coords):
        return self._frontier_key(goal_coords) in self.retired_frontiers

    def is_blacklisted(self, goal_coords):
        self.prune_blacklist()
        if self.is_retired(goal_coords):
            return True
        for entry in self.goal_blacklist:
            gx, gy = entry["goal"]
            if np.hypot(goal_coords[0] - gx, goal_coords[1] - gy) < self.goal_blacklist_radius:
                return True
        return False

    def _record_no_gain_attempt(self, goal_coords, known_area_gain):
        frontier_key = self._frontier_key(goal_coords)
        entry = self.frontier_no_gain_stats.get(
            frontier_key,
            {"count": 0, "goal": (goal_coords[0], goal_coords[1])},
        )
        entry["count"] += 1
        entry["goal"] = (goal_coords[0], goal_coords[1])
        self.frontier_no_gain_stats[frontier_key] = entry
        rospy.loginfo(
            "Frontier (%.2f, %.2f) produced only %d known cells gain (%d/%d)",
            goal_coords[0], goal_coords[1], known_area_gain,
            entry["count"], self.max_no_gain_attempts
        )
        if entry["count"] >= self.max_no_gain_attempts:
            self.retire_frontier(goal_coords, reason="long_term_no_gain")

    def _clear_no_gain_attempts(self, goal_coords):
        frontier_key = self._frontier_key(goal_coords)
        if frontier_key in self.frontier_no_gain_stats:
            del self.frontier_no_gain_stats[frontier_key]

    def _lookup_robot_position(self, robot_name):
        target_frame = f"/{self.global_frame}"
        source_frame = f"/{robot_name}/{self.base_frame_suffix}"
        trans, _ = self.tf_listener.lookupTransform(target_frame, source_frame, rospy.Time(0))
        return trans[0], trans[1]

    def _build_pose_stamped(self, x_coord, y_coord, yaw=0.0):
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = self.global_frame
        pose.pose.position.x = x_coord
        pose.pose.position.y = y_coord
        quaternion = quaternion_from_euler(0.0, 0.0, yaw)
        pose.pose.orientation.x = quaternion[0]
        pose.pose.orientation.y = quaternion[1]
        pose.pose.orientation.z = quaternion[2]
        pose.pose.orientation.w = quaternion[3]
        return pose

    def _get_make_plan_client(self, robot_name):
        if self.make_plan_ready.get(robot_name, False):
            return self.make_plan_clients[robot_name]
        try:
            rospy.wait_for_service(
                f'/{robot_name}/move_base/make_plan',
                timeout=self.make_plan_wait_timeout,
            )
            self.make_plan_ready[robot_name] = True
            return self.make_plan_clients[robot_name]
        except rospy.ROSException:
            rospy.logwarn_throttle(
                5.0,
                "make_plan service not ready for %s, skipping reachability filtering",
                robot_name,
            )
            return None

    def _candidate_robot_names_for_frontier(self, frontier, robot_positions):
        ranked = []
        for robot_name in self.robot_names:
            if robot_name not in robot_positions:
                continue
            rx, ry = robot_positions[robot_name]
            distance = np.hypot(frontier[0] - rx, frontier[1] - ry)
            ranked.append((distance, robot_name))
        ranked.sort(key=lambda item: item[0])
        if self.validation_robot_limit > 0:
            ranked = ranked[:self.validation_robot_limit]
        return [robot_name for _, robot_name in ranked]

    def is_goal_reachable(self, robot_name, goal_coords, start_coords=None):
        if self.is_blacklisted(goal_coords):
            return False

        client = self._get_make_plan_client(robot_name)
        if client is None:
            return True

        if start_coords is None:
            try:
                start_coords = self._lookup_robot_position(robot_name)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rospy.logwarn("Cannot get start pose for %s, skipping reachability filtering", robot_name)
                return True

        request = GetPlanRequest(
            start=self._build_pose_stamped(start_coords[0], start_coords[1]),
            goal=self._build_pose_stamped(goal_coords[0], goal_coords[1]),
            tolerance=self.plan_tolerance,
        )
        try:
            response = client(request)
        except rospy.ServiceException as exc:
            self.make_plan_ready[robot_name] = False
            rospy.logwarn_throttle(5.0, "make_plan failed for %s: %s", robot_name, str(exc))
            return False

        reachable = len(response.plan.poses) >= self.min_plan_poses
        if not reachable:
            rospy.loginfo_throttle(
                5.0,
                "Frontier (%.2f, %.2f) rejected for %s: no valid global plan",
                goal_coords[0], goal_coords[1], robot_name
            )
        return reachable

    def filter_valid_frontiers(self, frontiers, robot_positions):
        valid_frontiers = []
        if not robot_positions:
            return []

        for frontier in frontiers:
            if self.is_blacklisted(frontier):
                continue

            for robot_name in self._candidate_robot_names_for_frontier(frontier, robot_positions):
                if self.is_goal_reachable(robot_name, frontier, robot_positions.get(robot_name)):
                    valid_frontiers.append(frontier)
                    break

        return valid_frontiers

    def send_goal(self, robot_name: str, goal_coords: tuple, known_area=None):
        """发布导航目标到指定机器人

        Args:
            robot_name: 机器人名称，如 'tb3_1'
            goal_coords: 目标坐标 (x, y, size)，map坐标系
        """
        goal = MoveBaseGoal()
        
        # 通过 global_map_fuser 发布的动态 TF，由 move_base 将全局目标转换到各自的局部 map 坐标系
        goal.target_pose.header.frame_id = self.global_frame
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = goal_coords[0]
        goal.target_pose.pose.position.y = goal_coords[1]
        goal.target_pose.pose.orientation.w = 1.0
        
        # 标记机器人为忙碌状态
        self.robot_status[robot_name] = 'BUSY'
        self.current_goals[robot_name] = goal_coords
        self.goal_start_times[robot_name] = rospy.Time.now()
        self.goal_start_known_areas[robot_name] = self.latest_known_area if known_area is None else known_area
        
        # 发送目标，使用 lambda 来传递 robot_name
        self.clients[robot_name].send_goal(
            goal, 
            done_cb=lambda status, result: self.goal_done_callback(robot_name, status, result)
        )
        rospy.loginfo(f"Sent goal to {robot_name}: ({goal_coords[0]:.2f}, {goal_coords[1]:.2f})")

    def goal_done_callback(self, robot_name, status, result):
        """目标完成回调
        
        Args:
            robot_name: 机器人名称
            status: 目标状态 (actionlib.GoalStatus)
            result: 结果对象
        """
        goal_coords = self.current_goals.get(robot_name)
        self.robot_status[robot_name] = 'IDLE'

        if robot_name in self.current_goals:
            del self.current_goals[robot_name]
        if robot_name in self.goal_start_times:
            del self.goal_start_times[robot_name]
        start_known_area = self.goal_start_known_areas.pop(robot_name, self.latest_known_area)

        status_names = {
            GoalStatus.PENDING: 'PENDING',
            GoalStatus.ACTIVE: 'ACTIVE',
            GoalStatus.PREEMPTED: 'PREEMPTED',
            GoalStatus.SUCCEEDED: 'SUCCEEDED',
            GoalStatus.ABORTED: 'ABORTED',
            GoalStatus.REJECTED: 'REJECTED',
            GoalStatus.PREEMPTING: 'PREEMPTING',
            GoalStatus.RECALLING: 'RECALLING',
            GoalStatus.RECALLED: 'RECALLED',
            GoalStatus.LOST: 'LOST',
        }
        status_name = status_names.get(status, f'UNKNOWN({status})')
        if status == GoalStatus.SUCCEEDED and goal_coords is not None:
            known_area_gain = max(0, self.latest_known_area - start_known_area)
            if known_area_gain < self.no_gain_min_cells:
                self._record_no_gain_attempt(goal_coords, known_area_gain)
            else:
                self._clear_no_gain_attempts(goal_coords)
        if status in (GoalStatus.ABORTED, GoalStatus.REJECTED, GoalStatus.LOST):
            self.blacklist_goal(goal_coords, reason=status_name.lower())
        rospy.loginfo(f"{robot_name} goal completed with status: {status_name}")
    
    def get_idle_robots(self):
        """获取当前空闲的机器人列表"""
        return [name for name, status in self.robot_status.items() if status == 'IDLE']
    
    def all_robots_idle(self):
        """检查是否所有机器人都空闲"""
        return all(status == 'IDLE' for status in self.robot_status.values())
    
    def check_timeouts(self, timeout_sec=40.0):
        """检查是否有机器人的目标执行超时"""
        current_time = rospy.Time.now()
        for robot_name, status in list(self.robot_status.items()):
            if status == 'BUSY' and robot_name in self.goal_start_times:
                elapsed = (current_time - self.goal_start_times[robot_name]).to_sec()
                if elapsed > timeout_sec:
                    rospy.logwarn(f"Goal for {robot_name} timed out ({timeout_sec}s). Cancelling...")
                    self.blacklist_goal(self.current_goals.get(robot_name), reason="timeout")
                    self.clients[robot_name].cancel_goal()
                    self.robot_status[robot_name] = 'IDLE'
                    if robot_name in self.current_goals:
                        del self.current_goals[robot_name]
                    if robot_name in self.goal_start_times:
                        del self.goal_start_times[robot_name]
                    if robot_name in self.goal_start_known_areas:
                        del self.goal_start_known_areas[robot_name]
    
    def cancel_all_goals(self):
        """取消所有机器人的当前目标"""
        for robot_name, client in self.clients.items():
            if self.robot_status[robot_name] == 'BUSY':
                client.cancel_goal()
                rospy.loginfo(f"Cancelled goal for {robot_name}")
        self.robot_status = {name: 'IDLE' for name in self.robot_status.keys()}
        self.current_goals.clear()
        self.goal_start_times.clear()
        self.goal_start_known_areas.clear()


class StoppingConditionEvaluator:
    def __init__(self):
        self.enabled = bool(rospy.get_param("~stopping_condition_enabled", True))
        self.stop_confirm_cycles = int(rospy.get_param("~stop_confirm_cycles", 8))
        self.map_growth_window_sec = float(rospy.get_param("~map_growth_window_sec", 15.0))
        self.known_cells_growth_abs_threshold = int(
            rospy.get_param("~known_cells_growth_abs_threshold", 80)
        )
        self.known_cells_growth_ratio_threshold = float(
            rospy.get_param("~known_cells_growth_ratio_threshold", 0.005)
        )
        self.max_exploration_runtime_sec = float(
            rospy.get_param("~max_exploration_runtime_sec", 1800.0)
        )
        self.known_area_history = deque()
        self.stop_counter = 0
        self.finished = False
        self.finish_reason = ""
        self.start_time_sec = rospy.Time.now().to_sec()

    def update(self, valid_frontiers, all_idle, known_cells, current_time_sec, raw_frontier_count=0):
        if self.finished:
            return True, self.finish_reason, self.build_debug_string(
                valid_frontiers, all_idle, known_cells, 0, 0.0, raw_frontier_count
            )

        runtime_sec = max(0.0, current_time_sec - self.start_time_sec)
        self._update_known_area_history(current_time_sec, known_cells)
        growth_abs, growth_ratio = self._compute_growth_metrics()
        growth_small = self._is_growth_small(current_time_sec, growth_abs, growth_ratio)
        no_valid_frontiers = len(valid_frontiers) == 0
        composite_reached = (
            self.enabled
            and no_valid_frontiers
            and all_idle
            and growth_small
        )

        if composite_reached:
            self.stop_counter += 1
        else:
            self.stop_counter = 0

        if self.max_exploration_runtime_sec > 0.0 and runtime_sec >= self.max_exploration_runtime_sec:
            self.finished = True
            self.finish_reason = "timeout"
        elif self.stop_counter >= self.stop_confirm_cycles:
            self.finished = True
            self.finish_reason = "frontier_exhausted_and_map_stagnant"

        debug_string = self.build_debug_string(
            valid_frontiers, all_idle, known_cells, growth_abs, growth_ratio, raw_frontier_count
        )
        return self.finished, self.finish_reason, debug_string

    def _update_known_area_history(self, current_time_sec, known_cells):
        if self.known_area_history and self.known_area_history[-1][0] == current_time_sec:
            self.known_area_history[-1] = (current_time_sec, known_cells)
        else:
            self.known_area_history.append((current_time_sec, known_cells))

        while len(self.known_area_history) > 1:
            next_entry_time = self.known_area_history[1][0]
            if next_entry_time < (current_time_sec - self.map_growth_window_sec):
                self.known_area_history.popleft()
            else:
                break

    def _compute_growth_metrics(self):
        if not self.known_area_history:
            return 0, 0.0
        oldest_time, oldest_known_cells = self.known_area_history[0]
        _, current_known_cells = self.known_area_history[-1]
        growth_abs = current_known_cells - oldest_known_cells
        growth_ratio = float(growth_abs) / float(max(oldest_known_cells, 1))
        return growth_abs, growth_ratio

    def _is_growth_small(self, current_time_sec, growth_abs, growth_ratio):
        if not self.known_area_history:
            return False
        oldest_time = self.known_area_history[0][0]
        if (current_time_sec - oldest_time) < self.map_growth_window_sec:
            return False
        return (
            growth_abs < self.known_cells_growth_abs_threshold
            or growth_ratio < self.known_cells_growth_ratio_threshold
        )

    def build_debug_string(self, valid_frontiers, all_idle, known_cells, growth_abs, growth_ratio, raw_frontier_count):
        return (
            f"raw_frontiers={raw_frontier_count} "
            f"valid_frontiers={len(valid_frontiers)} "
            f"all_idle={str(all_idle).lower()} "
            f"known_cells={known_cells} "
            f"growth_abs={growth_abs} "
            f"growth_ratio={growth_ratio:.6f} "
            f"stop_counter={self.stop_counter} "
            f"finished={str(self.finished).lower()} "
            f"finish_reason={self.finish_reason or 'none'}"
        )


if __name__ == "__main__":
    rospy.init_node("frontier_explorer")
    
    # 初始化探索器和控制器
    explorer = FrontierExplorer()
    task_assigner = TaskAssigner()
    stopping_evaluator = StoppingConditionEvaluator()
    tf_listener = tf.TransformListener()
    
    # 等待一段时间让TF数据准备好
    rospy.sleep(2.0)
    rospy.loginfo("Starting multi-robot frontier exploration...")
    
    rate = rospy.Rate(1)  # 1 Hz
    exploration_completed = False

    while not rospy.is_shutdown():
        try:
            task_assigner.prune_blacklist()
            # 检查目标是否超时 (40秒)
            task_assigner.check_timeouts(40.0)

            # 步骤1: 检查状态 - 获取空闲机器人
            idle_robots = task_assigner.get_idle_robots()
            all_idle = task_assigner.all_robots_idle()
            
            # 步骤2: 获取位姿
            robot_positions = explorer.get_robot_positions(tf_listener)
            
            # 步骤3: 检测 Frontier
            frontiers = explorer.detect_frontiers()
            valid_frontiers = task_assigner.filter_valid_frontiers(frontiers, robot_positions)
            rospy.loginfo(
                f"Detected {len(frontiers)} raw frontiers, {len(valid_frontiers)} valid frontiers, {len(idle_robots)} robots idle"
            )
            
            # 发布可视化标记
            explorer.publish_markers(frontiers)
            
            current_time_sec = rospy.Time.now().to_sec()
            current_known_area = explorer.get_known_area()
            task_assigner.update_known_area(current_known_area)
            finished, finish_reason, stop_debug = stopping_evaluator.update(
                valid_frontiers=valid_frontiers,
                all_idle=all_idle,
                known_cells=current_known_area,
                current_time_sec=current_time_sec,
                raw_frontier_count=len(frontiers),
            )
            rospy.loginfo_throttle(5.0, f"Stopping check: {stop_debug}")

            # 步骤4: 判断终止 - 连续多个周期满足“无frontier + 全部空闲 + 地图增长停滞”才真正结束。
            if exploration_completed or finished:
                if not exploration_completed:
                    rospy.loginfo("="*50)
                    rospy.loginfo(f"Exploration completed! reason={finish_reason}")
                    rospy.loginfo("="*50)
                    exploration_completed = True
                    task_assigner.cancel_all_goals()
                
                # 持续保持结束状态以更新可视化，不再分配新任务且不重置完成状态
                rate.sleep()
                continue
            
            # 步骤5: 任务分配 - 只为空闲机器人分配
            if len(idle_robots) > 0 and len(valid_frontiers) > 0:
                # 传入其他机器人正在执行的目标（不包含本循环中的空闲机器人，避免自身的影响）
                active_goals_copy = task_assigner.current_goals.copy()
                assignments = explorer.assign_tasks(
                    valid_frontiers,
                    robot_positions,
                    idle_robots,
                    active_goals=active_goals_copy,
                )
                
                # 步骤6: 执行 - 发送目标到机器人
                if len(assignments) > 0:
                    rospy.loginfo(f"Assigning {len(assignments)} tasks...")
                    for robot_name, frontier in assignments.items():
                        # 步骤7: 发送目标会自动标记为忙碌
                        task_assigner.send_goal(robot_name, frontier, known_area=current_known_area)
                else:
                    rospy.loginfo("No assignments made (all idle robots may be too far or no suitable frontiers)")
            
            # 持续发布当前所有活跃的任务分配标记（包括正在执行的）
            explorer.publish_assignments(task_assigner.current_goals)
            
            rate.sleep()
            
        except KeyboardInterrupt:
            rospy.loginfo("Keyboard interrupt, cancelling all goals...")
            task_assigner.cancel_all_goals()
            break
        except Exception as e:
            rospy.logerr(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
            rate.sleep()
    
    rospy.loginfo("Exploration node shutting down...")
