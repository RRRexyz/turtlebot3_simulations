#!/bin/python3
import rospy
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
from collections import deque
import tf
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal


class FrontierExplorer:
    def __init__(self):
        self.map_data = None
        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        self.marker_pub = rospy.Publisher("/frontier_markers", MarkerArray, queue_size=10)
        self.assignment_pub = rospy.Publisher("/assignment_markers", MarkerArray, queue_size=10)
        
        # 为不同机器人定义颜色
        self.robot_colors = {
            'tb3_1': (0.0, 0.0, 1.0),  # 蓝色
            'tb3_2': (1.0, 1.0, 0.0),  # 黄色
            'tb3_3': (1.0, 0.0, 1.0),  # 品红
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

    def assign_tasks(self, frontiers, robot_positions, idle_robots):
        """为空闲机器人分配frontier目标
        
        Args:
            frontiers: 列表 [(x, y, size), ...]
            robot_positions: 字典 { 'tb3_1': (x, y), 'tb3_2': (x, y) }
            idle_robots: 空闲机器人名称列表
            
        Returns:
            assignments: 字典 { 'robot_name': (x, y, size), ... }
        """
        assignments = dict()
        # 复制一份用于删除已分配的
        available_frontiers = frontiers.copy()

        # 只为空闲的机器人分配任务
        for robot_name in idle_robots:
            if robot_name not in robot_positions:
                rospy.logwarn(f"Cannot get position for {robot_name}, skipping assignment")
                continue
            
            if not available_frontiers:
                break  # 没有frontier可分配

            pos = robot_positions[robot_name]
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
            
            marker.lifetime = rospy.Duration(0)  # 0表示永久显示，需要手动删除
            
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
            text_marker.lifetime = rospy.Duration(0)  # 0表示永久显示
            
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


class CentralController:
    def __init__(self):
        self.clients = {
            'tb3_1': actionlib.SimpleActionClient('/tb3_1/move_base', MoveBaseAction),
            'tb3_2': actionlib.SimpleActionClient('/tb3_2/move_base', MoveBaseAction),
            'tb3_3': actionlib.SimpleActionClient('/tb3_3/move_base', MoveBaseAction),
        }
        
        # 机器人状态跟踪：IDLE 或 BUSY
        self.robot_status = {
            'tb3_1': 'IDLE',
            'tb3_2': 'IDLE',
            'tb3_3': 'IDLE',
        }
        
        # 记录当前分配给每个机器人的目标
        self.current_goals = {}

        # 连接服务器
        for name, client in self.clients.items():
            rospy.loginfo(f"Waiting for {name} move_base server...")
            client.wait_for_server()
            rospy.loginfo(f"{name} move_base server connected.")

    def send_goal(self, robot_name: str, goal_coords: tuple):
        """发布导航目标到指定机器人

        Args:
            robot_name: 机器人名称，如 'tb3_1'
            goal_coords: 目标坐标 (x, y, size)，map坐标系
        """
        goal = MoveBaseGoal()
        
        # 直接使用map坐标系的目标点
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = goal_coords[0]
        goal.target_pose.pose.position.y = goal_coords[1]
        goal.target_pose.pose.orientation.w = 1.0
        
        # 标记机器人为忙碌状态
        self.robot_status[robot_name] = 'BUSY'
        self.current_goals[robot_name] = goal_coords
        
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
        # 标记机器人为空闲
        self.robot_status[robot_name] = 'IDLE'
        
        if robot_name in self.current_goals:
            del self.current_goals[robot_name]
        
        status_names = {
            0: 'PENDING',
            1: 'ACTIVE',
            2: 'PREEMPTED',
            3: 'SUCCEEDED',
            4: 'ABORTED',
            5: 'REJECTED'
        }
        status_name = status_names.get(status, f'UNKNOWN({status})')
        rospy.loginfo(f"{robot_name} goal completed with status: {status_name}")
    
    def get_idle_robots(self):
        """获取当前空闲的机器人列表"""
        return [name for name, status in self.robot_status.items() if status == 'IDLE']
    
    def all_robots_idle(self):
        """检查是否所有机器人都空闲"""
        return all(status == 'IDLE' for status in self.robot_status.values())
    
    def cancel_all_goals(self):
        """取消所有机器人的当前目标"""
        for robot_name, client in self.clients.items():
            if self.robot_status[robot_name] == 'BUSY':
                client.cancel_goal()
                rospy.loginfo(f"Cancelled goal for {robot_name}")
        self.robot_status = {name: 'IDLE' for name in self.robot_status.keys()}
        self.current_goals.clear()


if __name__ == "__main__":
    rospy.init_node("frontier_explorer")
    
    # 初始化探索器和控制器
    explorer = FrontierExplorer()
    controller = CentralController()
    tf_listener = tf.TransformListener()
    
    # 等待一段时间让TF数据准备好
    rospy.sleep(2.0)
    rospy.loginfo("Starting multi-robot frontier exploration...")
    
    rate = rospy.Rate(1)  # 1 Hz
    exploration_completed = False

    while not rospy.is_shutdown():
        try:
            # 步骤1: 检查状态 - 获取空闲机器人
            idle_robots = controller.get_idle_robots()
            all_idle = controller.all_robots_idle()
            
            # 步骤2: 获取位姿
            robot_positions = explorer.get_robot_positions(tf_listener)
            
            # 步骤3: 检测 Frontier
            frontiers = explorer.detect_frontiers()
            rospy.loginfo(f"Detected {len(frontiers)} frontiers, {len(idle_robots)} robots idle")
            
            # 发布可视化标记
            explorer.publish_markers(frontiers)
            
            # 步骤4: 判断终止 - 如果没有frontier且所有机器人都空闲，探索完成
            if len(frontiers) == 0 and all_idle:
                if not exploration_completed:
                    rospy.loginfo("="*50)
                    rospy.loginfo("Exploration completed! No more frontiers.")
                    rospy.loginfo("="*50)
                    exploration_completed = True
                # 继续循环以保持可视化更新，但不分配新任务
                rate.sleep()
                continue
            
            # 如果有新的frontier出现，重置完成标志
            if len(frontiers) > 0:
                exploration_completed = False
            
            # 步骤5: 任务分配 - 只为空闲机器人分配
            if len(idle_robots) > 0 and len(frontiers) > 0:
                assignments = explorer.assign_tasks(frontiers, robot_positions, idle_robots)
                
                # 步骤6: 执行 - 发送目标到机器人
                if len(assignments) > 0:
                    rospy.loginfo(f"Assigning {len(assignments)} tasks...")
                    for robot_name, frontier in assignments.items():
                        # 步骤7: 发送目标会自动标记为忙碌
                        controller.send_goal(robot_name, frontier)
                else:
                    rospy.loginfo("No assignments made (all idle robots may be too far or no suitable frontiers)")
            
            # 持续发布当前所有活跃的任务分配标记（包括正在执行的）
            explorer.publish_assignments(controller.current_goals)
            
            rate.sleep()
            
        except KeyboardInterrupt:
            rospy.loginfo("Keyboard interrupt, cancelling all goals...")
            controller.cancel_all_goals()
            break
        except Exception as e:
            rospy.logerr(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
            rate.sleep()
    
    rospy.loginfo("Exploration node shutting down...")