你是一个资深 ROS1 多机器人系统工程师。请在一个现有的 ROS1 Python 工程中，实现一个“基于 pose-registration 的多机器人中央协调器”，用于替代原先依赖 multirobot_map_merge 的协调器主逻辑。

# 项目背景
我当前的系统是：
- ROS1
- TurtleBot3 官方 Gazebo 仿真
- 每台机器人独立运行 slam_toolbox
- 每台机器人独立运行 move_base
- 每台机器人拥有独立坐标链：robot_i/map -> robot_i/odom -> robot_i/base_link
- 中央协调器需要做多机器人 frontier 协同探索
- 不再把 multirobot_map_merge 作为控制主输入
- 目标是通过 pose-registration 的方式，在统一 global_map 坐标系中融合各机器人局部 frontier，并完成任务分配

# 目标
请为我实现一个可运行的 ROS1 Python 包，名称为：
turtlebot3_exploration

要求实现以下核心能力：
1. 读取多机器人配置
2. 维护机器人状态表
3. 查询 TF 并统一成 2D pose 表示
4. 根据机器人 spawn pose 和初始 map->base pose 建立：
   T(global_map <- robot_i/map)
5. 订阅每台机器人的局部 OccupancyGrid
6. 在各自局部地图上提取 frontier
7. 将局部 frontier 投影到 global_map
8. 对全局 frontier 做聚类、去重、过滤
9. 给空闲机器人分配 frontier
10. 将 global_map 下的目标转换回对应机器人 local map 坐标
11. 通过 move_base action 发送目标
12. 监控 move_base 状态，支持 timeout / failed / reached 后重分配
13. 发布 RViz 调试 markers

# 关键架构约束
请严格遵守以下设计：
- 协调器不依赖 multirobot_map_merge 作为主控制输入
- frontier 必须先在每台机器人 local map 上独立提取
- 统一全局融合必须通过 pose registration 完成，而不是先做全局 OccupancyGrid 拼接
- move_base 目标发送前，必须先把 global 目标转换到该机器人自己的 map 坐标系
- 代码 Python 为主
- ROS1 风格，使用 rospy、tf/tf2、actionlib、nav_msgs、move_base_msgs、visualization_msgs
- 优先保证清晰、模块化、可调试
- 不要把所有逻辑堆到一个超长脚本里

# 需要生成的文件结构
请生成或补全以下文件：
注意你现在已经在turtlebot3_exploration文件夹下

turtlebot3_exploration/
  launch/
    coordinator.launch
  config/
    robots.yaml
    coordinator.yaml
    frontier.yaml
  scripts/
    coordinator_node.py
  src/multi_robot_coordinator/
    __init__.py
    data_types.py
    tf_utils.py
    robot_registry.py
    pose_registration.py
    map_manager.py
    frontier_extractor.py
    frontier_fusion.py
    frontier_filter.py
    robot_pose_updater.py
    task_allocator.py
    goal_dispatcher.py
    exploration_monitor.py
    visualizer.py

# 需要定义的数据结构
请优先定义这些 dataclass 或等价类：
- Pose2D
- RobotConfig
- RobotState
- FrontierLocal
- FrontierGlobal
- Assignment

# 功能要求
请分别实现以下模块：

## 1. robot_registry.py
- 从 YAML 读取机器人配置
- 维护 robot_id -> RobotState
- 保存 namespace、frame 名、topic 名、spawn pose

## 2. tf_utils.py
- lookup_pose2d(target_frame, source_frame)
- pose2d_to_matrix
- matrix_to_pose2d
- compose_pose2d
- inverse_pose2d

## 3. pose_registration.py
- initialize_registration(robot_id, global_base_init, local_base_init)
- local_to_global(robot_id, pose_local)
- global_to_local(robot_id, pose_global)
- get_registration(robot_id)

数学上使用：
T(global <- robot_map) = T(global <- robot_base at t0) * inverse(T(robot_map <- robot_base at t0))

## 4. map_manager.py
- 订阅每台机器人 /<ns>/map
- 缓存最新 OccupancyGrid
- 提供 get_latest_map / is_map_valid / is_map_stale

## 5. frontier_extractor.py
- 在单张 OccupancyGrid 上提取 frontier cells
- 对 frontier cells 聚类
- 计算 cluster centroid
- 估计 gain

## 6. frontier_fusion.py
- 将各机器人局部 frontier 投影到 global_map
- 进行空间聚类和去重
- 输出 FrontierGlobal 列表

## 7. frontier_filter.py
- 过滤太小、太近、无效、已分配或明显不可达的 frontier
- 第一版可以先实现基础规则，不要求复杂路径规划判定

## 8. robot_pose_updater.py
- 更新每台机器人的 pose_map_to_base
- 更新每台机器人的 pose_global_to_base

## 9. task_allocator.py
- 只给空闲机器人分配任务
- 用贪心分配
- 代价函数至少包含：
  - distance cost
  - information gain
  - conflict penalty

## 10. goal_dispatcher.py
- 为每台机器人创建 move_base action client
- 将 global 目标转换到 local map
- 发送 MoveBaseGoal
- 支持 cancel_goal

## 11. exploration_monitor.py
- 读取 move_base status
- 判断 goal reached / timeout / failed
- 支持触发重分配

## 12. visualizer.py
- 发布 robot markers
- 发布 frontier markers
- 发布 assigned goal markers

# 主节点要求
coordinator_node.py 应该：
1. 初始化所有模块
2. 周期性刷新地图和 TF
3. 周期性提取 local frontiers
4. 做 global frontier 融合和过滤
5. 更新机器人状态
6. 为空闲机器人分配目标
7. 下发 move_base goal
8. 处理 timeout / failure / reached
9. 发布可视化 markers

# 配置要求
请提供合理默认值：
- frontier clustering radius
- frontier min size
- assignment weights
- goal timeout
- visualization topic names
- 主循环频率

# 代码要求
- Python 代码应包含必要注释
- 不要写伪代码，要写接近可运行的真实代码
- 对未确定的 topic/frame，集中写在 robots.yaml 和 coordinator.yaml 中
- 尽量避免硬编码
- 对 TF 查询失败、空地图、action server 未连接等情况做错误处理
- 所有函数要有清晰的输入输出
- 优先使用小函数和清晰类边界

# 输出要求
请按以下顺序输出：
1. 项目目录树
2. 每个文件的完整代码
3. launch 和 yaml 配置
4. 运行说明
5. 已知限制
6. 下一步待办事项

在开始写代码前，请先给出：
- 你的实现计划
- 模块依赖关系
- 任何你需要我确认的假设
如果假设可以合理默认，请直接默认，不要停下来等我确认。


下一步待办事项

  - 把 robots.yaml 改成与你当前 TurtleBot3 多机 launch 完全一致的 namespace、
    frame 和 spawn pose。
  - 增加基于 make_plan 或 navfn 的可达性过滤，替换当前几何近似。
  - 给 monitor 加 goal_id 级别跟踪，避免并发取消/重发时状态歧义。
  - 给 frontier 加失败记忆和冷却衰减，而不只是简单 blacklist。
  - 如果你愿意，我下一步可以继续替你做一版“和现有 Gazebo 多机 launch 对齐”的集成
    适配


请在现有 ROS1 多机器人中央协调器工程中，实现“探索任务结束条件（exploration stopping condition）”模块，并将其接入 pose-registration 版协调器主循环。

# 系统背景
当前系统具有以下结构：
- ROS1
- TurtleBot3 Gazebo 仿真
- 每台机器人独立运行 slam_toolbox
- 每台机器人独立运行 move_base
- 每台机器人具有独立坐标链：
  robot_i/map -> robot_i/odom -> robot_i/base_link
- 中央协调器基于 pose-registration 进行多机器人 frontier 协同探索
- frontier 先在各机器人 local map 上提取，再投影到 global_map 融合
- 协调器不依赖 multirobot_map_merge 作为主控制输入

# 目标
实现一个“探索结束条件判定模块”，用于在协调器运行过程中判断：
- 当前探索是否已经完成
- 是否应该停止继续分配探索任务
- 是否应发布 exploration_finished 状态

# 设计原则
请不要使用“瞬时没有 frontier 就立即结束”这种过于脆弱的逻辑。
请实现一个“组合式结束条件 + 连续确认窗口”的机制。

本项目采用如下停止逻辑：

## 主结束条件
当以下 3 个条件同时成立时，认为系统进入“可结束状态”：

1. valid_global_frontier_count == 0
   这里的 valid_global_frontier 指的是：
   - 已经完成 global 融合和去重
   - 已经过滤无效/过小/已占用/明显不可达 frontier
   - 至少对系统来说是“有价值且可执行”的 frontier

2. all_robots_idle_or_terminal == true
   即所有机器人当前都没有 active exploration goal：
   - 没有处于 ACTIVE / PENDING 的 move_base 探索目标
   - 已完成、失败或空闲都可以视为非执行态
   - 但如果失败后还能重新分配目标，则不能算最终 idle

3. map_growth_below_threshold == true
   即最近一个时间窗口内，全局探索进展已经基本停滞。
   工程上用“地图已知区域增长量”来近似判断探索是否仍有明显推进。

## 连续确认机制
即使上述 3 个条件成立，也不要立刻停止。
需要连续满足 stop_confirm_cycles 个调度周期，才真正宣布 exploration finished。

这样做是为了避免：
- 地图更新延迟
- frontier 瞬时丢失
- TF 抖动
- 某台机器人刚完成目标但系统还没来得及生成新目标
- 局部 planner 短时失败

# 你需要实现的功能

## 1. 新建 stopping_condition.py
请实现一个独立模块：
src/multi_robot_coordinator/stopping_condition.py

建议包含以下类：

class ExplorationStopState:
    # 保存最近一次/最近窗口的停止判据统计量

class StoppingConditionEvaluator:
    # 核心判定器

## 2. 需要维护和计算的量
请实现对以下指标的维护：

- valid_global_frontier_count: int
- active_goal_count: int
- idle_robot_count: int
- terminal_robot_count: int
- known_cell_count_current: int
- known_cell_count_history: deque
- known_cell_growth_in_window: int
- map_growth_ratio_in_window: float
- stop_counter: int
- finished: bool
- finish_reason: str

## 3. 地图增长判据
请实现一个函数，用于根据 recent global exploration map 或融合后的 exploration coverage 统计最近时间窗口内的地图增长量。

建议定义：
- known cell = OccupancyGrid 中不等于 unknown 的 cell
- known_cell_count_current = 当前已知 cell 数
- known_cell_growth_in_window = 当前值 - 窗口起始值
- map_growth_ratio_in_window = growth / max(previous_known_cells, 1)

支持以下两种阈值判定，任一达到即可认为 map growth 很小：
- 绝对增长量 < known_cells_growth_abs_threshold
- 相对增长率 < known_cells_growth_ratio_threshold

## 4. 机器人执行状态判定
请实现一个函数，判断系统中是否还有机器人在执行探索任务。

要求：
- 如果 robot_state.goal_status in ["ACTIVE", "PENDING"]，则视为 active
- 如果机器人 goal_status 为 ["IDLE", "SUCCEEDED", "ABORTED", "FAILED", "CANCELED"]，则视为非 active
- 但如果存在待重分配逻辑且机器人仍可继续参与探索，需要由协调器上层决定是否仍算 terminal
- 第一版可以简化为：
  只要没有 ACTIVE/PENDING goal，就算 non-active

## 5. frontier 结束判据
请实现函数：
has_no_valid_frontiers(frontiers) -> bool

要求：
- 输入是已经过 global 融合和过滤后的 FrontierGlobal 列表
- 如果列表为空，则返回 true
- 如果列表中 frontier 全部被 assigned 且没有新的可分配 frontier，也可以视为 no valid frontiers
- 第一版先按“过滤后的可分配 frontier 数 == 0”实现

## 6. 连续确认逻辑
请实现：
update_stop_state(...) -> ExplorationStopState

逻辑如下：
- 若本周期满足：
  no_valid_frontiers and all_non_active and map_growth_small
  则 stop_counter += 1
- 否则 stop_counter = 0
- 当 stop_counter >= stop_confirm_cycles 时：
  finished = True
  finish_reason = "frontier_exhausted_and_map_stagnant"

## 7. 强制停止条件
请额外支持工程保护性的停止条件：

- 超过 max_exploration_runtime_sec
- 连续 reassign 失败次数超过 max_failed_reassign_rounds
- 所有机器人长期失活（可选）
- 手动 stop flag（可选）

如果触发这些条件，也允许 finished = True，并设置不同的 finish_reason，例如：
- "timeout"
- "too_many_failed_reassignments"
- "manual_stop"

## 8. 配置参数
请在 coordinator.yaml 中加入以下配置项，并在代码中读取：

stopping_condition:
  enabled: true
  stop_confirm_cycles: 8
  map_growth_window_sec: 15.0
  known_cells_growth_abs_threshold: 80
  known_cells_growth_ratio_threshold: 0.005
  max_exploration_runtime_sec: 1800.0
  max_failed_reassign_rounds: 10
  publish_finished_topic: "/exploration_finished"

请允许通过 rosparam 覆盖这些值。

## 9. 接入 coordinator_node.py
请把该模块接入中央协调器主循环。

要求在每轮调度时：
1. 获取当前 valid global frontier 列表
2. 获取当前 robot states
3. 获取当前全局探索地图或可用于统计 known cells 的地图
4. 调用 stopping condition evaluator 更新状态
5. 如果 finished == True：
   - 停止继续分配新 frontier
   - 可选：取消所有机器人当前探索目标
   - 发布 exploration finished 状态
   - 输出日志说明 finish_reason

## 10. 发布状态
请发布以下内容：
- std_msgs/Bool: /exploration_finished
- std_msgs/String: /exploration_finish_reason
- 可选调试 topic：
  - /exploration_stop_debug

其中 debug 信息至少包括：
- valid_global_frontier_count
- active_goal_count
- known_cell_growth_in_window
- stop_counter
- finished

# 推荐接口设计

请实现类似下面的接口：

class StoppingConditionEvaluator:
    def __init__(self, config):
        ...

    def update(
        self,
        filtered_global_frontiers,
        robot_states,
        global_map_msg,
        current_time_sec,
        failed_reassign_rounds=0,
        manual_stop=False,
    ) -> ExplorationStopState:
        ...

    def is_finished(self) -> bool:
        ...

    def get_finish_reason(self) -> str:
        ...

    def reset(self):
        ...

# 工程要求
- 使用 Python
- 使用 dataclass/type hints/docstring
- 尽量不要依赖额外第三方库
- 对空地图、时间窗口未填满、frontier 列表为空、某些机器人无状态等情况做好健壮处理
- 不要只写伪代码，请写接近可运行的真实 ROS1 代码
- 所有阈值集中配置化，不要硬编码
- 保持与现有模块低耦合

# 输出要求
请按以下顺序输出：
1. stopping_condition.py 的完整代码
2. coordinator_node.py 需要修改的部分
3. coordinator.yaml 需要新增的配置
4. 发布 finished 状态的话题说明
5. 一个简短的测试方案
6. 已知限制

如果你发现当前工程里缺少某些辅助函数，请一并补充，并明确指出新增了哪些接口。