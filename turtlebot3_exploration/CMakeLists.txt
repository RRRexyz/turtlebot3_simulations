cmake_minimum_required(VERSION 3.0.2)
project(turtlebot3_exploration)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  actionlib_msgs
  geometry_msgs
  move_base_msgs
  nav_msgs
  rospy
  std_msgs
  tf
  tf2_ros
  visualization_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS
    actionlib
    actionlib_msgs
    geometry_msgs
    move_base_msgs
    nav_msgs
    rospy
    std_msgs
    tf
    tf2_ros
    visualization_msgs
)

catkin_install_python(PROGRAMS
  scripts/coordinator_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
