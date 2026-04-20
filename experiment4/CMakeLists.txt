cmake_minimum_required(VERSION 3.0.2)
project(experiment4)

find_package(catkin REQUIRED COMPONENTS
  rosbag
  rospy
  rtabmap_launch
  tf
  turtlebot3_central
  turtlebot3_reconstruct
)

catkin_package(
  CATKIN_DEPENDS
    rosbag
    rospy
    rtabmap_launch
    tf
    turtlebot3_central
    turtlebot3_reconstruct
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/build_global_model.py
  scripts/export_local_models.py
  scripts/preview_mesh.py
  scripts/preview_point_cloud.py
  scripts/preview_registration_clouds.py
  scripts/record_all.py
  scripts/register_local_models.py
  scripts/run_experiment4_batch.py
  scripts/visualize_results.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(PROGRAMS
  scripts/record_common.sh
  scripts/record_tb3_1.sh
  scripts/record_tb3_2.sh
  scripts/record_tb3_3.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config results
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
