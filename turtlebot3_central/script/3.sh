#!/bin/bash
rosbag record -O src/turtlebot3_simulations/turtlebot3_central/bag/3.bag \
    /tb3_3/camera/rgb/image_raw \
    /tb3_3/camera/rgb/camera_info \
    /tb3_3/camera/depth/image_raw \
    /tb3_3/camera/depth/camera_info \
    /tb3_3/imu \
    /tb3_3/odom \
    /tf \
    /tf_static