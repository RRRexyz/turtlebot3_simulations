#!/bin/bash
rosbag record -O src/turtlebot3_simulations/turtlebot3_central/bag/2.bag \
    /tb3_2/camera/rgb/image_raw \
    /tb3_2/camera/rgb/camera_info \
    /tb3_2/camera/depth/image_raw \
    /tb3_2/camera/depth/camera_info \
    /tb3_2/imu \
    /tb3_2/odom \
    /tf \
    /tf_static