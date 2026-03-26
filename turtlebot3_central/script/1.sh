#!/bin/bash
rosbag record -O src/turtlebot3_simulations/turtlebot3_central/bag/1.bag \
    /tb3_1/camera/rgb/image_raw \
    /tb3_1/camera/rgb/camera_info \
    /tb3_1/camera/depth/image_raw \
    /tb3_1/camera/depth/camera_info \
    /tb3_1/imu \
    /tb3_1/odom \
    /tf \
    /tf_static