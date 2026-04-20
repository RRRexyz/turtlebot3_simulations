#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BAG_DIR="${SCRIPT_DIR}/../bag"
mkdir -p "${BAG_DIR}"

rosbag record -O "${BAG_DIR}/2.bag" \
    /tb3_2/camera/rgb/image_raw \
    /tb3_2/camera/rgb/camera_info \
    /tb3_2/camera/depth/image_raw \
    /tb3_2/camera/depth/camera_info \
    /tb3_2/odom
