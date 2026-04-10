#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BAG_DIR="${SCRIPT_DIR}/../bag"
mkdir -p "${BAG_DIR}"

rosbag record -O "${BAG_DIR}/3.bag" \
    /tb3_3/camera/rgb/image_raw \
    /tb3_3/camera/rgb/camera_info \
    /tb3_3/camera/depth/image_raw \
    /tb3_3/camera/depth/camera_info \
    /tb3_3/odom
