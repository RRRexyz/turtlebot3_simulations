#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BAG_DIR="${SCRIPT_DIR}/../bag"
mkdir -p "${BAG_DIR}"

rosbag record -O "${BAG_DIR}/common.bag" \
    /tf \
    /tf_static \
    /gazebo/model_states
