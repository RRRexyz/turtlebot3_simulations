#!/usr/bin/env python3
"""experiment4 wrapper for mesh preview with package-local defaults."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RECONSTRUCT_SCRIPT = PACKAGE_ROOT.parent / "turtlebot3_reconstruct" / "scripts" / "preview_mesh.py"
DEFAULT_MESH = PACKAGE_ROOT / "output" / "global_model_textured_mesh.ply"
# DEFAULT_MESH = "/home/adminwe/project/catkin_ws/src/turtlebot3_simulations/experiment4/results/group_d_gt_seed_overlap_icp/20260420_131519_834085/global_model_textured_mesh.ply"

def main() -> int:
    extra_args = sys.argv[1:]
    command = [sys.executable, str(RECONSTRUCT_SCRIPT)]
    if not extra_args:
        command.append(str(DEFAULT_MESH))
    command.extend(extra_args)
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
