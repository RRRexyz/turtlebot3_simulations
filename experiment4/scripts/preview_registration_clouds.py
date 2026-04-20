#!/usr/bin/env python3
"""experiment4 wrapper for registration cloud preview with package-local defaults."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RECONSTRUCT_SCRIPT = PACKAGE_ROOT.parent / "turtlebot3_reconstruct" / "scripts" / "preview_registration_clouds.py"


def main() -> int:
    command = [
        sys.executable,
        str(RECONSTRUCT_SCRIPT),
        "--target-cloud",
        str(PACKAGE_ROOT / "output" / "tb3_1_cloud.ply"),
        "--source-clouds",
        str(PACKAGE_ROOT / "output" / "tb3_2_to_tb3_1_cloud.ply"),
        str(PACKAGE_ROOT / "output" / "tb3_3_to_tb3_1_cloud.ply"),
        *sys.argv[1:],
    ]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
