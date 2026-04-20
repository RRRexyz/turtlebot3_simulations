#!/usr/bin/env python3
"""experiment4 wrapper for global point-cloud fusion with package-local defaults."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RECONSTRUCT_SCRIPT = PACKAGE_ROOT.parent / "turtlebot3_reconstruct" / "scripts" / "build_global_model.py"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "output"


def main() -> int:
    command = [
        sys.executable,
        str(RECONSTRUCT_SCRIPT),
        "--input-clouds",
        str(DEFAULT_OUTPUT_DIR / "tb3_1_cloud.ply"),
        str(DEFAULT_OUTPUT_DIR / "tb3_2_to_tb3_1_cloud.ply"),
        str(DEFAULT_OUTPUT_DIR / "tb3_3_to_tb3_1_cloud.ply"),
        "--output-dir",
        str(DEFAULT_OUTPUT_DIR),
        *sys.argv[1:],
    ]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
