#!/usr/bin/env python3
"""experiment4 wrapper for local registration with package-local defaults."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RECONSTRUCT_SCRIPT = PACKAGE_ROOT.parent / "turtlebot3_reconstruct" / "scripts" / "register_local_models.py"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "output"
DEFAULT_MANIFEST = DEFAULT_OUTPUT_DIR / "local_exports_manifest.json"
DEFAULT_COMMON_BAG = PACKAGE_ROOT / "bag" / "common.bag"


def main() -> int:
    command = [
        sys.executable,
        str(RECONSTRUCT_SCRIPT),
        "--manifest",
        str(DEFAULT_MANIFEST),
        "--output-dir",
        str(DEFAULT_OUTPUT_DIR),
        "--common-bag",
        str(DEFAULT_COMMON_BAG),
        *sys.argv[1:],
    ]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
