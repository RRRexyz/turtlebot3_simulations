#!/usr/bin/env python3
"""Entry point for global point-cloud fusion and mesh reconstruction."""

from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from recon.global_fusion import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
