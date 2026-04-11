#!/usr/bin/env python3

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from recon.registration_local import main


if __name__ == "__main__":
    sys.exit(main())
