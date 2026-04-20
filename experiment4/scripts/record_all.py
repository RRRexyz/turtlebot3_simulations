#!/usr/bin/env python3
"""Launch experiment4 bag-recording scripts for selected robots."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
COMMON_SCRIPT = SCRIPT_DIR / "record_common.sh"
ROBOT_TO_SCRIPT = {
    "tb3_1": SCRIPT_DIR / "record_tb3_1.sh",
    "tb3_2": SCRIPT_DIR / "record_tb3_2.sh",
    "tb3_3": SCRIPT_DIR / "record_tb3_3.sh",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record experiment4 topics for one or more robots.",
    )
    parser.add_argument(
        "--robots",
        nargs="+",
        choices=sorted(ROBOT_TO_SCRIPT.keys()),
        default=None,
        help="Robots to record. Defaults to all robots when omitted.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Explicitly record all robots.",
    )
    return parser


def resolve_scripts(args: argparse.Namespace) -> list[Path]:
    if args.all or not args.robots:
        selected_robots = list(ROBOT_TO_SCRIPT.keys())
    else:
        selected_robots = list(dict.fromkeys(args.robots))

    scripts = [COMMON_SCRIPT]
    scripts.extend(ROBOT_TO_SCRIPT[robot_name] for robot_name in selected_robots)
    return scripts


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    scripts_to_run = resolve_scripts(args)

    processes = []
    print("[experiment4] Starting recording scripts...")

    for script_path in scripts_to_run:
        print(f"[experiment4] Launching: {script_path}")
        process = subprocess.Popen(["bash", str(script_path)], preexec_fn=os.setsid)
        processes.append(process)

    try:
        for process in processes:
            process.wait()
    except KeyboardInterrupt:
        print("\n[experiment4] Ctrl+C received, stopping all recording processes...")
        for process in processes:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except Exception as exc:  # pragma: no cover - best effort cleanup
                print(f"[experiment4] Failed to stop process group {process.pid}: {exc}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
