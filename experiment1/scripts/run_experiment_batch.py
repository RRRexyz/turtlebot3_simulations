#!/usr/bin/env python3
"""Run experiment 1 groups individually or sequentially."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


GROUP_TO_LAUNCH = {
    "group_a_single_robot": "group_a_single_robot.launch",
    "group_b_ordered_nearest": "group_b_ordered_nearest.launch",
    "group_c_distance_greedy": "group_c_distance_greedy.launch",
    "group_d_full_method": "group_d_full_method.launch",
}


def resolve_package_root() -> Path:
    try:
        import rospkg  # type: ignore

        return Path(rospkg.RosPack().get_path("experiment1"))
    except Exception:
        return Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run experiment 1 comparison groups.")
    parser.add_argument(
        "--group",
        default="all",
        choices=["all", *GROUP_TO_LAUNCH.keys()],
        help="Run one experiment group or all groups sequentially.",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=resolve_package_root() / "results",
        help="Directory used by metrics_recorders to save outputs.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Run Gazebo with GUI enabled.",
    )
    parser.add_argument(
        "--no-rviz",
        action="store_true",
        help="Disable map_merge.rviz during experiment runs.",
    )
    return parser


def run_group(group_name: str, result_root: Path, gui: bool, open_rviz: bool) -> None:
    launch_file = GROUP_TO_LAUNCH[group_name]
    run_name = time.strftime("%Y%m%d_%H%M%S")
    command = [
        "roslaunch",
        "experiment1",
        launch_file,
        f"result_root:={result_root}",
        f"run_name:={run_name}",
        f"gui:={'true' if gui else 'false'}",
        f"open_rviz:={'true' if open_rviz else 'false'}",
    ]
    print(f"[experiment1] Running {group_name}: {' '.join(command)}")
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{group_name} failed with exit code {completed.returncode}")


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result_root = args.result_root.expanduser().resolve()
    result_root.mkdir(parents=True, exist_ok=True)

    groups = list(GROUP_TO_LAUNCH.keys()) if args.group == "all" else [args.group]
    for index, group_name in enumerate(groups):
        run_group(group_name, result_root, args.gui, not args.no_rviz)
        if index != len(groups) - 1:
            time.sleep(3.0)

    print(f"[experiment1] Completed groups: {', '.join(groups)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
