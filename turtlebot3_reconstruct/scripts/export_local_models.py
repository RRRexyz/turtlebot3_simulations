#!/usr/bin/env python3
"""Export local colored point clouds and poses from RTAB-Map databases."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_ROBOTS = ("tb3_1", "tb3_2", "tb3_3")


def resolve_package_root() -> Path:
    try:
        import rospkg  # type: ignore

        return Path(rospkg.RosPack().get_path("turtlebot3_reconstruct"))
    except Exception:
        return Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export local clouds and poses from RTAB-Map databases.",
    )
    parser.add_argument(
        "--db-dir",
        type=Path,
        default=resolve_package_root() / "output",
        help="Directory containing tb3_1.db / tb3_2.db / tb3_3.db.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=resolve_package_root() / "output",
        help="Directory where clouds, poses and manifest will be written.",
    )
    parser.add_argument(
        "--robots",
        nargs="+",
        default=list(DEFAULT_ROBOTS),
        help="Robot names to export, e.g. tb3_1 tb3_2 tb3_3.",
    )
    parser.add_argument(
        "--opt",
        type=int,
        default=2,
        help="RTAB-Map optimization mode passed to rtabmap-export (default: 2, use optimized poses in DB).",
    )
    parser.add_argument(
        "--decimation",
        type=int,
        default=4,
        help="Depth decimation before cloud generation.",
    )
    parser.add_argument(
        "--voxel",
        type=float,
        default=0.01,
        help="Voxel size for exported local cloud.",
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="Export PLY clouds in ASCII format.",
    )
    return parser


def ensure_rtabmap_export() -> str:
    executable = shutil.which("rtabmap-export")
    if not executable:
        raise RuntimeError("rtabmap-export was not found in PATH.")
    return executable


def export_one_robot(
    executable: str,
    robot_name: str,
    db_dir: Path,
    output_dir: Path,
    opt: int,
    decimation: int,
    voxel: float,
    ascii_mode: bool,
):
    db_path = db_dir / f"{robot_name}.db"
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    output_prefix = robot_name
    cmd = [
        executable,
        "--cloud",
        "--poses",
        "--poses_camera",
        "--output",
        output_prefix,
        "--output_dir",
        str(output_dir),
        "--opt",
        str(opt),
        "--decimation",
        str(decimation),
        "--voxel",
        str(voxel),
    ]
    if ascii_mode:
        cmd.append("--ascii")
    cmd.append(str(db_path))

    print(f"[export] {robot_name}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    cloud_path = output_dir / f"{output_prefix}_cloud.ply"
    robot_pose_path = output_dir / f"{output_prefix}_poses.txt"
    camera_pose_path = output_dir / f"{output_prefix}_camera_poses.txt"

    return {
        "robot_name": robot_name,
        "database": str(db_path),
        "cloud": str(cloud_path),
        "robot_poses": str(robot_pose_path),
        "camera_poses": str(camera_pose_path),
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    executable = ensure_rtabmap_export()
    db_dir = args.db_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "db_dir": str(db_dir),
        "output_dir": str(output_dir),
        "robots": [],
        "export_settings": {
            "opt": args.opt,
            "decimation": args.decimation,
            "voxel": args.voxel,
            "ascii": args.ascii,
        },
    }

    for robot_name in args.robots:
        result = export_one_robot(
            executable=executable,
            robot_name=robot_name,
            db_dir=db_dir,
            output_dir=output_dir,
            opt=args.opt,
            decimation=args.decimation,
            voxel=args.voxel,
            ascii_mode=args.ascii,
        )
        manifest["robots"].append(result)

    manifest_path = output_dir / "local_exports_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"[done] wrote manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
