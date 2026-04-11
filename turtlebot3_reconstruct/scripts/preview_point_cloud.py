#!/usr/bin/env python3
"""Quick preview utility for point-cloud PLY files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import open3d as o3d


def build_parser() -> argparse.ArgumentParser:
    default_path = (
        Path(__file__).resolve().parents[1]
        / "output"
        / "global_model_filtered_cloud.ply"
    )
    parser = argparse.ArgumentParser(description="Preview a point-cloud PLY file with Open3D.")
    parser.add_argument(
        "cloud",
        nargs="?",
        type=Path,
        default=default_path,
        help="Point-cloud PLY file to preview.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=2.0,
        help="Rendered point size.",
    )
    parser.add_argument(
        "--show-normals",
        action="store_true",
        help="Show estimated normals if present in the file.",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cloud_path = args.cloud.expanduser().resolve()
    if not cloud_path.exists():
        raise FileNotFoundError(f"Point cloud file does not exist: {cloud_path}")

    cloud = o3d.io.read_point_cloud(str(cloud_path))
    if cloud.is_empty():
        raise RuntimeError(f"Point cloud is empty: {cloud_path}")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Point Cloud Preview: {cloud_path.name}")
    vis.add_geometry(cloud)
    render_option = vis.get_render_option()
    render_option.point_size = float(args.point_size)
    render_option.mesh_show_back_face = True
    render_option.point_show_normal = bool(args.show_normals)
    vis.run()
    vis.destroy_window()
    return 0


if __name__ == "__main__":
    sys.exit(main())
