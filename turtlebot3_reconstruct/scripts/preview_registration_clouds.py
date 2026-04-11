#!/usr/bin/env python3
"""Preview registered multi-robot point clouds in one Open3D window."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import open3d as o3d


def build_parser() -> argparse.ArgumentParser:
    output_dir = Path(__file__).resolve().parents[1] / "output"
    parser = argparse.ArgumentParser(
        description="Preview tb3_1 / tb3_2_to_tb3_1 / tb3_3_to_tb3_1 clouds together.",
    )
    parser.add_argument(
        "--target-cloud",
        type=Path,
        default=output_dir / "tb3_1_cloud.ply",
        help="Reference cloud in the shared frame.",
    )
    parser.add_argument(
        "--source-clouds",
        nargs="+",
        type=Path,
        default=[
            output_dir / "tb3_2_to_tb3_1_cloud.ply",
            output_dir / "tb3_3_to_tb3_1_cloud.ply",
        ],
        help="Registered source clouds in the target frame.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=2.5,
        help="Rendered point size.",
    )
    return parser


def load_cloud(path: Path) -> o3d.geometry.PointCloud:
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise RuntimeError(f"Point cloud is empty: {path}")
    return cloud


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    target_path = args.target_cloud.expanduser().resolve()
    source_paths = [path.expanduser().resolve() for path in args.source_clouds]

    for path in [target_path, *source_paths]:
        if not path.exists():
            raise FileNotFoundError(f"Point cloud file does not exist: {path}")

    colors = [
        [0.95, 0.35, 0.30],
        [0.20, 0.75, 0.35],
        [0.20, 0.45, 0.95],
        [0.95, 0.80, 0.20],
    ]

    labeled_paths = [("target", target_path)] + [
        (f"source_{index + 1}", path) for index, path in enumerate(source_paths)
    ]

    geometries: List[o3d.geometry.PointCloud] = []
    for index, (_, path) in enumerate(labeled_paths):
        cloud = load_cloud(path)
        cloud.paint_uniform_color(colors[index % len(colors)])
        geometries.append(cloud)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Registration Preview")
    for geometry in geometries:
        vis.add_geometry(geometry)

    render_option = vis.get_render_option()
    render_option.point_size = float(args.point_size)
    render_option.mesh_show_back_face = True

    print("Loaded point clouds:")
    for label, path in labeled_paths:
        print(f"  {label}: {path}")
    print("Color legend: target=red, source_1=green, source_2=blue")

    vis.run()
    vis.destroy_window()
    return 0


if __name__ == "__main__":
    sys.exit(main())
