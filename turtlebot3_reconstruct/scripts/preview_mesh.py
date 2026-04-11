#!/usr/bin/env python3
"""Quick preview utility for mesh PLY files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import open3d as o3d


def build_parser() -> argparse.ArgumentParser:
    default_path = (
        Path(__file__).resolve().parents[1]
        / "output"
        / "global_model_textured_mesh.ply"
    )
    parser = argparse.ArgumentParser(description="Preview a mesh PLY file with Open3D.")
    parser.add_argument(
        "mesh",
        nargs="?",
        type=Path,
        default=default_path,
        help="Mesh PLY file to preview.",
    )
    parser.add_argument(
        "--wireframe",
        action="store_true",
        help="Show triangle wireframe on top of the mesh.",
    )
    parser.add_argument(
        "--back-face",
        action="store_true",
        help="Render back faces to inspect thin surfaces.",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    mesh_path = args.mesh.expanduser().resolve()
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file does not exist: {mesh_path}")

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise RuntimeError(f"Mesh is empty: {mesh_path}")

    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Mesh Preview: {mesh_path.name}")
    vis.add_geometry(mesh)
    render_option = vis.get_render_option()
    render_option.mesh_show_wireframe = bool(args.wireframe)
    render_option.mesh_show_back_face = bool(args.back_face)
    vis.run()
    vis.destroy_window()
    return 0


if __name__ == "__main__":
    sys.exit(main())
