"""Global point-cloud fusion and mesh reconstruction utilities."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class GlobalFusionConfig:
    voxel_size: float = 0.03
    outlier_nb_neighbors: int = 30
    outlier_std_ratio: float = 1.5
    normal_radius: float = 0.08
    poisson_depth: int = 9
    mesh_density_quantile: float = 0.04
    target_triangle_count: int = 250000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fuse registered local clouds into a global colored cloud and mesh.",
    )
    default_output_dir = Path(__file__).resolve().parents[1] / "output"
    parser.add_argument(
        "--input-clouds",
        nargs="+",
        type=Path,
        default=[
            default_output_dir / "tb3_1_cloud.ply",
            default_output_dir / "tb3_2_to_tb3_1_cloud.ply",
            default_output_dir / "tb3_3_to_tb3_1_cloud.ply",
        ],
        help="Input point clouds that already share the same global frame.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory for fused outputs.",
    )
    parser.add_argument(
        "--output-prefix",
        default="global_model",
        help="Prefix used when naming fused outputs.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.03,
        help="Voxel size used to downsample the merged cloud.",
    )
    parser.add_argument(
        "--outlier-nb-neighbors",
        type=int,
        default=30,
        help="Neighbor count for statistical outlier removal.",
    )
    parser.add_argument(
        "--outlier-std-ratio",
        type=float,
        default=1.5,
        help="Standard deviation multiplier for statistical outlier removal.",
    )
    parser.add_argument(
        "--normal-radius",
        type=float,
        default=0.08,
        help="Neighborhood radius used for normal estimation before meshing.",
    )
    parser.add_argument(
        "--poisson-depth",
        type=int,
        default=9,
        help="Poisson reconstruction depth.",
    )
    parser.add_argument(
        "--mesh-density-quantile",
        type=float,
        default=0.04,
        help="Drop the lowest-density mesh vertices under this quantile.",
    )
    parser.add_argument(
        "--target-triangle-count",
        type=int,
        default=250000,
        help="Simplify the mesh to at most this many triangles when possible.",
    )
    return parser


def load_point_cloud(path: Path) -> o3d.geometry.PointCloud:
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise RuntimeError(f"Input point cloud is empty: {path}")
    return cloud


def concatenate_clouds(clouds: Sequence[o3d.geometry.PointCloud]) -> o3d.geometry.PointCloud:
    merged = o3d.geometry.PointCloud()
    for cloud in clouds:
        merged += cloud
    if merged.is_empty():
        raise RuntimeError("Merged cloud became empty.")
    return merged


def estimate_normals(cloud: o3d.geometry.PointCloud, normal_radius: float) -> None:
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=max(normal_radius, 0.03), max_nn=30)
    )
    cloud.orient_normals_consistent_tangent_plane(20)


def transfer_colors_to_mesh(
    mesh: o3d.geometry.TriangleMesh,
    reference_cloud: o3d.geometry.PointCloud,
) -> o3d.geometry.TriangleMesh:
    if mesh.is_empty():
        raise RuntimeError("Cannot color an empty mesh.")

    vertices = np.asarray(mesh.vertices)
    reference_points = np.asarray(reference_cloud.points)
    if reference_points.size == 0:
        raise RuntimeError("Reference cloud is empty while transferring colors.")

    colored_mesh = o3d.geometry.TriangleMesh(mesh)
    if not reference_cloud.has_colors():
        colored_mesh.paint_uniform_color([0.7, 0.7, 0.7])
        return colored_mesh

    reference_colors = np.asarray(reference_cloud.colors)
    tree = cKDTree(reference_points)
    _, indices = tree.query(vertices, k=1)
    vertex_colors = reference_colors[indices]
    colored_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    return colored_mesh


def build_global_model(
    input_cloud_paths: Sequence[Path],
    output_dir: Path,
    output_prefix: str,
    config: GlobalFusionConfig,
) -> Dict[str, object]:
    resolved_input_paths = [path.expanduser().resolve() for path in input_cloud_paths]
    for path in resolved_input_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing input point cloud: {path}")

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    input_clouds = [load_point_cloud(path) for path in resolved_input_paths]
    merged_raw = concatenate_clouds(input_clouds)

    fused_down = merged_raw.voxel_down_sample(config.voxel_size)
    if fused_down.is_empty():
        raise RuntimeError("Voxel downsampling removed all points from the merged cloud.")

    filtered_cloud, inlier_indices = fused_down.remove_statistical_outlier(
        nb_neighbors=config.outlier_nb_neighbors,
        std_ratio=config.outlier_std_ratio,
    )
    if filtered_cloud.is_empty():
        raise RuntimeError("Outlier removal removed all points from the merged cloud.")

    estimate_normals(filtered_cloud, config.normal_radius)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        filtered_cloud,
        depth=config.poisson_depth,
    )
    densities = np.asarray(densities)
    density_threshold = float(np.quantile(densities, config.mesh_density_quantile))
    keep_mask = densities >= density_threshold
    mesh.remove_vertices_by_mask(~keep_mask)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    if config.target_triangle_count > 0 and len(mesh.triangles) > config.target_triangle_count:
        mesh = mesh.simplify_quadric_decimation(config.target_triangle_count)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

    mesh.compute_vertex_normals()
    colored_mesh = transfer_colors_to_mesh(mesh, filtered_cloud)
    colored_mesh.compute_vertex_normals()

    raw_cloud_path = output_dir / f"{output_prefix}_raw_cloud.ply"
    filtered_cloud_path = output_dir / f"{output_prefix}_filtered_cloud.ply"
    mesh_path = output_dir / f"{output_prefix}_mesh.ply"
    colored_mesh_path = output_dir / f"{output_prefix}_textured_mesh.ply"
    summary_path = output_dir / f"{output_prefix}_summary.json"

    o3d.io.write_point_cloud(str(raw_cloud_path), merged_raw)
    o3d.io.write_point_cloud(str(filtered_cloud_path), filtered_cloud)
    o3d.io.write_triangle_mesh(str(mesh_path), mesh, write_vertex_normals=True)
    o3d.io.write_triangle_mesh(str(colored_mesh_path), colored_mesh, write_vertex_normals=True)

    summary: Dict[str, object] = {
        "input_clouds": [str(path) for path in resolved_input_paths],
        "raw_cloud": str(raw_cloud_path),
        "filtered_cloud": str(filtered_cloud_path),
        "mesh": str(mesh_path),
        "textured_mesh": str(colored_mesh_path),
        "config": {
            "voxel_size": config.voxel_size,
            "outlier_nb_neighbors": config.outlier_nb_neighbors,
            "outlier_std_ratio": config.outlier_std_ratio,
            "normal_radius": config.normal_radius,
            "poisson_depth": config.poisson_depth,
            "mesh_density_quantile": config.mesh_density_quantile,
            "target_triangle_count": config.target_triangle_count,
        },
        "stats": {
            "raw_point_count": int(np.asarray(merged_raw.points).shape[0]),
            "downsampled_point_count": int(np.asarray(fused_down.points).shape[0]),
            "filtered_point_count": int(np.asarray(filtered_cloud.points).shape[0]),
            "outlier_removed_count": int(np.asarray(fused_down.points).shape[0] - len(inlier_indices)),
            "mesh_vertex_count": int(np.asarray(mesh.vertices).shape[0]),
            "mesh_triangle_count": int(np.asarray(mesh.triangles).shape[0]),
            "density_threshold": density_threshold,
        },
        "notes": {
            "texturing": "Vertex colors are transferred from the fused colored point cloud to the reconstructed mesh. This is vertex-color texturing, not UV atlas baking.",
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    summary["summary_path"] = str(summary_path)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = build_global_model(
        input_cloud_paths=args.input_clouds,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        config=GlobalFusionConfig(
            voxel_size=args.voxel_size,
            outlier_nb_neighbors=args.outlier_nb_neighbors,
            outlier_std_ratio=args.outlier_std_ratio,
            normal_radius=args.normal_radius,
            poisson_depth=args.poisson_depth,
            mesh_density_quantile=args.mesh_density_quantile,
            target_triangle_count=args.target_triangle_count,
        ),
    )
    print(f"[done] wrote global reconstruction summary: {summary['summary_path']}")
    return 0

