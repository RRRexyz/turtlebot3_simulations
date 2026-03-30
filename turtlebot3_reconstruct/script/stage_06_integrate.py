from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recon.config import load_config
from recon.io_utils import ProjectPaths, save_json, save_mesh, save_pcd, setup_logger
from recon.pair_selector import load_all_keyframes
from recon.tsdf_fusion import integrate_global_map, load_optimized_pose_map


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for final TSDF integration."""

    parser = argparse.ArgumentParser(description="Fuse optimized RGB-D keyframes into a global TSDF map.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "default.yaml",
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    config = load_config(args.config)
    paths = ProjectPaths(config)
    logger = setup_logger("stage_06_integrate", paths.log_file("stage_06_integrate"))
    stage_dir = paths.stage_dir("stage_06_integrate")

    optimized_pose_map = load_optimized_pose_map(paths.stage_file("stage_05_optimize", "optimized_poses.json"))
    keyframes = load_all_keyframes(config)
    mesh, pointcloud, stats = integrate_global_map(config, keyframes, optimized_pose_map)

    final_mesh_path = save_mesh(stage_dir / "final_mesh.ply", mesh)
    final_map_path = save_pcd(stage_dir / "final_map.ply", pointcloud)
    save_json(stage_dir / "integration_stats.json", asdict(stats))

    logger.info(
        "Integrated global map | keyframes=%d vertices=%d triangles=%d points=%d",
        stats.integrated_keyframe_count,
        stats.mesh_vertex_count,
        stats.mesh_triangle_count,
        stats.point_count,
    )
    print("final_mesh:", final_mesh_path)
    print("final_map:", final_map_path)


if __name__ == "__main__":
    main()
