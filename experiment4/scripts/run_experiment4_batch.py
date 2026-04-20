#!/usr/bin/env python3
"""Run experiment 4 offline reconstruction comparison groups."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import open3d as o3d
import rospy
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SIM_ROOT = PACKAGE_ROOT.parent
RECONSTRUCT_ROOT = SIM_ROOT / "turtlebot3_reconstruct"

if str(RECONSTRUCT_ROOT) not in sys.path:
    sys.path.insert(0, str(RECONSTRUCT_ROOT))

from recon import GlobalFusionConfig, PairwiseRegistrationConfig, build_global_model, run_registration  # noqa: E402


GROUP_TO_CONFIG = {
    "group_a_no_gt_seed": "group_a_no_gt_seed.yaml",
    "group_b_gt_seed_transform_only": "group_b_gt_seed_transform_only.yaml",
    "group_c_gt_seed_direct_icp": "group_c_gt_seed_direct_icp.yaml",
    "group_d_gt_seed_overlap_icp": "group_d_gt_seed_overlap_icp.yaml",
}

DISABLED_OVERLAP_MIN_POINTS = 10**9


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run experiment 4 comparison groups for offline 3D reconstruction.",
    )
    parser.add_argument(
        "--group",
        default="all",
        choices=["all", *GROUP_TO_CONFIG.keys()],
        help="Run one group or all groups sequentially.",
    )
    parser.add_argument(
        "--config-root",
        type=Path,
        default=PACKAGE_ROOT / "config",
        help="Directory containing experiment4 group yaml files.",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=PACKAGE_ROOT / "results",
        help="Directory for experiment outputs.",
    )
    parser.add_argument(
        "--reconstruct-output",
        type=Path,
        default=PACKAGE_ROOT / "output",
        help="Directory containing tb3_1.db / tb3_2.db / tb3_3.db and exported clouds.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PACKAGE_ROOT / "output" / "local_exports_manifest.json",
        help="Manifest produced by export_local_models.py.",
    )
    parser.add_argument(
        "--common-bag",
        type=Path,
        default=PACKAGE_ROOT / "bag" / "common.bag",
        help="Bag containing /gazebo/model_states for GT seeding.",
    )
    parser.add_argument(
        "--force-export",
        action="store_true",
        help="Force re-export local clouds and poses before running the experiment.",
    )
    return parser


def load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid YAML object in {path}")
    return payload


def timestamp_string() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def required_robot_names(config_payload: Dict[str, Any]) -> List[str]:
    target_robot = str(config_payload["target_robot"])
    source_robots = [str(name) for name in config_payload.get("source_robots", [])]
    ordered = [target_robot, *source_robots]
    return list(dict.fromkeys(ordered))


def validate_required_databases(reconstruct_output: Path, robot_names: Sequence[str]) -> List[Path]:
    missing_paths: List[Path] = []
    for robot_name in robot_names:
        db_path = reconstruct_output / f"{robot_name}.db"
        if not db_path.exists():
            missing_paths.append(db_path)
    return missing_paths


def ensure_manifest(
    manifest_path: Path,
    reconstruct_output: Path,
    required_robots: Sequence[str],
    force_export: bool,
) -> Path:
    manifest_path = manifest_path.expanduser().resolve()
    reconstruct_output = reconstruct_output.expanduser().resolve()

    missing_dbs = validate_required_databases(reconstruct_output, required_robots)
    if missing_dbs:
        missing_robot_names = [path.stem for path in missing_dbs]
        reconstruct_hints = "\n".join(
            f"  ./reconstruct_{robot_name}.sh" for robot_name in missing_robot_names
        )
        raise RuntimeError(
            "Missing RTAB-Map database files required by this experiment group:\n"
            + "\n".join(f"  - {path}" for path in missing_dbs)
            + "\nPlease generate them first, for example:\n"
            + reconstruct_hints
        )

    if manifest_path.exists() and not force_export:
        try:
            manifest_payload = load_manifest(manifest_path)
            exported_robot_names = {
                str(robot.get("robot_name"))
                for robot in manifest_payload.get("robots", [])
            }
            if all(robot_name in exported_robot_names for robot_name in required_robots):
                print(f"[experiment4] Reusing existing manifest: {manifest_path}")
                return manifest_path
            print(
                f"[experiment4] Existing manifest is missing required robots "
                f"{required_robots}, re-exporting local models."
            )
        except Exception:
            print(f"[experiment4] Existing manifest is invalid, re-exporting local models.")

    command = [
        "python3",
        str(RECONSTRUCT_ROOT / "scripts" / "export_local_models.py"),
        "--db-dir",
        str(reconstruct_output),
        "--output-dir",
        str(reconstruct_output),
        "--robots",
        *required_robots,
    ]
    print(f"[experiment4] Exporting local models: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Failed to export local models for robots {list(required_robots)} from {reconstruct_output}"
        ) from exc

    if not manifest_path.exists():
        raise FileNotFoundError(f"Expected manifest was not generated: {manifest_path}")
    return manifest_path


def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def find_robot_manifest_entry(manifest: Dict[str, Any], robot_name: str) -> Dict[str, Any]:
    for robot in manifest.get("robots", []):
        if robot.get("robot_name") == robot_name:
            return robot
    raise KeyError(f"Robot {robot_name!r} not found in manifest {manifest}")


def build_registration_config(config_payload: Dict[str, Any]) -> PairwiseRegistrationConfig:
    registration = dict(config_payload["registration"])
    min_overlap_points = int(registration["min_overlap_points"])
    if registration.get("disable_overlap_crop", False):
        min_overlap_points = DISABLED_OVERLAP_MIN_POINTS

    return PairwiseRegistrationConfig(
        voxel_size=float(registration["voxel_size"]),
        max_correspondence_distance=float(registration["max_correspondence_distance"]),
        fitness_threshold=float(registration["fitness_threshold"]),
        pose_match_radius=float(registration["pose_match_radius"]),
        min_pose_correspondences=int(registration["min_pose_correspondences"]),
        overlap_crop_distance=float(registration["overlap_crop_distance"]),
        min_overlap_points=min_overlap_points,
        enable_icp_refinement=not bool(registration.get("disable_icp_refinement", False)),
    )


def build_global_fusion_config(config_payload: Dict[str, Any]) -> GlobalFusionConfig:
    fusion = config_payload["global_fusion"]
    return GlobalFusionConfig(
        voxel_size=float(fusion["voxel_size"]),
        outlier_nb_neighbors=int(fusion["outlier_nb_neighbors"]),
        outlier_std_ratio=float(fusion["outlier_std_ratio"]),
        normal_radius=float(fusion["normal_radius"]),
        poisson_depth=int(fusion["poisson_depth"]),
        mesh_density_quantile=float(fusion["mesh_density_quantile"]),
        target_triangle_count=int(fusion["target_triangle_count"]),
    )


def summarize_registration_metrics(registration_payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not registration_payload:
        return {
            "pair_count": 0,
            "valid_pair_count": 0,
            "invalid_pair_count": 0,
            "valid_pair_ratio": 0.0,
            "mean_fitness": None,
            "min_fitness": None,
            "max_fitness": None,
            "mean_inlier_rmse": None,
            "max_inlier_rmse": None,
            "mean_fitness_valid_only": None,
            "mean_inlier_rmse_valid_only": None,
            "gt_error_pair_count": 0,
            "mean_gt_translation_error_m": None,
            "mean_gt_rotation_error_deg": None,
            "mean_gt_translation_error_valid_only_m": None,
            "mean_gt_rotation_error_valid_only_deg": None,
            "gt_seed_pair_count": 0,
            "overlap_crop_pair_count": 0,
        }

    pairs = registration_payload.get("sources", [])
    if not pairs:
        return {
            "pair_count": 0,
            "valid_pair_count": 0,
            "invalid_pair_count": 0,
            "valid_pair_ratio": 0.0,
            "mean_fitness": None,
            "min_fitness": None,
            "max_fitness": None,
            "mean_inlier_rmse": None,
            "max_inlier_rmse": None,
            "mean_fitness_valid_only": None,
            "mean_inlier_rmse_valid_only": None,
            "gt_error_pair_count": 0,
            "mean_gt_translation_error_m": None,
            "mean_gt_rotation_error_deg": None,
            "mean_gt_translation_error_valid_only_m": None,
            "mean_gt_rotation_error_valid_only_deg": None,
            "gt_seed_pair_count": 0,
            "overlap_crop_pair_count": 0,
        }

    valid_pairs = [
        item
        for item in pairs
        if bool(item.get("valid_pair", False))
        and item.get("fitness") is not None
        and item.get("inlier_rmse") is not None
    ]
    fitness_values = [float(item["fitness"]) for item in valid_pairs]
    rmse_values = [float(item["inlier_rmse"]) for item in valid_pairs]
    gt_error_pairs = [
        item
        for item in pairs
        if bool(item.get("gt_relative_transform_available", False))
        and item.get("gt_relative_translation_error_m") is not None
        and item.get("gt_relative_rotation_error_deg") is not None
    ]
    gt_error_valid_pairs = [
        item for item in gt_error_pairs
        if bool(item.get("valid_pair", False))
    ]
    gt_translation_values = [float(item["gt_relative_translation_error_m"]) for item in gt_error_pairs]
    gt_rotation_values = [float(item["gt_relative_rotation_error_deg"]) for item in gt_error_pairs]
    gt_translation_valid_values = [float(item["gt_relative_translation_error_m"]) for item in gt_error_valid_pairs]
    gt_rotation_valid_values = [float(item["gt_relative_rotation_error_deg"]) for item in gt_error_valid_pairs]
    gt_seed_pair_count = sum(int(bool(item.get("gt_seed_available", 0.0))) for item in pairs)
    overlap_crop_pair_count = sum(int(bool(item.get("overlap_crop_used", False))) for item in pairs)
    valid_pair_count = len(valid_pairs)
    invalid_pair_count = len(pairs) - valid_pair_count
    return {
        "pair_count": len(pairs),
        "valid_pair_count": valid_pair_count,
        "invalid_pair_count": invalid_pair_count,
        "valid_pair_ratio": (
            float(valid_pair_count) / float(len(pairs))
            if len(pairs) > 0
            else 0.0
        ),
        "mean_fitness": float(np.mean(fitness_values)) if fitness_values else None,
        "min_fitness": float(np.min(fitness_values)) if fitness_values else None,
        "max_fitness": float(np.max(fitness_values)) if fitness_values else None,
        "mean_inlier_rmse": float(np.mean(rmse_values)) if rmse_values else None,
        "max_inlier_rmse": float(np.max(rmse_values)) if rmse_values else None,
        "mean_fitness_valid_only": float(np.mean(fitness_values)) if fitness_values else None,
        "mean_inlier_rmse_valid_only": float(np.mean(rmse_values)) if rmse_values else None,
        "gt_error_pair_count": len(gt_error_pairs),
        "mean_gt_translation_error_m": float(np.mean(gt_translation_values)) if gt_translation_values else None,
        "mean_gt_rotation_error_deg": float(np.mean(gt_rotation_values)) if gt_rotation_values else None,
        "mean_gt_translation_error_valid_only_m": (
            float(np.mean(gt_translation_valid_values)) if gt_translation_valid_values else None
        ),
        "mean_gt_rotation_error_valid_only_deg": (
            float(np.mean(gt_rotation_valid_values)) if gt_rotation_valid_values else None
        ),
        "gt_seed_pair_count": int(gt_seed_pair_count),
        "overlap_crop_pair_count": int(overlap_crop_pair_count),
    }


def analyze_global_outputs(global_summary: Dict[str, Any]) -> Dict[str, Any]:
    filtered_cloud_path = Path(global_summary["filtered_cloud"])
    textured_mesh_path = Path(global_summary["textured_mesh"])

    filtered_cloud = o3d.io.read_point_cloud(str(filtered_cloud_path))
    mesh = o3d.io.read_triangle_mesh(str(textured_mesh_path))

    bbox = filtered_cloud.get_axis_aligned_bounding_box()
    extent = np.asarray(bbox.get_extent(), dtype=np.float64)
    bbox_volume = float(np.prod(extent))

    cloud_points = np.asarray(filtered_cloud.points)
    cloud_colors = np.asarray(filtered_cloud.colors) if filtered_cloud.has_colors() else np.empty((0, 3))

    mesh_component_count = 0
    largest_component_triangle_ratio = 0.0
    if not mesh.is_empty() and len(mesh.triangles) > 0:
        _, cluster_triangle_counts, _ = mesh.cluster_connected_triangles()
        cluster_triangle_counts = np.asarray(cluster_triangle_counts, dtype=np.int64)
        mesh_component_count = int(cluster_triangle_counts.size)
        if cluster_triangle_counts.size > 0 and int(np.sum(cluster_triangle_counts)) > 0:
            largest_component_triangle_ratio = float(
                float(np.max(cluster_triangle_counts)) / float(np.sum(cluster_triangle_counts))
            )

    color_mean = None
    color_std = None
    if cloud_colors.size > 0:
        color_mean = [float(value) for value in np.mean(cloud_colors, axis=0)]
        color_std = [float(value) for value in np.std(cloud_colors, axis=0)]

    stats = global_summary["stats"]
    return {
        "filtered_bbox_extent_xyz": [float(value) for value in extent],
        "filtered_bbox_volume": bbox_volume,
        "filtered_point_count": int(stats["filtered_point_count"]),
        "raw_point_count": int(stats["raw_point_count"]),
        "outlier_removed_count": int(stats["outlier_removed_count"]),
        "outlier_removed_ratio": (
            float(stats["outlier_removed_count"]) / float(stats["downsampled_point_count"])
            if int(stats["downsampled_point_count"]) > 0
            else 0.0
        ),
        "mesh_vertex_count": int(stats["mesh_vertex_count"]),
        "mesh_triangle_count": int(stats["mesh_triangle_count"]),
        "mesh_component_count": mesh_component_count,
        "mesh_largest_component_triangle_ratio": largest_component_triangle_ratio,
        "cloud_has_color": bool(filtered_cloud.has_colors()),
        "cloud_color_mean_rgb": color_mean,
        "cloud_color_std_rgb": color_std,
        "cloud_point_count_check": int(cloud_points.shape[0]),
    }


def write_pairwise_metrics_csv(run_dir: Path, registration_payload: Optional[Dict[str, Any]]) -> Path:
    output_path = run_dir / "pairwise_metrics.csv"
    fieldnames = [
        "source_robot",
        "target_robot",
        "coarse_method",
        "valid_pair",
        "fitness",
        "inlier_rmse",
        "gt_relative_transform_available",
        "gt_relative_translation_error_m",
        "gt_relative_rotation_error_deg",
        "optimization_fitness",
        "optimization_inlier_rmse",
        "refinement_used",
        "fit_threshold_ok",
        "gt_seed_available",
        "overlap_crop_used",
        "overlap_source_points",
        "overlap_target_points",
        "evaluation_overlap_source_points",
        "evaluation_overlap_target_points",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if registration_payload:
            for item in registration_payload.get("sources", []):
                writer.writerow({
                    "source_robot": item.get("source_robot"),
                    "target_robot": item.get("target_robot"),
                    "coarse_method": item.get("coarse_method"),
                    "valid_pair": item.get("valid_pair"),
                    "fitness": item.get("fitness"),
                    "inlier_rmse": item.get("inlier_rmse"),
                    "gt_relative_transform_available": item.get("gt_relative_transform_available"),
                    "gt_relative_translation_error_m": item.get("gt_relative_translation_error_m"),
                    "gt_relative_rotation_error_deg": item.get("gt_relative_rotation_error_deg"),
                    "optimization_fitness": item.get("optimization_fitness"),
                    "optimization_inlier_rmse": item.get("optimization_inlier_rmse"),
                    "refinement_used": item.get("refinement_used"),
                    "fit_threshold_ok": item.get("fit_threshold_ok"),
                    "gt_seed_available": item.get("gt_seed_available"),
                    "overlap_crop_used": item.get("overlap_crop_used"),
                    "overlap_source_points": item.get("overlap_source_points"),
                    "overlap_target_points": item.get("overlap_target_points"),
                    "evaluation_overlap_source_points": item.get("evaluation_overlap_source_points"),
                    "evaluation_overlap_target_points": item.get("evaluation_overlap_target_points"),
                })
    return output_path


def write_global_metrics_csv(run_dir: Path, global_metrics: Dict[str, Any]) -> Path:
    output_path = run_dir / "global_metrics.csv"
    fieldnames = [
        "raw_point_count",
        "filtered_point_count",
        "outlier_removed_count",
        "outlier_removed_ratio",
        "bbox_extent_x",
        "bbox_extent_y",
        "bbox_extent_z",
        "bbox_volume",
        "mesh_vertex_count",
        "mesh_triangle_count",
        "mesh_component_count",
        "mesh_largest_component_triangle_ratio",
        "cloud_has_color",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "raw_point_count": global_metrics["raw_point_count"],
            "filtered_point_count": global_metrics["filtered_point_count"],
            "outlier_removed_count": global_metrics["outlier_removed_count"],
            "outlier_removed_ratio": global_metrics["outlier_removed_ratio"],
            "bbox_extent_x": global_metrics["filtered_bbox_extent_xyz"][0],
            "bbox_extent_y": global_metrics["filtered_bbox_extent_xyz"][1],
            "bbox_extent_z": global_metrics["filtered_bbox_extent_xyz"][2],
            "bbox_volume": global_metrics["filtered_bbox_volume"],
            "mesh_vertex_count": global_metrics["mesh_vertex_count"],
            "mesh_triangle_count": global_metrics["mesh_triangle_count"],
            "mesh_component_count": global_metrics["mesh_component_count"],
            "mesh_largest_component_triangle_ratio": global_metrics["mesh_largest_component_triangle_ratio"],
            "cloud_has_color": global_metrics["cloud_has_color"],
        })
    return output_path


def flatten_summary_for_batch(summary: Dict[str, Any]) -> Dict[str, Any]:
    registration = summary["registration_metrics"]
    global_metrics = summary["global_metrics"]
    return {
        "group_name": summary["group_name"],
        "display_name": summary["display_name"],
        "mean_fitness": registration["mean_fitness"],
        "min_fitness": registration["min_fitness"],
        "mean_inlier_rmse": registration["mean_inlier_rmse"],
        "valid_pair_count": registration["valid_pair_count"],
        "invalid_pair_count": registration["invalid_pair_count"],
        "valid_pair_ratio": registration["valid_pair_ratio"],
        "mean_gt_translation_error_m": registration["mean_gt_translation_error_m"],
        "mean_gt_rotation_error_deg": registration["mean_gt_rotation_error_deg"],
        "mean_gt_translation_error_valid_only_m": registration["mean_gt_translation_error_valid_only_m"],
        "mean_gt_rotation_error_valid_only_deg": registration["mean_gt_rotation_error_valid_only_deg"],
        "gt_seed_pair_count": registration["gt_seed_pair_count"],
        "overlap_crop_pair_count": registration["overlap_crop_pair_count"],
        "raw_point_count": global_metrics["raw_point_count"],
        "filtered_point_count": global_metrics["filtered_point_count"],
        "bbox_extent_x": global_metrics["filtered_bbox_extent_xyz"][0],
        "bbox_extent_y": global_metrics["filtered_bbox_extent_xyz"][1],
        "bbox_extent_z": global_metrics["filtered_bbox_extent_xyz"][2],
        "bbox_volume": global_metrics["filtered_bbox_volume"],
        "outlier_removed_ratio": global_metrics["outlier_removed_ratio"],
        "mesh_vertex_count": global_metrics["mesh_vertex_count"],
        "mesh_triangle_count": global_metrics["mesh_triangle_count"],
        "mesh_component_count": global_metrics["mesh_component_count"],
        "mesh_largest_component_triangle_ratio": global_metrics["mesh_largest_component_triangle_ratio"],
    }


def write_batch_summary(result_root: Path, batch_id: str, summaries: Sequence[Dict[str, Any]]) -> None:
    if not summaries:
        return

    batch_json = result_root / f"batch_summary_{batch_id}.json"
    batch_csv = result_root / f"batch_summary_{batch_id}.csv"
    batch_payload = {
        "batch_id": batch_id,
        "generated_at": datetime.now().isoformat(),
        "groups": summaries,
    }
    batch_json.write_text(json.dumps(batch_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    rows = [flatten_summary_for_batch(summary) for summary in summaries]
    with batch_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_group(
    group_name: str,
    config_root: Path,
    result_root: Path,
    manifest_path: Path,
    reconstruct_output: Path,
    common_bag_path: Path,
    force_export: bool,
) -> Dict[str, Any]:
    config_path = config_root / GROUP_TO_CONFIG[group_name]
    config_payload = load_yaml(config_path)
    run_dir = (result_root / group_name / timestamp_string()).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[experiment4] Starting {group_name} -> {run_dir}")

    target_robot = str(config_payload["target_robot"])
    source_robots = [str(name) for name in config_payload.get("source_robots", [])]
    required_robots = required_robot_names(config_payload)
    resolved_manifest = ensure_manifest(manifest_path, reconstruct_output, required_robots, force_export)
    manifest_payload = load_manifest(resolved_manifest)
    target_entry = find_robot_manifest_entry(manifest_payload, target_robot)

    registration_enabled = bool(config_payload["registration"]["enabled"])
    registration_summary_path: Optional[Path] = None
    registration_payload: Optional[Dict[str, Any]] = None
    input_clouds: List[Path] = [Path(target_entry["cloud"]).expanduser().resolve()]

    if registration_enabled:
        registration_config = build_registration_config(config_payload)
        print(
            f"[experiment4] Running registration for {group_name}: "
            f"target={target_robot}, sources={','.join(source_robots)}, "
            f"gt_seed={'on' if bool(config_payload['registration']['use_gt_seed']) else 'off'}, "
            f"overlap_crop={'off' if bool(config_payload['registration'].get('disable_overlap_crop', False)) else 'on'}, "
            f"icp_refinement={'off' if bool(config_payload['registration'].get('disable_icp_refinement', False)) else 'on'}"
        )
        registration_summary_path = run_registration(
            manifest_path=resolved_manifest,
            target_robot=target_robot,
            source_robots=source_robots,
            output_dir=run_dir,
            config=registration_config,
            common_bag_path=common_bag_path.expanduser().resolve()
            if common_bag_path.exists()
            else None,
            gt_match_max_dt=float(config_payload["registration"]["gt_match_max_dt"]),
            use_ground_truth_seed=bool(config_payload["registration"]["use_gt_seed"]),
        )
        registration_payload = json.loads(registration_summary_path.read_text(encoding="utf-8"))
        valid_pairs = [
            item for item in registration_payload.get("sources", [])
            if bool(item.get("valid_pair", False))
        ]
        input_clouds.extend(
            Path(item["transformed_cloud"]).expanduser().resolve()
            for item in valid_pairs
        )

    print(
        f"[experiment4] Building global model for {group_name} from {len(input_clouds)} cloud(s)."
    )
    global_summary = build_global_model(
        input_cloud_paths=input_clouds,
        output_dir=run_dir,
        output_prefix="global_model",
        config=build_global_fusion_config(config_payload),
    )
    global_metrics = analyze_global_outputs(global_summary)
    registration_metrics = summarize_registration_metrics(registration_payload)

    pairwise_csv_path = write_pairwise_metrics_csv(run_dir, registration_payload)
    global_csv_path = write_global_metrics_csv(run_dir, global_metrics)

    summary = {
        "group_name": group_name,
        "display_name": config_payload["display_name"],
        "description": config_payload["description"],
        "run_dir": str(run_dir),
        "target_robot": target_robot,
        "source_robots": list(config_payload.get("source_robots", [])),
        "registration_enabled": registration_enabled,
        "gt_seed_enabled": bool(config_payload["registration"]["use_gt_seed"]),
        "overlap_crop_enabled": not bool(config_payload["registration"].get("disable_overlap_crop", False)),
        "icp_refinement_enabled": not bool(config_payload["registration"].get("disable_icp_refinement", False)),
        "manifest_path": str(resolved_manifest),
        "registration_summary_path": str(registration_summary_path) if registration_summary_path else None,
        "global_summary_path": str(global_summary["summary_path"]),
        "pairwise_metrics_csv": str(pairwise_csv_path),
        "global_metrics_csv": str(global_csv_path),
        "input_clouds": [str(path) for path in input_clouds],
        "registration_config": config_payload["registration"],
        "global_fusion_config": config_payload["global_fusion"],
        "registration_metrics": registration_metrics,
        "global_metrics": global_metrics,
        "included_source_robots": (
            [item.get("source_robot") for item in registration_payload.get("sources", []) if bool(item.get("valid_pair", False))]
            if registration_payload
            else []
        ),
        "excluded_source_robots": (
            [item.get("source_robot") for item in registration_payload.get("sources", []) if not bool(item.get("valid_pair", False))]
            if registration_payload
            else []
        ),
        "artifacts": {
            "filtered_cloud": global_summary["filtered_cloud"],
            "raw_cloud": global_summary["raw_cloud"],
            "mesh": global_summary["mesh"],
            "textured_mesh": global_summary["textured_mesh"],
        },
    }

    if registration_payload:
        summary["pairwise_results"] = registration_payload.get("sources", [])

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[experiment4] Completed {group_name}: {summary_path}")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    ros_argv = rospy.myargv(argv=list(argv) if argv is not None else sys.argv)
    args = parser.parse_args(ros_argv[1:])

    config_root = args.config_root.expanduser().resolve()
    result_root = args.result_root.expanduser().resolve()
    reconstruct_output = args.reconstruct_output.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    common_bag_path = args.common_bag.expanduser().resolve()

    result_root.mkdir(parents=True, exist_ok=True)

    groups = list(GROUP_TO_CONFIG.keys()) if args.group == "all" else [args.group]
    batch_id = timestamp_string()
    summaries: List[Dict[str, Any]] = []

    try:
        for group_name in groups:
            summary = run_group(
                group_name=group_name,
                config_root=config_root,
                result_root=result_root,
                manifest_path=manifest_path,
                reconstruct_output=reconstruct_output,
                common_bag_path=common_bag_path,
                force_export=args.force_export,
            )
            summaries.append(summary)
    except Exception as exc:
        print(f"[experiment4] ERROR: {exc}", file=sys.stderr)
        return 1

    write_batch_summary(result_root, batch_id, summaries)
    print(f"[experiment4] Completed groups: {', '.join(groups)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
