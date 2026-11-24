"""Evaluate paired depth-derived point clouds in the reference world coordinates."""

import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from gsdepth.benchmarks.experiment import write_xyzrgb
from gsdepth.benchmarks.public_benchmark import camera_at, read_image, read_truth
from gsdepth.benchmarks.run_refinement import ERROR_THRESHOLDS, SOURCES, run
from gsdepth.core.depth_refinement import project, world_from_rays
from gsdepth.core.renderer import read_binary_ply_vertices


METHOD = "geometry_mass05"
VISIBILITY_TOLERANCE = 0.01
DENSE_REFERENCE_SIZE = 256


def replay_depths(dataset, data, evidence, ray_depth):
    variant = "ray_depth" if ray_depth else "weak_layer_guard"
    recorded = evidence / "followup" / variant / f"{dataset}_heldout"
    summary = json.loads((recorded / "summary.json").read_text(encoding="utf-8"))
    predictions = {}

    def capture(index, depths, truth, valid, scale):
        np.testing.assert_allclose(scale, summary["model_scale"], rtol=0, atol=1e-12)
        predictions[index] = (depths[METHOD] / scale, truth, valid)

    with TemporaryDirectory(prefix="gsdepth-surface-") as scratch:
        output = Path(scratch)
        run(dataset, "heldout", data, output, size=summary["size"], weak_layer_check=True,
            ray_depth=ray_depth, on_view=capture)
        if (output / "metrics.csv").read_bytes() != (recorded / "metrics.csv").read_bytes():
            raise ValueError(f"The depth replay differs from the published run for {dataset}/{variant}")
        result = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        for key in ("source_sha256", "views", "model_scale", "normalization", "neighbors", "rules", "weak_layer_check"):
            if result[key] != summary[key]:
                raise ValueError(f"The replay differs for {dataset}/{variant}/{key}")
    return predictions, summary


def visible_reference(points, native_camera, native_depth, valid):
    """Identify reference points on the visible surface and inside the paired evaluation region."""
    pixels, depths = project(points, native_camera)
    inside = (np.isfinite(pixels).all(axis=1)
              & (pixels >= 0).all(axis=1)
              & (pixels[:, 0] < native_camera.width)
              & (pixels[:, 1] < native_camera.height))
    indices = np.flatnonzero(inside)
    uv = pixels[inside]
    x, y = np.floor(uv).astype(int).T
    x_small = np.floor(uv[:, 0] * valid.shape[1] / native_camera.width).astype(int)
    y_small = np.floor(uv[:, 1] * valid.shape[0] / native_camera.height).astype(int)
    agrees = np.abs(native_depth[y, x] - depths[inside]) <= VISIBILITY_TOLERANCE
    return indices[agrees & valid[y_small, x_small]]


def describe_accuracy(distances):
    return {
        "points": len(distances),
        "mean_distance": float(distances.mean()),
        "median_distance": float(np.median(distances)),
        "p95_distance": float(np.quantile(distances, .95)),
    }


def describe_surface(accuracy, completeness):
    row = describe_accuracy(accuracy)
    row["visible_reference_points"] = len(completeness)
    row["completeness_mean_distance"] = float(completeness.mean())
    for threshold in ERROR_THRESHOLDS:
        precision = float(np.mean(accuracy <= threshold))
        recall = float(np.mean(completeness <= threshold))
        row[f"precision_{threshold:g}"] = precision
        row[f"recall_{threshold:g}"] = recall
        row[f"f1_{threshold:g}"] = 2 * precision * recall / (precision + recall) if precision + recall else 0
        row[f"outliers_{threshold:g}"] = int((accuracy > threshold).sum())
    return row


def save_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def resampled_reference(archive, metadata, view, pose, native_intrinsics, valid):
    """Sample a denser visible reference in the same per-view evaluation region."""
    size = DENSE_REFERENCE_SIZE
    scale = size / metadata["w"]
    intrinsics = np.diag([scale, scale, 1]) @ native_intrinsics
    camera = camera_at(pose, intrinsics, size)
    depth = read_truth(archive, metadata, view, size)
    x = np.floor((np.arange(size) + .5) * valid.shape[1] / size).astype(int)
    y = np.floor((np.arange(size) + .5) * valid.shape[0] / size).astype(int)
    reference_pixels = np.isfinite(depth) & (depth > 0) & valid[np.ix_(y, x)]
    return world_from_rays(camera, camera.rays()[reference_pixels], depth[reference_pixels])


def validate(dataset, data, evidence, output):
    old, baseline = replay_depths(dataset, data, evidence, ray_depth=False)
    new, updated = replay_depths(dataset, data, evidence, ray_depth=True)
    if old.keys() != new.keys() or baseline["source_sha256"] != updated["source_sha256"]:
        raise ValueError("Point clouds require the same source and the same target views")

    size = baseline["size"]
    output.mkdir(parents=True, exist_ok=True)
    clouds = {"center": [], "ray": [], "reference_depth_control": []}
    colors = []
    view_rows = []
    visibility_by_view = {}
    dense_reference = []
    reference_pixels = 0
    with ZipFile(data / dataset / "reference.zip") as archive:
        metadata = json.loads(archive.read("transforms.json"))
        if metadata["w"] != metadata["h"]:
            raise ValueError("The archived DX.GL evaluation expects square reference images")
        original_size = metadata["w"]
        native_intrinsics = np.array([[metadata["fl_x"], 0, metadata["cx"]],
                                      [0, metadata["fl_y"], metadata["cy"]], [0, 0, 1]])
        small_intrinsics = np.diag([size / original_size, size / original_size, 1]) @ native_intrinsics
        with archive.open(metadata["ply_file_path"]) as stream:
            source = read_binary_ply_vertices(stream)
        reference = np.column_stack([source[field] for field in ("x", "y", "z")]).astype(float)
        reference_tree = cKDTree(reference)
        visible = np.zeros(len(reference), dtype=bool)
        for view in baseline["views"]:
            before, truth, valid = old[view]
            after, reference_depth, updated_valid = new[view]
            np.testing.assert_array_equal(valid, updated_valid)
            np.testing.assert_array_equal(truth, reference_depth)
            reference_pixels += int((np.isfinite(truth) & (truth > 0)).sum())
            pose = np.asarray(metadata["frames"][view]["transform_matrix"])
            camera = camera_at(pose, small_intrinsics, size)
            native_camera = camera_at(pose, native_intrinsics, original_size)
            native_truth = read_truth(archive, metadata, view, original_size)
            indices = visible_reference(reference, native_camera, native_truth, valid)
            visible[indices] = True
            visibility_by_view[view] = int(len(indices))
            dense_reference.append(resampled_reference(archive, metadata, view, pose, native_intrinsics, valid))
            y, x = np.nonzero(valid)
            pixels = np.column_stack((x + .5, y + .5))
            rgb = read_image(archive, metadata["frames"][view]["file_path"], size, Image.Resampling.BILINEAR)
            colors.append(rgb[valid, :3])
            for method, depth in zip(clouds, (before, after, truth)):
                points = world_from_rays(camera, camera.rays()[valid], depth[valid])
                if method == "reference_depth_control":
                    reprojected, repro_depths = project(points, camera)
                    np.testing.assert_allclose(reprojected, pixels, rtol=0, atol=1e-10)
                    np.testing.assert_allclose(repro_depths, depth[valid], rtol=0, atol=1e-10)
                clouds[method].append(points)
                distance = reference_tree.query(points)[0]
                view_rows.append({"view": view, "method": method, **describe_accuracy(distance),
                                  "outliers_0.05": int((distance > .05).sum())})

    if not visible.any():
        raise ValueError("No reference surface is visible in the evaluated views")
    rows = []
    dense_rows = []
    reference_visible = reference[visible]
    dense_points = np.concatenate(dense_reference)
    dense_tree = cKDTree(dense_points)
    color = np.concatenate(colors)
    for method, parts in clouds.items():
        points = np.concatenate(parts)
        distance = reference_tree.query(points)[0]
        predicted_tree = cKDTree(points)
        completeness = predicted_tree.query(reference_visible)[0]
        row = {"dataset": dataset, "method": method, **describe_surface(distance, completeness)}
        rows.append(row)
        dense_rows.append({"dataset": dataset, "method": method,
                           **describe_surface(dense_tree.query(points)[0], predicted_tree.query(dense_points)[0])})
        write_xyzrgb(output / f"{method}.ply", points, color)
    np.savez_compressed(output / "geometry.npz", reference_visible=reference_visible,
                        center=np.concatenate(clouds["center"]), ray=np.concatenate(clouds["ray"]))
    save_csv(output / "surface_metrics.csv", rows)
    save_csv(output / "surface_per_view.csv", view_rows)
    save_csv(output / "surface_dense_reference.csv", dense_rows)
    notes = {
        "dataset": dataset, "source": baseline["source"], "source_sha256": baseline["source_sha256"],
        "views": baseline["views"], "image_size": size, "original_image_size": original_size,
        "reference_foreground_pixels": reference_pixels, "paired_valid_pixels": rows[0]["points"],
        "reference_points": len(reference), "visible_reference_points": int(visible.sum()),
        "visible_reference_points_by_view": visibility_by_view,
        "visibility_tolerance_scene_units": VISIBILITY_TOLERANCE,
        "dense_reference_size": DENSE_REFERENCE_SIZE, "dense_reference_points": len(dense_points),
        "dense_reference": "Back-projected source reference depths from the same views at 256x256 within the shared region",
        "distance_thresholds_scene_units": ERROR_THRESHOLDS, "depth_policy": METHOD,
        "coordinate_system": "DX.GL dataset world, prediction camera Z divided by the recorded training scale",
        "accuracy_reference": "Full supplied source point cloud",
        "completeness_reference": "Supplied source points consistent with native reference depth and in the shared mask",
        "alignment": "Fixed source cameras, no fitting to the reference point cloud",
        "depth_replay": "Both viewwise CSVs match the archived depth experiments byte for byte",
    }
    (output / "surface_protocol.json").write_text(json.dumps(notes, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", choices=SOURCES, required=True)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--evidence", type=Path, default=Path("evidence"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    validate(args.object, args.data, args.evidence,
             args.output or Path("results/surface_validation") / args.object)


if __name__ == "__main__":
    main()
