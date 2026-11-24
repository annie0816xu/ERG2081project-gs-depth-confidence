"""Replay the two example views and export the source arrays for Figure 3."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

from gsdepth.benchmarks.public_benchmark import camera_at, normalized_poses, read_image, read_truth
from gsdepth.benchmarks.run_refinement import evaluate, load_luminance
from gsdepth.core.depth_refinement import Rules, ViewEvidence, refine
from gsdepth.core.renderer import load_gaussian_ply, render


def replay(dataset, view, data, evidence, metric="joint"):
    root = data / dataset
    logs = evidence / f"{dataset}_heldout"
    recorded = json.loads((logs / "summary.json").read_text(encoding="utf-8"))
    for filename, expected in recorded["source_sha256"].items():
        with (root / filename).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"Input hash differs from the recorded experiment: {root / filename}")
    gaussians = load_gaussian_ply(root / "scene.ply")
    size = recorded["size"]
    neighbors = recorded["neighbors"][str(view)]
    rules = Rules(**recorded["rules"])
    method = "geometry" if metric == "geometry_mass05" else metric
    mass = recorded["weak_layer_check"]["relaxed_mass"] if metric == "geometry_mass05" else 0.10
    ray_depth = "fragment_depth" in recorded
    with ZipFile(root / "reference.zip") as archive:
        metadata = json.loads(archive.read("transforms.json"))
        poses = np.array([item["transform_matrix"] for item in metadata["frames"]])
        scale = 1.0
        if recorded["normalization"] == "nerfstudio_default":
            poses, scale = normalized_poses(poses)
        np.testing.assert_allclose(scale, recorded["model_scale"], atol=1e-12, rtol=0)
        intrinsics = np.array([
            [metadata["fl_x"], 0, metadata["cx"]], [0, metadata["fl_y"], metadata["cy"]], [0, 0, 1],
        ])
        intrinsics = np.diag([size / metadata["w"], size / metadata["h"], 1]) @ intrinsics

        def build(index):
            camera = camera_at(poses[index], intrinsics, size)
            frame = render(gaussians, camera, rules.minimum_layer_gap,
                           minimum_layer_mass=mass, ray_depth=ray_depth)
            return ViewEvidence(camera, frame, load_luminance(archive, metadata, index, size))

        anchor = build(view)
        depths, candidates, decisions, trace = refine(anchor, [build(j) for j in neighbors], rules)
        truth = read_truth(archive, metadata, view, size)
        rgb = read_image(archive, metadata["frames"][view]["file_path"], size, Image.Resampling.BILINEAR)
    rows, valid = evaluate(depths, anchor.frame, candidates, decisions, truth, scale, view, rules)
    current = next(row for row in rows if row["threshold"] == 0.05 and row["method"] == method)
    with (logs / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        previous = next(
            row for row in csv.DictReader(stream)
            if row["view"] == str(view) and row["threshold"] == "0.05" and row["method"] == metric
        )
    for key in ("valid", "edits", "fixed", "broken", "bad"):
        if current[key] != int(previous[key]):
            raise ValueError(f"Replayed {dataset} view {view} differs from the recorded view for {key}")
    np.testing.assert_allclose(current["mae"], float(previous["mae"]), atol=1e-12, rtol=0)
    return {
        "rgb": rgb[..., :3], "truth": truth, "valid": valid, "edited": decisions[method] > 0,
        "median": depths["median"] / scale, method: depths[method] / scale,
    }, recorded["source"], neighbors, trace, decisions[method], rules


def write_case(output, name, dataset, view, arrays, pixel_xy, source, neighbors, trace, choice, rules):
    x, y = pixel_xy
    radius = 14
    left, top = x - radius, y - radius
    if min(left, top) < 0 or x + radius >= arrays["rgb"].shape[1] or y + radius >= arrays["rgb"].shape[0]:
        raise ValueError("The requested crop extends outside the source view")
    crop = (slice(top, y + radius + 1), slice(left, x + radius + 1))
    np.savez_compressed(output / f"{name}.npz", **{
        key: value[crop] for key, value in arrays.items()
    }, marker_xy=np.array([radius, radius]), full_rgb=arrays["rgb"])
    z = arrays["truth"][y, x]
    target, = np.flatnonzero(np.all(trace["coordinates_yx"] == (y, x), axis=1))
    index = int(choice[y, x])
    states = trace["geometry_states"][:, index, target]
    cost = trace["photo_costs"][:, index, target]
    median_cost = trace["photo_costs"][:, 0, target]
    gain = median_cost - cost
    photo = np.isfinite(gain) & (cost <= rules.maximum_photo_cost) & (gain >= rules.minimum_photo_gain)
    return {
        "dataset": dataset, "view": view, "neighbor_views": neighbors, "pixel_xy": [int(x), int(y)],
        "crop_xywh": [int(left), int(top), 2 * radius + 1, 2 * radius + 1], "source": source,
        "reference_depth": float(z), "median_depth": float(arrays["median"][y, x]),
        "joint_depth": float(arrays["joint"][y, x]),
        "median_abs_error": float(abs(arrays["median"][y, x] - z)),
        "joint_abs_error": float(abs(arrays["joint"][y, x] - z)),
        "selected_candidate": ["median", "front", "back"][index],
        "geometry_states": states.tolist(),
        "photo_pass": photo.tolist(),
        "joint_pass": ((states == 1) & photo).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--evidence", type=Path, default=Path("evidence"))
    parser.add_argument("--output", type=Path, default=Path("figures/source_data"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    chair_view = 12
    chair, *chair_replay = replay("chair", chair_view, args.data, args.evidence)
    before, after = np.abs(chair["median"] - chair["truth"]), np.abs(chair["joint"] - chair["truth"])
    fixed = chair["valid"] & chair["edited"] & (before > 0.05) & (after <= 0.05)
    if not fixed.any():
        raise ValueError("No repaired pixel in the recorded example")
    y, x = np.unravel_index(np.argmax(np.where(fixed, before - after, -np.inf)), fixed.shape)
    success = write_case(args.output, "chair_fixed", "chair", chair_view, chair, (x, y), *chair_replay)

    record_path = args.evidence / "sign_heldout" / "false_edit_view_120.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    sign_view = record["view"]
    sign, *sign_replay = replay("sign", sign_view, args.data, args.evidence)
    x, y = record["pixel_xy"]
    np.testing.assert_allclose(sign["truth"][y, x], record["ref"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(sign["joint"][y, x], record["median_front_back"][record["selected"]], atol=1e-12, rtol=0)
    failure = write_case(args.output, "sign_false_edit", "sign", sign_view, sign, (x, y), *sign_replay)
    overview = {
        "selection": "Largest absolute error reduction among fixed chair pixels in view 12; documented sign false edit in view 120.",
        "error_threshold_scene_units": 0.05,
        "chair_fixed": success,
        "sign_false_edit": failure,
    }
    (args.output / "case_provenance.json").write_text(json.dumps(overview, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(overview, indent=2))


if __name__ == "__main__":
    main()
