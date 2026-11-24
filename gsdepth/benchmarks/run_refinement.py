"""Run, score and export the fixed-policy multi-view depth correction experiment."""

import argparse
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np
from PIL import Image

from gsdepth.core.depth_refinement import Rules, ViewEvidence, refine, refine_weak_layers
from gsdepth.core.depth_reprojection import refine_with_reprojection
from gsdepth.benchmarks.experiment import write_ply
from gsdepth.benchmarks.public_benchmark import CALIBRATION, VIEWS, calibrate, camera_at, check_depth_definition, read_image, read_truth
from gsdepth.core.renderer import load_gaussian_ply, render


HELD_OUT = (12, 34, 55, 75, 99, 120, 143, 177)
SOURCES = {
    "chair": "https://dx.gl/datasets/KLBJAuie9JaB",
    "sign": "https://dx.gl/datasets/tHdRul1GzzoU",
    "drill": "https://dx.gl/datasets/A0dcsk7HHgAg",
}
ERROR_THRESHOLDS = (0.025, 0.05, 0.10)
WEAK_LAYER_MASS = 0.05
WEAK_MINIMUM_VIEWS = 3


def neighbor_indices(directions, index, held_out, count=4):
    # Choose by camera position direction, with view ID as the stable tie-break.
    order = np.argsort(-(directions @ directions[index]), kind="stable")
    unavailable = set(held_out) | set(CALIBRATION)
    return [int(j) for j in order if j not in unavailable][:count]


def load_luminance(archive, metadata, index, size):
    image = read_image(archive, metadata["frames"][index]["file_path"], size, Image.Resampling.BILINEAR)
    return (image[..., :3].astype(float) / 255) @ np.array([0.2126, 0.7152, 0.0722])


def evaluate(depths, frame, target, choices, truth, scale, index, rules, reprojected=None):
    valid = np.isfinite(truth) & (truth > 0) & np.isfinite(depths["median"])
    if not valid.any():
        raise ValueError(f"View {index} has no valid overlapping depth pixels")
    errors = {name: np.abs(depth[valid] / scale - truth[valid]) for name, depth in depths.items()}
    if any(not np.isfinite(err).all() for err in errors.values()):
        raise ValueError(f"View {index} has inconsistent coverage")

    candidates = np.stack((frame["front"], frame["back"]))
    eligible = target[None] & (abs(candidates - frame["median"]) >= rules.minimum_depth_change)
    candidate_errors = np.abs(candidates[:, valid] / scale - truth[valid])
    oracle = np.minimum(errors["median"], np.where(eligible[:, valid], candidate_errors, np.inf).min(axis=0))
    if reprojected is not None:
        candidates = reprojected[:, valid]
        eligible = np.isfinite(candidates) & (abs(candidates - frame["median"][valid]) >= rules.minimum_depth_change)
        candidate_errors = np.abs(candidates / scale - truth[valid])
        extended_oracle = np.minimum(oracle, np.where(eligible, candidate_errors, np.inf).min(axis=0))

    rows = []
    for threshold in ERROR_THRESHOLDS:
        bad_before = errors["median"] > threshold
        recoverable = bad_before & (oracle <= threshold)
        for name in depths:
            err = errors[name]
            bad = err > threshold
            edited = choices[name][valid] > 0 if name in choices else None
            fixed = int((edited & bad_before & ~bad).sum()) if edited is not None else None
            broken = int((edited & ~bad_before & bad).sum()) if edited is not None else None
            if edited is not None and int(bad.sum()) != int(bad_before.sum()) - fixed + broken:
                raise AssertionError("Error transitions must account for every difference from the median")
            rows.append({
                "view": index, "threshold": threshold, "method": name, "valid": int(valid.sum()),
                "eligible_targets": int((target & valid).sum()), "mae": float(err.mean()),
                "bad": int(bad.sum()), "bad_fraction": float(bad.mean()),
                "edits": int(edited.sum()) if edited is not None else None,
                "fixed": fixed, "broken": broken,
                "net_fixed": fixed - broken if fixed is not None else None,
                "median_bad": int(bad_before.sum()), "recoverable": int(recoverable.sum()),
                "oracle_bad": int((oracle > threshold).sum()),
            })
            if reprojected is not None:
                rows[-1]["reproject_recoverable"] = int((bad_before & (extended_oracle <= threshold)).sum())
    return rows, valid


def export_example(directory, index, neighbors, anchor, depths, choices, target, trace, truth, valid, scale):
    np.savez_compressed(
        directory / "example_trace.npz", view=index, neighbor_views=neighbors,
        rgb=anchor.frame["rgb"], truth=truth, valid=valid, target=target, model_scale=scale,
        threshold=0.05, front=anchor.frame["front"]/scale, back=anchor.frame["back"]/scale,
        **{f"{name}_depth": depth/scale for name, depth in depths.items()},
        **{f"{name}_choice": value for name, value in choices.items()},
        **trace,
    )
    write_ply(directory / "median.ply", anchor.camera, depths["median"], anchor.frame["rgb"], valid)
    write_ply(directory / "joint.ply", anchor.camera, depths["joint"], anchor.frame["rgb"], valid)
    if "geometry_weak_guard" in depths:
        write_ply(directory / "geometry_weak_guard.ply", anchor.camera, depths["geometry_weak_guard"], anchor.frame["rgb"], valid)
    if "reproject_consensus" in depths:
        write_ply(directory / "reproject_consensus.ply", anchor.camera, depths["reproject_consensus"], anchor.frame["rgb"], valid)


def summarize(rows):
    summary = {}
    for threshold in ERROR_THRESHOLDS:
        portion = [row for row in rows if row["threshold"] == threshold]
        median = [row for row in portion if row["method"] == "median"]
        threshold_summary = {
            "valid": sum(row["valid"] for row in median),
            "eligible_targets": sum(row["eligible_targets"] for row in median),
            "median_bad": sum(row["median_bad"] for row in median),
            "recoverable": sum(row["recoverable"] for row in median),
            "oracle_bad": sum(row["oracle_bad"] for row in median),
        }
        if "reproject_recoverable" in median[0]:
            threshold_summary["reproject_recoverable"] = sum(row["reproject_recoverable"] for row in median)
        for method in dict.fromkeys(row["method"] for row in portion):
            scores = [row for row in portion if row["method"] == method]
            result = {
                "macro_mae": float(np.mean([row["mae"] for row in scores])),
                "macro_bad_fraction": float(np.mean([row["bad_fraction"] for row in scores])),
                "bad": sum(row["bad"] for row in scores),
            }
            if scores[0]["edits"] is not None:
                result.update({key: sum(row[key] for row in scores) for key in ("edits", "fixed", "broken", "net_fixed")})
            threshold_summary[method] = result
        summary[str(threshold)] = threshold_summary
    return summary


def run(name, split, data, output, size=128, reproject=False, minimum_layer_mass=0.10, weak_layer_check=False,
        ray_depth=False, on_view=None):
    if name == "sign" and split != "heldout":
        raise ValueError("The sign dataset is reserved for held-out evaluation")
    root = data / name
    views = VIEWS if split == "development" else HELD_OUT
    rules = Rules()
    gaussians = load_gaussian_ply(root / "scene.ply")
    output.mkdir(parents=True, exist_ok=True)
    with ZipFile(root / "reference.zip") as archive:
        metadata = json.loads(archive.read("transforms.json"))
        intrinsics = np.array([
            [metadata["fl_x"], 0, metadata["cx"]], [0, metadata["fl_y"], metadata["cy"]], [0, 0, 1],
        ])
        intrinsics = np.diag([size/metadata["w"], size/metadata["h"], 1]) @ intrinsics
        depth_checks = check_depth_definition(archive, metadata, intrinsics, size)
        normalization, poses, scale, alignment = calibrate(archive, metadata, gaussians, intrinsics, size)
        direction = poses[:, :3, 3] - poses[:, :3, 3].mean(axis=0)
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        cache = {}
        rows, neighbor_log = [], {}

        def get_view(index):
            if index not in cache:
                camera = camera_at(poses[index], intrinsics, size)
                frame = render(gaussians, camera, rules.minimum_layer_gap, minimum_layer_mass=minimum_layer_mass,
                               weak_layer_mass=WEAK_LAYER_MASS if weak_layer_check and index in views else None,
                               ray_depth=ray_depth)
                luminance = load_luminance(archive, metadata, index, size)
                cache[index] = ViewEvidence(camera, frame, luminance)
            return cache[index]

        example_view = None
        for index in views:
            start = perf_counter()
            neighbors = neighbor_indices(direction, index, views)
            neighbor_log[index] = neighbors
            anchor = get_view(index)
            auxiliary = [get_view(j) for j in neighbors]
            depths, target, choices, trace = refine(anchor, auxiliary, rules)
            proposals = None
            if weak_layer_check:
                additions = refine_weak_layers(anchor, auxiliary, depths["geometry"], rules, WEAK_MINIMUM_VIEWS)
                depths.update(additions)
                choices.update({method: np.isfinite(depth) & (depth != depths["median"])
                                for method, depth in additions.items()})
            if reproject:
                additions, proposals = refine_with_reprojection(anchor, auxiliary, depths["geometry"], rules)
                depths.update(additions)
                choices.update({method: np.isfinite(depth) & (depth != depths["median"])
                                for method, depth in additions.items()})
            # The full selection is finished before the evaluation depth is loaded.
            truth = read_truth(archive, metadata, index, size)
            view_rows, valid = evaluate(depths, anchor.frame, target, choices, truth, scale, index, rules, proposals)
            rows.extend(view_rows)
            if on_view is not None:
                on_view(index, depths, truth, valid, scale)
            mode = "geometry_weak_guard" if weak_layer_check else "reproject_consensus" if reproject else "joint"
            scored = next(row for row in view_rows if row["method"] == mode and row["threshold"] == 0.05)
            if example_view is None and (scored["edits"] > 0 or index == views[-1]):
                export_example(output, index, neighbors, anchor, depths, choices, target, trace, truth, valid, scale)
                example_view = index
            print(f"{name} {split} view {index:03} {mode}: {scored['edits']} edits, {scored['fixed']} fixed, "
                  f"{scored['broken']} broken; {perf_counter()-start:.2f} s", flush=True)

    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    fingerprints = {}
    for filename in ("scene.ply", "reference.zip"):
        with (root / filename).open("rb") as stream:
            fingerprints[filename] = hashlib.file_digest(stream, "sha256").hexdigest()
    result = {
        "dataset": name, "source": SOURCES[name], "split": split, "size": size, "views": views,
        "example_view": example_view,
        "neighbors": neighbor_log, "rules": asdict(rules), "gaussians": len(gaussians.centers),
        "normalization": normalization, "model_scale": scale,
        "calibration_ious": alignment, "depth_convention_checks": depth_checks,
        "source_sha256": fingerprints, "by_error_threshold": summarize(rows),
    }
    if reproject:
        result["reprojection"] = {
            "reproject": "Keep original geometric edits; check new depths with other views.",
            "reproject_photo": "Also require at least one verifying view with improved patch match.",
            "reproject_consensus": "Also allow small median-depth updates from at least three agreeing views.",
        }
    if minimum_layer_mass != 0.10:
        result["minimum_layer_mass"] = minimum_layer_mass
    if weak_layer_check:
        result["weak_layer_check"] = {
            "base_mass": minimum_layer_mass, "relaxed_mass": WEAK_LAYER_MASS, "minimum_views": WEAK_MINIMUM_VIEWS,
            "mode": "Keep baseline edits; consider only new weak depths, with at least three supporting views.",
        }
    if ray_depth:
        result["fragment_depth"] = "per-pixel ray Gaussian maximum, sorted per ray"
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", choices=SOURCES, required=True)
    parser.add_argument("--split", choices=("development", "heldout"), default="heldout")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--reproject", action="store_true", help="Evaluate additional depths from neighboring views")
    parser.add_argument("--layer-mass", type=float, default=0.10, help="Minimum compositing weight on either side of the split")
    parser.add_argument("--weak-layer-check", action="store_true", help="Compare uniform and three-view guarded weak splits")
    parser.add_argument("--ray-depth", action="store_true", help="Sort fragments by their Gaussian maximum along each ray")
    args = parser.parse_args()
    if args.size < 3:
        parser.error("--size must be at least 3")
    if not 0 < args.layer_mass < 0.5:
        parser.error("--layer-mass must be between 0 and 0.5")
    if args.weak_layer_check and args.layer_mass <= WEAK_LAYER_MASS:
        parser.error("--weak-layer-check requires a baseline mass greater than 0.05")
    suffix = "_reproject" if args.reproject else ""
    if args.weak_layer_check:
        suffix += "_weak_guard"
    if args.ray_depth:
        suffix += "_ray_depth"
    if args.layer_mass != 0.10:
        suffix += f"_mass{args.layer_mass:g}"
    output = args.output or Path("results/refinement") / f"{args.object}_{args.split}{suffix}"
    run(args.object, args.split, args.data, output, args.size,
        reproject=args.reproject, minimum_layer_mass=args.layer_mass, weak_layer_check=args.weak_layer_check,
        ray_depth=args.ray_depth)


if __name__ == "__main__":
    main()
