"""Evaluate training-free depth risk scores in a CPU-only 3DGS pipeline."""

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from gsdepth.core.renderer import Camera, Gaussians, composite, render


METHODS = {
    "opacity": "Accumulated opacity",
    "gradient": "Depth gradient",
    "std": "Weighted standard deviation",
    "spread": "Weighted 10–90% interval",
}
COVERAGES = np.array([50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 97, 98, 99, 100]) / 100


@dataclass(frozen=True)
class SyntheticCase:
    name: str
    near_z: float
    far_z: float
    half_width: float
    near_opacity: float
    angle: float


CASES = (
    SyntheticCase("canonical", 2.0, 5.0, 0.145, 0.45, 0),
    SyntheticCase("short_gap", 2.0, 2.7, 0.145, 0.50, 10),
    SyntheticCase("wide_gap", 1.8, 5.7, 0.150, 0.55, -12),
    SyntheticCase("thin_strip", 2.0, 5.0, 0.038, 0.55, 13),
    SyntheticCase("weak_foreground", 2.1, 5.1, 0.090, 0.12, -11),
    SyntheticCase("strong_foreground", 2.0, 4.7, 0.160, 0.75, 18),
)


def synthetic_camera(case, width, height):
    angle = np.radians(case.angle)
    position = np.array([1.3*np.sin(angle), 0.04*np.sin(angle), 0.0])
    forward = np.array([0.0, 0.0, 3.5]) - position
    forward /= np.linalg.norm(forward)
    right = np.cross([0, 1, 0], forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    camera_to_world = np.eye(4)
    camera_to_world[:3, :3] = np.column_stack((right, down, forward))
    camera_to_world[:3, 3] = position
    focal = 0.88*width
    intrinsics = np.array([[focal, 0, width/2], [0, focal, height/2], [0, 0, 1]])
    return Camera(camera_to_world, intrinsics, width, height)


def plane_gaussians(half_width, half_height, z, step, opacity, color):
    x = np.linspace(-half_width, half_width, max(2, round(2*half_width/step) + 1))
    y = np.linspace(-half_height, half_height, max(2, round(2*half_height/step) + 1))
    x, y = np.meshgrid(x, y)
    centers = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, z)))
    std = np.array([step*0.57, step*0.57, step*0.16])
    return Gaussians(
        centers, np.broadcast_to(np.diag(std**2), (x.size, 3, 3)),
        np.full(x.size, opacity), np.broadcast_to(color, (x.size, 3)),
    )


def synthetic_scene(case, camera):
    narrow = plane_gaussians(
        case.half_width, 0.64, case.near_z, 0.030, case.near_opacity, [0.12, 0.47, 0.50],
    )
    wall = plane_gaussians(4.3, 3.5, case.far_z, 0.080, 0.93, [0.81, 0.70, 0.53])
    splats = Gaussians(*[
        np.concatenate((getattr(narrow, field), getattr(wall, field)))
        for field in ("centers", "covariances", "opacities", "colors")
    ])

    # Rays use camera z as the parameter. These analytic intersections do not use splats.
    rotation = camera.camera_to_world[:3, :3]
    origin = camera.camera_to_world[:3, 3]
    rays_world = camera.rays() @ rotation.T
    near = (case.near_z - origin[2]) / rays_world[..., 2]
    intersection = origin + near[..., None] * rays_world
    hits_strip = (abs(intersection[..., 0]) <= case.half_width) & (abs(intersection[..., 1]) <= 0.64)
    far = (case.far_z - origin[2]) / rays_world[..., 2]
    return splats, np.where(hits_strip, near, far)


def adjacent_difference(depth, include_silhouette=False):
    """Use the largest absolute depth difference to a four-connected neighbor."""
    difference = np.zeros_like(depth)
    for first, second in (
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
    ):
        a, b = depth[first], depth[second]
        delta = np.where(np.isfinite(a) & np.isfinite(b), np.abs(a-b), 0)
        if include_silhouette:
            delta[np.isfinite(a) ^ np.isfinite(b)] = np.inf
        np.maximum(difference[first], delta, out=difference[first])
        np.maximum(difference[second], delta, out=difference[second])
    return difference


def score_frame(frame, truth, bad_threshold):
    depth = frame["depth"]
    foreground = np.isfinite(truth) & (truth > 0)
    valid = foreground & np.isfinite(depth)
    if not np.any(valid):
        raise ValueError("No overlap between rendered and reference depth")
    edge = adjacent_difference(truth, include_silhouette=True) > 2*bad_threshold
    for _ in range(2):
        padded = np.pad(edge, 1)
        edge = edge | padded[:-2, 1:-1] | padded[2:, 1:-1] | padded[1:-1, :-2] | padded[1:-1, 2:]
    error_map = np.abs(depth - truth)
    errors = error_map[valid]
    large_error = errors > bad_threshold
    correct_edge = edge[valid] & ~large_error
    score_maps = {
        "opacity": 1 - frame["opacity"],
        "gradient": adjacent_difference(depth),
        "std": frame["std"],
        "spread": frame["spread"],
    }
    tie = np.random.default_rng(20261124).permutation(errors.size)
    rows, retained = [], {}
    for method, score in score_maps.items():
        ordered = np.lexsort((tie, score[valid]))
        for coverage in COVERAGES:
            count = max(1, round(coverage * len(ordered)))
            selected = ordered[:count]
            rows.append({
                "method": method, "target_coverage": float(coverage),
                "kept": count, "valid": len(ordered), "reference_pixels": int(foreground.sum()),
                "mae": float(errors[selected].mean()),
                "bad_fraction": float(large_error[selected].mean()),
                "correct_edge_retention": (
                    float(correct_edge[selected].sum() / correct_edge.sum())
                    if correct_edge.any() else None
                ),
            })
            if coverage == 0.9:
                mask = np.zeros(truth.shape, dtype=bool)
                mask.flat[np.flatnonzero(valid)[selected]] = True
                retained[method] = mask
    return rows, retained, error_map, valid, edge


def write_ply(path, camera, depth, rgb, mask):
    xyz_camera = camera.rays()[mask] * depth[mask, None]
    xyz_world = xyz_camera @ camera.camera_to_world[:3, :3].T + camera.camera_to_world[:3, 3]
    colors = np.rint(255*rgb[mask]).astype(np.uint8)
    with path.open("w", encoding="ascii") as stream:
        stream.write(
            f"ply\nformat ascii 1.0\nelement vertex {len(xyz_world)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        )
        np.savetxt(stream, np.column_stack((xyz_world, colors)), fmt="%.7f %.7f %.7f %d %d %d")


def save_detail(output, frame, truth, camera, retained, error, valid, edge, threshold):
    np.savez_compressed(
        output / "example.npz", depth=frame["depth"], truth=truth, error=error,
        spread=frame["spread"], opacity=frame["opacity"], rgb=frame["rgb"],
        valid=valid, edge=edge, retained=retained["spread"], threshold=threshold,
    )
    write_ply(output / "before.ply", camera, frame["depth"], frame["rgb"], valid)
    write_ply(output / "after.ply", camera, frame["depth"], frame["rgb"], retained["spread"])


def run_synthetic(output, width, height):
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in CASES:
        start = perf_counter()
        camera = synthetic_camera(case, width, height)
        splats, truth = synthetic_scene(case, camera)
        frame = render(splats, camera)
        metrics, retained, error, valid, edge = score_frame(frame, truth, 0.10)
        rows.extend({"scene": case.name, **row} for row in metrics)
        if case.name == "canonical":
            save_detail(output, frame, truth, camera, retained, error, valid, edge, 0.10)
        print(f"{case.name}: {valid.sum()} pixels, {perf_counter() - start:.2f} s", flush=True)
    metadata = {
        "kind": "Synthetic occlusion benchmark",
        "units": "m", "bad_threshold": 0.10, "width": width, "height": height,
        "cases": [vars(case) for case in CASES],
    }
    save_results(output, rows, metadata)


def save_results(output, rows, metadata):
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for method in METHODS:
        subset = [row for row in rows if row["target_coverage"] == 0.9 and row["method"] == method]
        summary[method] = {
            key: float(np.mean([row[key] for row in subset if row[key] is not None]))
            for key in ("mae", "bad_fraction", "correct_edge_retention")
            if any(row[key] is not None for row in subset)
        }
    metadata["macro_average_at_90_percent"] = summary
    (output / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def self_check():
    # Both pixels accumulate 0.90 opacity; their contributor depths have different support.
    output = composite(
        np.array([0, 0, 1, 1]), np.array([2.0, 5.0, 4.98, 5.02]),
        np.tile([0.45, 0.45 / 0.55], 2), np.ones((4, 3)), (1, 2),
    )
    np.testing.assert_allclose(output["opacity"], [[0.9, 0.9]])
    np.testing.assert_allclose(output["depth"], [[3.5, 5.0]])
    np.testing.assert_allclose(output["spread"], [[3.0, 0.04]])
    np.testing.assert_allclose(output["std"], [[1.5, 0.02]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="benchmark", required=True)
    synthetic = subparsers.add_parser("synthetic", help="Run the analytic two-surface benchmark")
    synthetic.add_argument("--output", type=Path, default=Path("results/synthetic"))
    synthetic.add_argument("--width", type=int, default=160)
    synthetic.add_argument("--height", type=int, default=120)
    chair = subparsers.add_parser("chair", help="Run the pretrained DX.GL chair benchmark")
    chair.add_argument("--data", type=Path, default=Path("data/chair"))
    chair.add_argument("--output", type=Path, default=Path("results/chair"))
    chair.add_argument("--size", type=int, default=160)
    subparsers.add_parser("check", help="Check the reference alpha-blending example")
    args = parser.parse_args()
    self_check()
    if args.benchmark == "synthetic":
        run_synthetic(args.output, args.width, args.height)
    elif args.benchmark == "chair":
        from gsdepth.benchmarks.public_benchmark import run_chair
        run_chair(args.data, args.output, args.size)
    else:
        print("Two-pixel alpha-compositing check passed.")


if __name__ == "__main__":
    main()
