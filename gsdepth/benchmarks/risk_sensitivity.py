"""Compare depth-spread percentiles at fixed retention on the three archived objects."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from gsdepth.benchmarks.experiment import save_results, score_frame
from gsdepth.benchmarks.public_benchmark import camera_at, normalized_poses, read_truth
from gsdepth.benchmarks.run_refinement import SOURCES
from gsdepth.core.renderer import load_gaussian_ply, render


ALTERNATIVE_INTERVALS = {(.05, .95): "spread_05_95", (.15, .85): "spread_15_85"}


def compare(dataset, data, evidence, output):
    group = evidence / "followup" / "weak_layer_guard" / f"{dataset}_heldout"
    recorded = json.loads((group / "summary.json").read_text(encoding="utf-8"))
    root = data / dataset
    for filename, expected in recorded["source_sha256"].items():
        with (root / filename).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"Unexpected source file: {root / filename}")
    gaussians = load_gaussian_ply(root / "scene.ply")
    size = recorded["size"]
    rows = []
    with ZipFile(root / "reference.zip") as archive:
        metadata = json.loads(archive.read("transforms.json"))
        intrinsics = np.array([[metadata["fl_x"], 0, metadata["cx"]],
                               [0, metadata["fl_y"], metadata["cy"]], [0, 0, 1]])
        intrinsics = np.diag([size / metadata["w"], size / metadata["h"], 1]) @ intrinsics
        poses = np.array([frame["transform_matrix"] for frame in metadata["frames"]])
        scale = 1.0
        if recorded["normalization"] == "nerfstudio_default":
            poses, scale = normalized_poses(poses)
        np.testing.assert_allclose(scale, recorded["model_scale"], rtol=0, atol=1e-12)
        for view in recorded["views"]:
            camera = camera_at(poses[view], intrinsics, size)
            frame = render(gaussians, camera, spread_intervals=tuple(ALTERNATIVE_INTERVALS))
            scores = {label: frame["alternative_spreads"][bounds] / scale
                      for bounds, label in ALTERNATIVE_INTERVALS.items()}
            for key in ("depth", "spread", "std"):
                frame[key] /= scale
            truth = read_truth(archive, metadata, view, size)
            view_rows, *_ = score_frame(frame, truth, .05, additional_scores=scores)
            rows.extend({"view": view, **row} for row in view_rows)

    output.mkdir(parents=True, exist_ok=True)
    save_results(output, rows, {
        "dataset": dataset, "views": recorded["views"], "size": size,
        "depth": "Center-depth expected value, matching the original risk scoring stage",
        "threshold": .05, "source_sha256": recorded["source_sha256"],
        "intervals": {"spread": [.10, .90],
                      **{key: bounds for bounds, key in ALTERNATIVE_INTERVALS.items()}},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", choices=SOURCES, required=True)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--evidence", type=Path, default=Path("evidence"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    compare(args.object, args.data, args.evidence,
            args.output or Path("results/risk_sensitivity") / args.object)


if __name__ == "__main__":
    main()
