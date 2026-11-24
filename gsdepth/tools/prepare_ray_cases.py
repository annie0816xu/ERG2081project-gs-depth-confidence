"""Replay paired center-depth and ray-depth views for the paper comparison."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from gsdepth.tools.prepare_paper_cases import replay


DATASETS = {"chair": "Chair", "sign": "Warning sign", "drill": "Drill"}
METHOD = "geometry_mass05"
THRESHOLD = 0.05
WINDOW = 29


def paired_rows(evidence):
    """Pair all eight scored views per data set on an identical evaluation mask size."""
    runs = evidence / "followup"
    pairs = []
    for dataset in DATASETS:
        sources = []
        for variant in ("weak_layer_guard", "ray_depth"):
            filename = runs / variant / f"{dataset}_heldout" / "metrics.csv"
            with filename.open(encoding="utf-8", newline="") as file:
                rows = [row for row in csv.DictReader(file)
                        if row["method"] == METHOD and float(row["threshold"]) == THRESHOLD]
            scores = {int(row["view"]): row for row in rows}
            if len(scores) != len(rows) or len(scores) != 8:
                raise ValueError(f"Expected eight unique views in {filename}")
            sources.append(scores)
        center_scores, ray_scores = sources
        if center_scores.keys() != ray_scores.keys():
            raise ValueError(f"Unpaired views for {dataset}")
        for view, center in sorted(center_scores.items()):
            ray = ray_scores[view]
            if center["valid"] != ray["valid"]:
                raise ValueError(f"Mismatched evaluation coverage for {dataset}, view {view}")
            center_mae, ray_mae = float(center["mae"]), float(ray["mae"])
            pairs.append({"dataset": dataset, "view": view, "valid": int(center["valid"]),
                          "center_mae": center_mae, "ray_mae": ray_mae, "delta_mae": ray_mae - center_mae,
                          "center_bad": int(center["bad"]), "ray_bad": int(ray["bad"])})
    return pairs


def write_comparison(output, row, data, evidence, expect_improvement):
    dataset, view = row["dataset"], row["view"]
    groups = evidence / "followup"
    before, source, neighbors, *_ = replay(dataset, view, data, groups / "weak_layer_guard", METHOD)
    after, ray_source, ray_neighbors, *_ = replay(dataset, view, data, groups / "ray_depth", METHOD)
    if source != ray_source or neighbors != ray_neighbors:
        raise ValueError(f"Replays are not paired for {dataset}, view {view}")
    for key in ("rgb", "truth", "valid"):
        np.testing.assert_array_equal(before[key], after[key])

    valid = before["valid"]
    center_error = np.abs(before["geometry"] - before["truth"])
    ray_error = np.abs(after["geometry"] - before["truth"])
    difference = ray_error - center_error
    direction = -1 if expect_improvement else 1
    score = sliding_window_view(np.where(valid, direction * difference, 0), (WINDOW, WINDOW)).sum(axis=(-2, -1))
    if score.max() <= 0:
        raise ValueError(f"No crop shows the signed change of {dataset}, view {view}")
    y, x = np.unravel_index(np.argmax(score), score.shape)
    crop = (slice(y, y + WINDOW), slice(x, x + WINDOW))
    crop_valid = valid[crop]
    np.savez_compressed(output / f"ray_case_{dataset}.npz", full_rgb=before["rgb"],
                        truth=before["truth"][crop], center=before["geometry"][crop],
                        ray=after["geometry"][crop], valid=crop_valid)
    return {**row, "source": source, "neighbor_views": neighbors,
            "crop_xywh": [int(x), int(y), WINDOW, WINDOW], "crop_valid": int(crop_valid.sum()),
            "crop_center_mae": float(center_error[crop][crop_valid].mean()),
            "crop_ray_mae": float(ray_error[crop][crop_valid].mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--evidence", type=Path, default=Path("evidence"))
    parser.add_argument("--output", type=Path, default=Path("figures/source_data"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pairs = paired_rows(args.evidence)
    examples = []
    for dataset, improved in (("chair", True), ("drill", True), ("sign", False)):
        choose = min if improved else max
        row = choose((item for item in pairs if item["dataset"] == dataset), key=lambda item: item["delta_mae"])
        examples.append(write_comparison(args.output, row, args.data, args.evidence, improved))
        print(f"{dataset}, view {row['view']}: {row['delta_mae']:+.6f} view MAE", flush=True)
    provenance = {"view_selection": "Smallest paired MAE change for chair and drill; largest for warning sign.",
                  "crop_selection": "Largest directional sum of per-pixel error changes in any 29x29 window.",
                  "method": METHOD, "examples": examples}
    (args.output / "ray_case_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
