"""Render depth-error evidence and coverage curves from saved experiment outputs."""

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, PercentFormatter
import numpy as np

from gsdepth.benchmarks.experiment import METHODS


STYLES = {
    "opacity": ("#8d939c", "--", "o"),
    "gradient": ("#51637c", "-.", "s"),
    "std": ("#00785d", "-", "^"),
    "spread": ("#6d45b6", "-", "D"),
}


def export(figure, stem, require_matplotlib_panel_alignment):
    if require_matplotlib_panel_alignment is not None:
        require_matplotlib_panel_alignment(figure, json_out=stem.with_suffix(".alignment.json"), strict=True)
    figure.savefig(stem.with_suffix(".pdf"))
    figure.savefig(stem.with_suffix(".svg"))
    figure.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(figure)


def evidence(directory, metadata, audit):
    with np.load(directory / "example.npz") as data:
        rgb = data["rgb"]
        valid = data["valid"]
        error = data["error"]
        risk = data["spread"]
        retained = data["retained"]
        threshold = float(data["threshold"])
    error_max = max(threshold, np.percentile(error[valid], 99))
    risk_max = max(threshold, np.percentile(risk[valid], 99))
    figure = plt.figure(figsize=(9, 3.1))
    grid = figure.add_gridspec(1, 4, left=.047, right=.985, top=.83, bottom=.25, wspace=.24)
    axes = [figure.add_subplot(grid[0, index]) for index in range(4)]
    figure.suptitle("Where Gaussian depth becomes unreliable", x=.047, y=.98, ha="left", fontweight="bold")
    axes[0].imshow(rgb)
    error_map = axes[1].imshow(np.where(valid, error, np.nan), cmap="magma", vmin=0, vmax=error_max)
    risk_map = axes[2].imshow(np.where(valid, risk, np.nan), cmap="viridis", vmin=0, vmax=risk_max)
    filtered = rgb*0.65 + 0.35
    filtered[valid & ~retained] = [0.72, 0.35, 0.19]
    axes[3].imshow(filtered)
    for axis, name in zip(axes, ("Gaussian render", "Absolute depth error", "Weighted depth interval", "Pixels rejected at 90%")):
        axis.set_title(name, loc="left", fontweight="bold", pad=7)
        axis.set_axis_off()
    figure.canvas.draw()
    for axis, im in ((axes[1], error_map), (axes[2], risk_map)):
        bounds = axis.get_position()
        scale = figure.add_axes([bounds.x0, .153, bounds.width, .030], label=f"<colorbar-{im.get_cmap().name}>")
        colorbar = figure.colorbar(im, cax=scale, orientation="horizontal", ticks=[0, im.norm.vmax])
        colorbar.ax.xaxis.set_major_formatter(FormatStrFormatter("%.3g"))
        colorbar.ax.tick_params(length=2, pad=2)
    figure.text(.047, .046, f"Color bars are in {metadata['units']}. Orange marks pixels rejected by the interval score.", fontsize=8)
    export(figure, directory / "depth_evidence", audit)


def coverage_curve(directory, metadata, audit):
    with (directory / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    figure.subplots_adjust(left=.13, right=.97, top=.83, bottom=.26)
    figure.suptitle("Depth error at matched pixel retention", x=.13, y=.965, ha="left", fontweight="bold")
    axis.set_title("Mean across benchmark views or cases", loc="left", pad=8, color="#535e70")
    for method, label in METHODS.items():
        grouped = {}
        for row in rows:
            if row["method"] == method:
                grouped.setdefault(float(row["target_coverage"]), []).append(float(row["mae"]))
        x = sorted(grouped)
        y = [np.mean(grouped[value]) for value in x]
        color, line, marker = STYLES[method]
        axis.plot(x, y, color=color, linestyle=line, marker=marker, linewidth=1.8, markersize=3.2, label=label)
    axis.set(xlabel="Eligible pixels retained", ylabel=f"Mean absolute depth error ({metadata['units']})")
    axis.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axis.set_xlim(.49, 1.01)
    axis.set_ylim(bottom=0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=.17)
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower left", bbox_to_anchor=(.123, .036), ncols=2, frameon=False)
    export(figure, directory / "risk_coverage", audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--audit-tools", type=Path, help="Optional path to the panel-alignment QA utility")
    args = parser.parse_args()
    audit = None
    if args.audit_tools is not None:
        sys.path.insert(0, str(args.audit_tools))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        audit = require_matplotlib_panel_alignment
    metadata = json.loads((args.results / "summary.json").read_text(encoding="utf-8"))
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
        "font.size": 9, "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.linewidth": 0.7, "savefig.facecolor": "white",
    })
    evidence(args.results, metadata, audit)
    coverage_curve(args.results, metadata, audit)


if __name__ == "__main__":
    main()
