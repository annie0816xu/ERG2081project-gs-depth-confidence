"""Create the three paper figures from the repository's recorded evidence."""

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np

from gsdepth.tools.prepare_ray_cases import DATASETS as RAY_DATASETS, paired_rows


INK = "#26364a"
MUTED = "#56667a"
TEAL = "#176d62"
AMBER = "#a45d34"
PURPLE = "#6849a7"
BLUE = "#235e8b"
GRAY = "#b2bbc7"
METHODS = {"geometry": "Geometry", "photo": "RGB", "joint": "Joint"}
DATASETS = {"chair": "Chair", "sign": "Warning sign"}


def save(figure, path, require_matplotlib_panel_alignment):
    if require_matplotlib_panel_alignment:
        require_matplotlib_panel_alignment(figure, json_out=path.with_suffix(".alignment.json"), strict=True)
    figure.savefig(path.with_suffix(".pdf"))
    figure.savefig(path.with_suffix(".svg"))
    figure.savefig(path.with_suffix(".png"), dpi=600)
    figure.savefig(path.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(figure)


def heading(axis, letter, title):
    axis.set_title(f"{letter}   {title}", fontweight="bold", loc="left", pad=7)


def method_figure(directory, provenance, audit):
    figure, (ray, reprojection, votes) = plt.subplots(1, 3, figsize=(7.16, 2.28))
    figure.subplots_adjust(left=.027, right=.975, top=.865, bottom=.075, wspace=.21)
    for axis in (ray, reprojection, votes):
        axis.set(xlim=(0, 1), ylim=(0, 1))
        axis.set_axis_off()

    heading(ray, "a", "Separate the ray layers")
    ray.annotate("", xy=(.96, .30), xytext=(.04, .30),
                 arrowprops={"arrowstyle": "->", "color": INK, "lw": .85})
    z = np.linspace(.06, .93, 220)
    for center, width, color, label in ((.28, .078, TEAL, "Near"), (.74, .061, AMBER, "Far")):
        density = .34 * np.exp(-.5*((z - center)/width)**2)
        ray.fill_between(z, .31, .31 + density, color=color, alpha=.20)
        ray.plot(z, .31 + density, color=color, lw=1.2)
        ray.text(center, .74, label, color=color, ha="center", fontweight="bold")
        ray.plot(center, .30, "o", color=color, markersize=4.3, zorder=4)
    ray.text(.02, .06, "Dots show the layer medians", color=MUTED, fontsize=7)
    ray.text(.96, .20, "Depth", ha="right", va="top", fontsize=7)
    ray.plot(.49, .30, "x", color=INK, markersize=5, zorder=4)
    ray.annotate("Mean", xy=(.49, .34), xytext=(.50, .49), ha="center", fontsize=7,
                 arrowprops={"arrowstyle": "-", "color": MUTED, "lw": .7})
    ray.text(.02, .89, "Compositing weight (schematic)", color=MUTED, fontsize=7)

    heading(reprojection, "b", "Check the same candidate")
    anchor = (.10, .40)
    neighbor = (.84, .56)
    candidate = (.45, .65)
    reprojection.plot([anchor[0], candidate[0]], [anchor[1], candidate[1]], color=INK, lw=.9)
    reprojection.plot([candidate[0], neighbor[0]], [candidate[1], neighbor[1]],
                      color=MUTED, lw=.9, linestyle="--")
    for (x, y), marker, color in ((anchor, ">", INK), (neighbor, "<", INK), (candidate, "o", TEAL)):
        reprojection.plot(x, y, marker, color=color, markersize=5.5, zorder=5)
    for (x, y), angle in (((.22, .51), -24), ((.71, .59), 22)):
        reprojection.add_patch(Rectangle((x-.045, y-.045), .09, .09, angle=angle,
                                        rotation_point="center", facecolor="#e5eeef",
                                        edgecolor=MUTED, lw=.75, zorder=3))
    reprojection.text(.45, .76, "3D candidate", ha="center", fontweight="bold", color=TEAL)
    reprojection.text(.04, .31, "Target view", va="top", fontsize=7)
    reprojection.text(.96, .45, "Neighbor view", ha="right", va="top", fontsize=7)
    reprojection.text(.02, .06, "Geometry   Depth agrees\nPhotometry   Patch cost falls",
                      va="bottom", fontsize=7, linespacing=1.5)

    heading(votes, "c", "Vote in four nearby views")
    case = provenance["sign_false_edit"]
    positions = (.45, .60, .75, .90)
    for x, view in zip(positions, case["neighbor_views"]):
        votes.text(x, .71, str(view), ha="center", color=MUTED, fontsize=7)
    rows = (
        ("Geometry", .55, [state == 1 for state in case["geometry_states"]]),
        ("Photometry", .41, case["photo_pass"]),
        ("Joint vote", .25, case["joint_pass"]),
    )
    for name, y, passes in rows:
        votes.text(.02, y, name, va="center", fontsize=7, fontweight="bold" if name == "Joint vote" else None)
        for x, passed in zip(positions, passes):
            votes.plot(x, y, "o", markersize=5, markeredgecolor=TEAL if passed else GRAY,
                       markerfacecolor=TEAL if passed else "white", markeredgewidth=.85)
    votes.text(.02, .86, "Near candidate on the sign (Fig. 2b)", color=MUTED, fontsize=7)
    votes.text(.02, .06, f"{sum(case['joint_pass'])} joint votes, no conflict → edit", color=TEAL,
               fontsize=7, fontweight="bold")
    save(figure, directory / "fig1_method", audit)


def read_comparisons(evidence, source_data):
    data = {}
    paired_rows = []
    for dataset in DATASETS:
        path = evidence / f"{dataset}_heldout" / "metrics.csv"
        with path.open(encoding="utf-8", newline="") as stream:
            observed = {
                (int(row["view"]), row["method"]): row
                for row in csv.DictReader(stream) if float(row["threshold"]) == 0.05
            }
        views = sorted({view for view, method in observed if method == "median"})
        if len(views) != 8:
            raise ValueError(f"Expected eight held-out views: {path}")
        data[dataset] = {}
        for method in METHODS:
            paired = []
            for view in views:
                entry, median = observed[view, method], observed[view, "median"]
                if entry["valid"] != median["valid"]:
                    raise ValueError(f"Coverage mismatch in {dataset}, view {view}")
                values = {
                    "dataset": dataset, "view": view, "method": method, "valid": int(entry["valid"]),
                    "median_mae": float(median["mae"]), "method_mae": float(entry["mae"]),
                    "mae_change": float(entry["mae"]) - float(median["mae"]),
                    "fixed": int(entry["fixed"]), "broken": int(entry["broken"]),
                }
                paired.append(values)
                paired_rows.append(values)
            data[dataset][method] = paired
    with (source_data / "fig3_paired_view_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=paired_rows[0])
        writer.writeheader()
        writer.writerows(paired_rows)
    return data


def ablation_figure(directory, evidence, source_data, audit):
    data = read_comparisons(evidence, source_data)
    figure, axes = plt.subplots(1, 3, figsize=(7.16, 2.53))
    figure.subplots_adjust(left=.092, right=.973, top=.84, bottom=.24, wspace=.35)
    positions = np.arange(len(METHODS))
    shifts = np.linspace(-.17, .17, 8)
    spans = []
    for idx, (dataset, label) in enumerate(DATASETS.items()):
        axis = axes[idx]
        heading(axis, "ab"[idx], label)
        axis.axhline(0, color=GRAY, linewidth=.75, linestyle="--", zorder=0)
        for i, method in enumerate(METHODS):
            changes = 1000 * np.array([row["mae_change"] for row in data[dataset][method]])
            spans.extend(changes)
            axis.scatter(i + shifts, changes, s=12, color=MUTED, alpha=.75, linewidths=0)
            axis.scatter(i, np.mean(changes), s=32, marker="D", color=PURPLE,
                         edgecolor="white", linewidth=.5, zorder=4)
        axis.set_xticks(positions, METHODS.values())
        axis.set_xlim(-.43, 2.43)
        axis.yaxis.set_major_locator(MaxNLocator(5))
    low, high = min(spans), max(spans)
    for axis in axes[:2]:
        axis.set_ylim(low - .30, high + .33)
    axes[0].set_ylabel("Δ MAE vs. median\n(0.001 scene units)")
    axes[1].tick_params(labelleft=False)
    figure.legend(
        handles=[
            Line2D([0], [0], color=MUTED, marker="o", linestyle="", markersize=4, label="Held-out view"),
            Line2D([0], [0], color=PURPLE, marker="D", linestyle="", markersize=4.5, label="Mean of 8 views"),
        ],
        loc="lower center", bbox_to_anchor=(.355, .02), ncols=2, handletextpad=.4, columnspacing=1.2,
    )

    axis = axes[2]
    heading(axis, "c", "Both objects")
    axis.axvline(0, color=GRAY, lw=.75, zorder=0)
    for key, color, direction in (("fixed", TEAL, 1), ("broken", AMBER, -1)):
        counts = [sum(row[key] for dataset in data.values() for row in dataset[method]) for method in METHODS]
        bars = axis.barh(positions, direction * np.array(counts), height=.48, color=color)
        for position, count, bar in zip(positions, counts, bars):
            if count >= 15:
                axis.text(bar.get_width() / 2, position, str(count), ha="center", va="center",
                          color="white", fontsize=7, fontweight="bold")
            else:
                axis.annotate(str(count), (bar.get_width(), position), xytext=(3 * direction, 0),
                              textcoords="offset points", ha="left" if direction > 0 else "right",
                              va="center", color=color, fontsize=7)
    axis.set_yticks(positions, METHODS.values())
    axis.invert_yaxis()
    axis.set_xlim(-37, 108)
    axis.set_xticks([-25, 0, 25, 50, 75, 100], ["25", "0", "25", "50", "75", "100"])
    axis.tick_params(axis="y", length=0, pad=3)
    axis.set_xlabel("Pixels crossing the threshold", labelpad=3)
    axis.text(.00, -.36, "Made worse", transform=axis.transAxes, color=AMBER, fontsize=7, fontweight="bold")
    axis.text(.98, -.36, "Fixed", transform=axis.transAxes, ha="right", color=TEAL, fontsize=7, fontweight="bold")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(length=2.4, width=.7)
    axis = axes[2]
    axis.spines["left"].set_visible(False)
    axis.tick_params(axis="y", length=0)
    save(figure, directory / "fig3_ablation", audit)


def outline(axis, xy, size, color):
    for stroke, width in ((INK, 2.0), (color, 1.05)):
        axis.add_patch(Rectangle(xy, *size, linewidth=width, edgecolor=stroke, facecolor="none"))


def scale_bar(figure, image, position, ticks, ticklabels=None, extend="neither"):
    scale = figure.add_axes(position, label="<colorbar>")
    bar = figure.colorbar(image, cax=scale, orientation="horizontal", ticks=ticks, extend=extend)
    bar.ax.tick_params(length=2, pad=1, labelsize=7)
    if ticklabels is not None:
        bar.set_ticklabels(ticklabels)
    return bar


def case_figure(directory, source_data, provenance, audit):
    figure, axes = plt.subplots(2, 5, figsize=(7.16, 4.05))
    figure.subplots_adjust(left=.031, right=.969, top=.852, bottom=.175, wspace=.075, hspace=.50)
    depth_cmap = plt.get_cmap("cividis").with_extremes(bad="#ebedf0")
    change_cmap = plt.get_cmap("RdBu_r").with_extremes(bad="#d7dbe1")
    for axis, name in zip(axes[0], ("Input RGB", "Reference depth", "Weighted median", "Joint refinement", "Error change")):
        axis.set_title(name, loc="center", pad=5, fontweight="bold", fontsize=8)
    for row, (name, label, color) in enumerate((
        ("chair_fixed", "(a) Chair · view 12", BLUE), ("sign_false_edit", "(b) Warning sign · view 120", AMBER),
    )):
        record = provenance[name]
        with np.load(source_data / f"{name}.npz") as case:
            rgb, truth, valid = case["full_rgb"], case["truth"], case["valid"]
            median, joint = case["median"], case["joint"]
            x, y = case["marker_xy"]

        rgb_axis = axes[row, 0]
        rgb_axis.imshow(rgb, interpolation="nearest")
        left, top, width, height = record["crop_xywh"]
        outline(rgb_axis, (left-.5, top-.5), (width, height), "#ffde68")
        for axis in axes[row]:
            axis.set_axis_off()
        row_top = rgb_axis.get_position().y1
        figure.text(.032, row_top + (.090 if row == 0 else .039), label, color=color, fontweight="bold", fontsize=9)

        masked_depths = [np.where(valid, image, np.nan) for image in (truth, median, joint)]
        lo = float(np.floor(10 * np.nanmin(masked_depths)) / 10)
        hi = float(np.ceil(10 * np.nanmax(masked_depths)) / 10)
        for col, image in enumerate(masked_depths, start=1):
            axis = axes[row, col]
            depth_image = axis.imshow(image, interpolation="nearest", cmap=depth_cmap, vmin=lo, vmax=hi)
            outline(axis, (x-1.5, y-1.5), (3, 3), "#ffffff")
        before, after = abs(median[y, x] - truth[y, x]), abs(joint[y, x] - truth[y, x])
        np.testing.assert_allclose([before, after], [record["median_abs_error"], record["joint_abs_error"]],
                                   rtol=0, atol=1e-12)
        change = np.where(valid, np.abs(joint - truth) - np.abs(median - truth), np.nan)
        image_diff = axes[row, 4].imshow(change, interpolation="nearest", cmap=change_cmap, vmin=-1.2, vmax=1.2)
        outline(axes[row, 4], (x-1.5, y-1.5), (3, 3), "#ffffff")
        labels = ("29 × 29 zoom", f"Depth {truth[y, x]:.3f}", f"Error {before:.3f}", f"Error {after:.3f}",
                  f"Δ {after - before:+.3f}")
        for col, (axis, caption) in enumerate(zip(axes[row], labels)):
            axis.text(.50, -.050, caption, ha="center", va="top", transform=axis.transAxes,
                      fontsize=7, color=color if col == 4 else INK, fontweight="bold" if col == 4 else None)
        position = axes[row, 2].get_position()
        y_bar = position.y0 - .085
        bar = scale_bar(figure, depth_image, [.326, y_bar, .34, .012], [lo, hi], [f"{lo:.1f}", f"{hi:.1f}"])
        bar.set_label("Depth (scene units)", fontsize=7, labelpad=-5)

    bar = scale_bar(figure, image_diff, [.801, .058, .158, .015], [-1.2, 0, 1.2],
                    ["−1.2", "0", "+1.2"], extend="both")
    bar.set_label("Δ error (scene units)", fontsize=7, labelpad=1)
    save(figure, directory / "fig2_comparison", audit)


def ray_metrics_figure(directory, evidence, source_data, audit):
    pairs = paired_rows(evidence)
    with (source_data / "fig4_ray_depth_paired.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=pairs[0])
        writer.writeheader()
        writer.writerows(pairs)

    figure, (change, tail) = plt.subplots(1, 2, figsize=(7.16, 2.72), sharey=True)
    figure.subplots_adjust(left=.177, right=.970, top=.87, bottom=.28, wspace=.22)
    heading(change, "a", "Paired MAE change")
    heading(tail, "b", "Pixels with error above 0.05")
    change.axvline(0, color=GRAY, lw=.85, linestyle="--", zorder=0)
    labels = []
    shifts = np.linspace(-.125, .125, 8)
    for index, (dataset, label) in enumerate(RAY_DATASETS.items()):
        scores = [row for row in pairs if row["dataset"] == dataset]
        deltas = 1000 * np.array([row["delta_mae"] for row in scores])
        labels.append(f"{label}  {np.count_nonzero(deltas < 0)}/8")
        change.scatter(deltas, index + shifts, s=20, c=np.where(deltas < 0, TEAL, AMBER), linewidths=0)
        change.scatter(deltas.mean(), index, marker="D", s=33, color=INK, edgecolor="white", linewidth=.55, zorder=4)

        size = sum(row["valid"] for row in scores)
        before, after = (sum(row[field] for row in scores) for field in ("center_bad", "ray_bad"))
        left, right = 100 * after / size, 100 * before / size
        tail.plot([left, right], [index + .045, index - .045], color=GRAY, lw=1.25, zorder=1)
        tail.scatter(right, index - .045, s=27, facecolor="white", edgecolor=MUTED, linewidth=1.0, zorder=2)
        tail.scatter(left, index + .045, s=24, color=BLUE, edgecolor="white", linewidth=.4, zorder=3)
        tail.text(max(left, right) + .60, index, f"{after - before:+,}", va="center", fontsize=7, color=BLUE)

    change.set(yticks=np.arange(len(labels)), yticklabels=labels, ylim=(2.54, -.54), xlim=(-6.7, 3.2),
               xlabel="Δ MAE (0.001 scene units)")
    tail.set(xlim=(5, 22.2), xticks=[5, 10, 15, 20], xlabel="Share of valid pixels (%)")
    for axis in (change, tail):
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
        axis.tick_params(axis="x", length=2.5, width=.7)
        axis.set_axisbelow(True)
    tail.tick_params(labelleft=False)

    left_legend = [Line2D([0], [0], color=color, marker=marker, linestyle="", markersize=4.2, label=label)
                   for marker, color, label in (("o", TEAL, "Lower"), ("o", AMBER, "Higher"), ("D", INK, "View mean"))]
    right_legend = [Line2D([0], [0], marker="o", linestyle="", markerfacecolor=fill, markeredgecolor=edge,
                           markersize=4.2, label=label)
                    for fill, edge, label in (("white", MUTED, "Center depth"), (BLUE, BLUE, "Ray depth"))]
    figure.legend(handles=left_legend, loc="lower center", bbox_to_anchor=(.355, .025), ncols=3,
                  handletextpad=.3, columnspacing=.9)
    figure.legend(handles=right_legend, loc="lower center", bbox_to_anchor=(.792, .025), ncols=2,
                  handletextpad=.3, columnspacing=1.25)
    save(figure, directory / "fig4_ray_depth_metrics", audit)


def ray_case_figure(directory, source_data, audit):
    provenance = json.loads((source_data / "ray_case_provenance.json").read_text(encoding="utf-8"))
    figure, axes = plt.subplots(3, 5, figsize=(7.16, 5.8))
    figure.subplots_adjust(left=.03, right=.97, top=.89, bottom=.155, wspace=.075, hspace=.45)
    cmap = plt.get_cmap("cividis").with_extremes(bad="#eaecf0")
    delta_cmap = plt.get_cmap("RdBu_r").with_extremes(bad="#d7dbe1")
    columns = ("Input RGB", "Reference depth", "Center depth + 5%", "Ray depth + 5%", "Error change")
    for axis, label in zip(axes[0], columns):
        axis.set_title(label, pad=5, fontweight="bold", fontsize=8)

    for row, case in enumerate(provenance["examples"]):
        dataset = case["dataset"]
        with np.load(source_data / f"ray_case_{dataset}.npz") as arrays:
            rgb, valid = arrays["full_rgb"], arrays["valid"]
            truth, center, ray = (np.where(valid, arrays[key], np.nan) for key in ("truth", "center", "ray"))
        for axis in axes[row]:
            axis.set_axis_off()
        left, top, width, height = case["crop_xywh"]
        input_axis = axes[row, 0]
        input_axis.imshow(rgb, interpolation="nearest")
        outline(input_axis, (left - .5, top - .5), (width, height), "#ffde68")
        change = np.abs(ray - truth) - np.abs(center - truth)
        row_color = BLUE if case["delta_mae"] < 0 else AMBER
        title = f"({chr(97 + row)}) {RAY_DATASETS[dataset]} · view {case['view']}"
        figure.text(.031, input_axis.get_position().y1 + (.062 if row == 0 else .038),
                    title, fontsize=9, fontweight="bold", color=row_color)

        depths = np.stack((truth, center, ray))
        lo = np.floor(np.nanmin(depths) * 10) / 10
        hi = np.ceil(np.nanmax(depths) * 10) / 10
        if lo == hi:
            hi = lo + .1
        for axis, depth in zip(axes[row, 1:4], depths):
            depth_image = axis.imshow(depth, interpolation="nearest", cmap=cmap, vmin=lo, vmax=hi)
        delta_axis = axes[row, 4]
        diff = delta_axis.imshow(change, interpolation="nearest", cmap=delta_cmap, vmin=-.10, vmax=.10)
        delta_axis.text(.5, -.065, f"View Δ MAE {case['delta_mae']:+.4f}", transform=delta_axis.transAxes,
                        ha="center", va="top", color=row_color, fontweight="bold", fontsize=7)
        bar = scale_bar(figure, depth_image, [.330, axes[row, 2].get_position().y0 - .031, .34, .010],
                        [lo, hi], [f"{lo:.1f}", f"{hi:.1f}"])
        bar.set_label("Depth (scene units)", labelpad=-5, fontsize=7)

    bar = scale_bar(figure, diff, [.802, .055, .158, .012], [-.1, 0, .1], ["−0.10", "0", "+0.10"], extend="both")
    bar.set_label("Δ error (scene units)", labelpad=2, fontsize=7)
    save(figure, directory / "fig5_ray_depth_views", audit)


def surface_figure(directory, evidence, audit):
    figure, (mean, outliers) = plt.subplots(1, 2, figsize=(7.16, 2.65), sharey=True)
    figure.subplots_adjust(left=.158, right=.971, top=.86, bottom=.26, wspace=.21)
    heading(mean, "a", "Mean distance to reference")
    heading(outliers, "b", "Off-surface points above 0.05")
    outliers.axvline(0, color=GRAY, lw=.8, linestyle="--", zorder=0)

    for row, dataset in enumerate(RAY_DATASETS):
        path = evidence / "surface_validation" / dataset / "surface_metrics.csv"
        with path.open(encoding="utf-8", newline="") as stream:
            values = {entry["method"]: entry for entry in csv.DictReader(stream)}
        center, ray = values["center"], values["ray"]
        left, right = (1000 * float(method["mean_distance"]) for method in (ray, center))
        mean.plot([left, right], [row + .045, row - .045], color=GRAY, lw=1.25, zorder=1)
        mean.scatter(right, row - .045, s=28, facecolor="white", edgecolor=MUTED, linewidth=1.0, zorder=2)
        mean.scatter(left, row + .045, s=24, color=BLUE, edgecolor="white", linewidth=.4, zorder=3)
        change = int(ray["outliers_0.05"]) - int(center["outliers_0.05"])
        color = TEAL if change < 0 else AMBER if change else MUTED
        outliers.plot([0, change], [row, row], color=color, lw=1.1, zorder=1)
        outliers.scatter(change, row, s=30, color=color, zorder=2)
        outliers.annotate(f"{change:+d}" if change else "0", (change, row),
                          xytext=(-5 if change < 0 else 5, 0), textcoords="offset points",
                          va="center", ha="right" if change < 0 else "left", color=color, fontsize=7)

    mean.set(yticks=np.arange(len(RAY_DATASETS)), yticklabels=list(RAY_DATASETS.values()), ylim=(2.54, -.54),
             xlim=(8.0, 21.0), xticks=[10, 15, 20], xlabel="Distance (0.001 scene units)")
    outliers.set(xlim=(-162, 162), xticks=[-150, -75, 0, 75, 150], xlabel="Change in number of points")
    for axis in (mean, outliers):
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
        axis.tick_params(axis="x", length=2.5, width=.7)
    outliers.tick_params(labelleft=False)
    legend = [Line2D([0], [0], marker="o", linestyle="", markerfacecolor=fill, markeredgecolor=edge,
                     markersize=4.2, label=label)
              for fill, edge, label in (("white", MUTED, "Center depth"), (BLUE, BLUE, "Ray depth"))]
    figure.legend(handles=legend, loc="lower center", bbox_to_anchor=(.35, .025), ncols=2,
                  handletextpad=.3, columnspacing=1.2)
    save(figure, directory / "fig6_surface", audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("figures"))
    parser.add_argument("--evidence", type=Path, default=Path("evidence"))
    parser.add_argument("--audit-tools", type=Path, help="Optional directory containing the panel-alignment audit")
    args = parser.parse_args()
    source_data = args.output / "source_data"
    source_data.mkdir(parents=True, exist_ok=True)
    audit = None
    if args.audit_tools is not None:
        sys.path.insert(0, str(args.audit_tools))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        audit = require_matplotlib_panel_alignment
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
        "axes.edgecolor": INK, "axes.linewidth": .75, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK, "savefig.facecolor": "white",
        "pdf.fonttype": 42, "svg.fonttype": "none", "legend.frameon": False,
    })
    provenance = json.loads((source_data / "case_provenance.json").read_text(encoding="utf-8"))
    method_figure(args.output, provenance, audit)
    case_figure(args.output, source_data, provenance, audit)
    ablation_figure(args.output, args.evidence, source_data, audit)
    ray_metrics_figure(args.output, args.evidence, source_data, audit)
    ray_case_figure(args.output, source_data, audit)
    surface_figure(args.output, args.evidence, audit)


if __name__ == "__main__":
    main()
