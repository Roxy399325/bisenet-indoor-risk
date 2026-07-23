#!/usr/bin/env python3
"""Validate indoor-risk score monotonicity and component-weight sensitivity."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
import os
import os.path as osp
import sys
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np


PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lib.risk.bisenet_features import (  # noqa: E402
    RISK_SCORING_VERSION,
    analyze_bisenet,
    render_analysis_overlay,
)
from lib.risk.scoring_validation import (  # noqa: E402
    DEFAULT_WEIGHT_DELTAS,
    RISK_LEVEL_THRESHOLDS,
    build_monotonic_sweeps,
    run_monotonicity_validation,
    run_weight_sensitivity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run controlled monotonicity sweeps and top-level weight sensitivity "
            "analysis for the indoor-risk score."
        )
    )
    parser.add_argument(
        "--output-dir",
        default=osp.join(PROJECT_ROOT, "validation", "scoring_validation"),
        help="Directory for JSON, CSV and PNG reports.",
    )
    parser.add_argument(
        "--weight-deltas",
        nargs="+",
        type=float,
        default=list(DEFAULT_WEIGHT_DELTAS),
        metavar="FRACTION",
        help="Fractional one-at-a-time weight changes (default: -0.2 -0.1 0.1 0.2).",
    )
    parser.add_argument(
        "--boundary-margin",
        type=float,
        default=3.0,
        help="Scores within this distance of 25/50/75 are boundary cases.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write JSON and CSV only; do not import matplotlib or create PNGs.",
    )
    return parser.parse_args()


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _write_monotonicity_csv(path: str, report: Dict[str, Any]) -> None:
    fieldnames = (
        "series",
        "point_index",
        "value",
        "label",
        "score",
        "level",
        "passed",
        "tracked_features_json",
    )
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for series in report["series"]:
            for index, point in enumerate(series["points"]):
                writer.writerow(
                    {
                        "series": series["name"],
                        "point_index": index,
                        "value": point["value"],
                        "label": point["label"],
                        "score": point["score"],
                        "level": point["level"],
                        "passed": series["passed"],
                        "tracked_features_json": json.dumps(
                            point["features"], ensure_ascii=False
                        ),
                    }
                )


def _write_sensitivity_csv(path: str, report: Dict[str, Any]) -> None:
    fieldnames = (
        "component",
        "fractional_delta",
        "scenario",
        "baseline_score",
        "variant_score",
        "score_delta",
        "absolute_delta",
        "baseline_level",
        "variant_level",
        "distance_to_level_boundary",
        "level_changed",
        "non_boundary_level_changed",
        "level_jump",
        "spearman_rank_correlation",
    )
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for variant in report["variants"]:
            for row in variant["scenarios"]:
                output = {
                    "component": variant["component"],
                    "fractional_delta": variant["fractional_delta"],
                    "spearman_rank_correlation": variant[
                        "spearman_rank_correlation"
                    ],
                }
                output.update(row)
                writer.writerow(output)


def _plot_monotonicity(path: str, report: Dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series_list = report["series"]
    columns = 2
    rows = int(math.ceil(len(series_list) / float(columns)))
    figure, axes = plt.subplots(rows, columns, figsize=(13, 3.3 * rows))
    axes_array = np.asarray(axes).reshape(-1)
    for axis, series in zip(axes_array, series_list):
        labels = [point["label"] for point in series["points"]]
        scores = [point["score"] for point in series["points"]]
        x_values = np.arange(len(scores))
        colour = "#16825d" if series["passed"] else "#c0392b"
        axis.plot(x_values, scores, marker="o", linewidth=2, color=colour)
        for boundary in RISK_LEVEL_THRESHOLDS:
            axis.axhline(boundary, color="#aaaaaa", linewidth=0.7, linestyle="--")
        axis.set_title(series["name"].replace("_", " "))
        axis.set_xticks(x_values)
        axis.set_xticklabels(labels, rotation=25, ha="right")
        axis.set_ylabel("Risk score")
        axis.set_ylim(0, max(40.0, max(scores) + 5.0))
        axis.grid(axis="y", alpha=0.2)
    for axis in axes_array[len(series_list) :]:
        axis.set_visible(False)
    figure.suptitle("Controlled monotonicity validation", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _matrix_for_metric(
    report: Dict[str, Any],
    components: Sequence[str],
    deltas: Sequence[float],
    metric: str,
) -> np.ndarray:
    lookup = {
        (item["component"], float(item["fractional_delta"])): item
        for item in report["variants"]
    }
    return np.asarray(
        [
            [float(lookup[(component, float(delta))][metric]) for delta in deltas]
            for component in components
        ],
        dtype=np.float64,
    )


def _plot_sensitivity(path: str, report: Dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    components = list(report["base_weights"])
    deltas = sorted(
        {float(item["fractional_delta"]) for item in report["variants"]}
    )
    max_delta = _matrix_for_metric(
        report, components, deltas, "max_absolute_score_delta"
    )
    rank_correlation = _matrix_for_metric(
        report, components, deltas, "spearman_rank_correlation"
    )

    level_flips = _matrix_for_metric(
        report, components, deltas, "level_flip_count"
    )

    figure, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    panels = (
        (axes[0], max_delta, "Maximum absolute score change", "magma", ".2f"),
        (axes[1], rank_correlation, "Spearman rank correlation", "viridis", ".3f"),
        (axes[2], level_flips, "Risk-level flips", "cividis", ".0f"),
    )
    for axis, values, title, colour_map, number_format in panels:
        image = axis.imshow(values, aspect="auto", cmap=colour_map)
        axis.set_title(title)
        axis.set_xticks(np.arange(len(deltas)))
        axis.set_xticklabels(["{:+.0f}%".format(delta * 100.0) for delta in deltas])
        axis.set_yticks(np.arange(len(components)))
        axis.set_yticklabels([name.replace("_", " ") for name in components])
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                axis.text(
                    column,
                    row,
                    format(values[row, column], number_format),
                    ha="center",
                    va="center",
                    color="white" if values[row, column] < values.mean() else "black",
                    fontsize=9,
                )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle("Top-level component weight sensitivity", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _write_position_montage(path: str) -> None:
    sweep = build_monotonic_sweeps()[0]
    panels: List[np.ndarray] = []
    for point in sweep.points:
        scene = point.scene
        analysis = analyze_bisenet(
            scene.image,
            scene.mask,
            yolo_detections=scene.detections,
        )
        panel = render_analysis_overlay(scene.image, analysis)
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 34), (20, 20, 20), -1)
        cv2.putText(
            panel,
            "{}  score={:.2f}".format(point.label, analysis.risk.score),
            (8, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        panels.append(panel)
    montage = cv2.hconcat(panels)
    if not cv2.imwrite(path, montage):
        raise RuntimeError("Could not write position montage: {}".format(path))


def main() -> int:
    args = parse_args()
    if any(delta <= -1.0 for delta in args.weight_deltas):
        raise ValueError("every weight delta must be greater than -1.0")
    if args.boundary_margin < 0.0:
        raise ValueError("boundary margin must be non-negative")

    output_dir = osp.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    monotonicity = run_monotonicity_validation()
    sensitivity = run_weight_sensitivity(
        deltas=args.weight_deltas,
        boundary_margin=args.boundary_margin,
    )
    report = {
        "metadata": {
            "generated_at": datetime.now().astimezone().isoformat(),
            "scoring_version": RISK_SCORING_VERSION,
            "scope": "scoring-layer validation with controlled masks and detections",
            "note": (
                "This report evaluates formula behaviour. It does not measure "
                "BiSeNet/YOLO recognition accuracy or real fall probability."
            ),
        },
        "monotonicity": monotonicity,
        "weight_sensitivity": sensitivity,
    }

    report_path = osp.join(output_dir, "risk_scoring_validation.json")
    monotonicity_csv = osp.join(output_dir, "monotonicity_points.csv")
    sensitivity_csv = osp.join(output_dir, "weight_sensitivity.csv")
    _write_json(report_path, report)
    _write_monotonicity_csv(monotonicity_csv, monotonicity)
    _write_sensitivity_csv(sensitivity_csv, sensitivity)

    generated = [report_path, monotonicity_csv, sensitivity_csv]
    if not args.no_plots:
        monotonicity_plot = osp.join(output_dir, "monotonicity_curves.png")
        sensitivity_plot = osp.join(output_dir, "weight_sensitivity.png")
        position_montage = osp.join(output_dir, "obstacle_position_montage.png")
        _plot_monotonicity(monotonicity_plot, monotonicity)
        _plot_sensitivity(sensitivity_plot, sensitivity)
        _write_position_montage(position_montage)
        generated.extend((monotonicity_plot, sensitivity_plot, position_montage))

    print(
        "Monotonicity: {} ({} series, {} violations)".format(
            "PASS" if monotonicity["summary"]["passed"] else "FAIL",
            monotonicity["summary"]["series_count"],
            monotonicity["summary"]["violation_count"],
        )
    )
    print(
        "Weight sensitivity: {} ({} scenarios, {} variants, max |score delta|={:.2f}, "
        "minimum rank correlation={:.3f}, level flips={}, non-boundary flips={})".format(
            "PASS" if sensitivity["acceptance"]["passed"] else "REVIEW",
            sensitivity["summary"]["scenario_count"],
            sensitivity["summary"]["variant_count"],
            sensitivity["summary"]["maximum_absolute_score_delta"],
            sensitivity["summary"]["minimum_spearman_rank_correlation"],
            sensitivity["summary"]["total_level_flips"],
            sensitivity["summary"]["total_non_boundary_level_flips"],
        )
    )
    for path in generated:
        print("Saved: {}".format(path))
    passed = bool(
        monotonicity["summary"]["passed"]
        and sensitivity["acceptance"]["passed"]
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
