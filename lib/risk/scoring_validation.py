#!/usr/bin/env python3
"""Controlled validation utilities for the rule-based indoor-risk score.

The helpers in this module validate the scoring layer rather than model
accuracy.  They construct deterministic masks and YOLO detections, vary one
factor at a time, and then call the same ``analyze_bisenet`` function used by
the inference pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from lib.risk.bisenet_features import (
    CLASS_IDS,
    RISK_COMPONENT_MAX_POINTS,
    analyze_bisenet,
)
from lib.risk.yolo import YoloDetection


IMAGE_HEIGHT = 240
IMAGE_WIDTH = 320
RISK_LEVELS = ("low", "medium", "high", "emergency")
RISK_LEVEL_THRESHOLDS = (25.0, 50.0, 75.0)
DEFAULT_WEIGHT_DELTAS = (-0.20, -0.10, 0.10, 0.20)


@dataclass
class ControlledScene:
    """One deterministic scoring input."""

    name: str
    image: np.ndarray = field(repr=False)
    mask: np.ndarray = field(repr=False)
    detections: Optional[List[YoloDetection]] = field(default=None, repr=False)


@dataclass
class SweepPoint:
    """One labelled point in a monotonicity sweep."""

    value: float
    label: str
    scene: ControlledScene = field(repr=False)


@dataclass
class MonotonicSweep:
    """An ordered sequence from safer evidence to more dangerous evidence."""

    name: str
    description: str
    points: List[SweepPoint]
    tracked_features: Tuple[str, ...] = ()


def _blank_image(brightness: int = 180) -> np.ndarray:
    return np.full(
        (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        int(np.clip(brightness, 0, 255)),
        dtype=np.uint8,
    )


def _walkable_mask() -> np.ndarray:
    return np.full(
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        CLASS_IDS["walkable_surface"],
        dtype=np.uint8,
    )


def _yolo_box(
    xyxy: Tuple[int, int, int, int],
    *,
    class_name: str = "chair_stool",
    confidence: float = 1.0,
    class_id: int = 0,
) -> YoloDetection:
    return YoloDetection(
        xyxy=xyxy,
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
    )


def _centered_rect_mask(class_name: str, area_ratio: float) -> np.ndarray:
    mask = _walkable_mask()
    if area_ratio <= 0.0:
        return mask
    area_px = max(1, int(round(area_ratio * IMAGE_HEIGHT * IMAGE_WIDTH)))
    width = max(8, int(round(np.sqrt(area_px * 1.5))))
    height = max(8, int(round(area_px / float(width))))
    width = min(width, int(IMAGE_WIDTH * 0.45))
    height = min(height, int(IMAGE_HEIGHT * 0.35))
    x1 = IMAGE_WIDTH // 2 - width // 2
    y1 = int(IMAGE_HEIGHT * 0.68) - height // 2
    mask[y1 : y1 + height, x1 : x1 + width] = CLASS_IDS[class_name]
    return mask


def _passage_mask(width_ratio: float) -> np.ndarray:
    mask = np.full(
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        CLASS_IDS["wall_boundary"],
        dtype=np.uint8,
    )
    passage_width = max(8, int(round(IMAGE_WIDTH * width_ratio)))
    x1 = IMAGE_WIDTH // 2 - passage_width // 2
    mask[:, x1 : x1 + passage_width] = CLASS_IDS["walkable_surface"]
    return mask


def _count_detections(count: int) -> List[YoloDetection]:
    boxes = (
        (122, 120, 142, 145),
        (150, 120, 170, 145),
        (178, 120, 198, 145),
        (130, 170, 150, 195),
        (170, 170, 190, 195),
    )
    return [_yolo_box(box) for box in boxes[:count]]


def build_monotonic_sweeps() -> List[MonotonicSweep]:
    """Build controlled sweeps ordered from safer to more dangerous."""

    position_centres = (35, 65, 90, 120, 160)
    position_points: List[SweepPoint] = []
    for index, center_x in enumerate(position_centres):
        box = (center_x - 18, 165, center_x + 18, 215)
        scene = ControlledScene(
            name="position_{:02d}".format(index),
            image=_blank_image(),
            mask=_walkable_mask(),
            detections=[_yolo_box(box)],
        )
        position_points.append(
            SweepPoint(
                value=float(index),
                label="x={}".format(center_x),
                scene=scene,
            )
        )

    obstacle_sizes = (0, 12, 20, 30, 40, 55, 70)
    obstacle_area_points: List[SweepPoint] = []
    for size in obstacle_sizes:
        detections = []
        if size:
            detections = [
                _yolo_box(
                    (
                        IMAGE_WIDTH // 2 - size // 2,
                        175 - size // 2,
                        IMAGE_WIDTH // 2 + size // 2,
                        175 + size // 2,
                    )
                )
            ]
        obstacle_area_points.append(
            SweepPoint(
                value=float(size * size),
                label="{}px".format(size),
                scene=ControlledScene(
                    name="obstacle_size_{}".format(size),
                    image=_blank_image(),
                    mask=_walkable_mask(),
                    detections=detections,
                ),
            )
        )

    obstacle_count_points = [
        SweepPoint(
            value=float(count),
            label=str(count),
            scene=ControlledScene(
                name="obstacle_count_{}".format(count),
                image=_blank_image(),
                mask=_walkable_mask(),
                detections=_count_detections(count),
            ),
        )
        for count in range(6)
    ]

    passage_widths = (0.60, 0.30, 0.20, 0.15, 0.10)
    passage_points = [
        SweepPoint(
            value=float(1.0 - width_ratio),
            label="{:.0f}%".format(width_ratio * 100.0),
            scene=ControlledScene(
                name="passage_width_{:.2f}".format(width_ratio),
                image=_blank_image(),
                mask=_passage_mask(width_ratio),
                detections=[],
            ),
        )
        for width_ratio in passage_widths
    ]

    hazard_ratios = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08)
    slippery_points = [
        SweepPoint(
            value=float(ratio),
            label="{:.2f}%".format(ratio * 100.0),
            scene=ControlledScene(
                name="slippery_{:.4f}".format(ratio),
                image=_blank_image(),
                mask=_centered_rect_mask("slippery_surface", ratio),
                detections=[],
            ),
        )
        for ratio in hazard_ratios
    ]
    step_points = [
        SweepPoint(
            value=float(ratio),
            label="{:.2f}%".format(ratio * 100.0),
            scene=ControlledScene(
                name="step_{:.4f}".format(ratio),
                image=_blank_image(),
                mask=_centered_rect_mask("step_threshold", ratio),
                detections=[],
            ),
        )
        for ratio in hazard_ratios
    ]

    brightness_values = (220, 180, 140, 100, 70, 40, 20)
    brightness_points = [
        SweepPoint(
            value=float(255 - brightness),
            label=str(brightness),
            scene=ControlledScene(
                name="brightness_{}".format(brightness),
                image=_blank_image(brightness),
                mask=_walkable_mask(),
                detections=[],
            ),
        )
        for brightness in brightness_values
    ]

    confidence_values = (0.25, 0.40, 0.55, 0.70, 0.85, 1.0)
    confidence_points = [
        SweepPoint(
            value=float(confidence),
            label="{:.2f}".format(confidence),
            scene=ControlledScene(
                name="liquid_confidence_{:.2f}".format(confidence),
                image=_blank_image(),
                mask=_walkable_mask(),
                detections=[
                    _yolo_box(
                        (142, 160, 178, 210),
                        class_name="liquid_spot",
                        confidence=confidence,
                        class_id=15,
                    )
                ],
            ),
        )
        for confidence in confidence_values
    ]

    return [
        MonotonicSweep(
            name="obstacle_toward_corridor_center",
            description="The same YOLO box moves from outside the passage toward its centre.",
            points=position_points,
            tracked_features=("yolo_corridor_occupancy",),
        ),
        MonotonicSweep(
            name="obstacle_area",
            description="One centred obstacle grows while all other evidence is fixed.",
            points=obstacle_area_points,
            tracked_features=("yolo_corridor_occupancy",),
        ),
        MonotonicSweep(
            name="obstacle_count",
            description="The number of non-overlapping corridor obstacles increases.",
            points=obstacle_count_points,
            tracked_features=("yolo_corridor_detection_count",),
        ),
        MonotonicSweep(
            name="passage_narrowing",
            description="The visible walkable passage becomes narrower.",
            points=passage_points,
            tracked_features=("narrowest_passage_width_ratio",),
        ),
        MonotonicSweep(
            name="slippery_area",
            description="The slippery mask area inside the passage increases.",
            points=slippery_points,
            tracked_features=("corridor_slippery_ratio",),
        ),
        MonotonicSweep(
            name="step_area",
            description="The step or threshold mask area increases.",
            points=step_points,
            tracked_features=("step_threshold_area_ratio",),
        ),
        MonotonicSweep(
            name="darkness",
            description="Uniform image brightness decreases from 220 to 20.",
            points=brightness_points,
            tracked_features=("low_light_score",),
        ),
        MonotonicSweep(
            name="liquid_detection_confidence",
            description="The confidence of a fixed liquid-spot detection increases.",
            points=confidence_points,
            tracked_features=("yolo_liquid_spot_confidence",),
        ),
    ]


def _point_record(point: SweepPoint, analysis: Any, tracked: Sequence[str]) -> Dict[str, Any]:
    return {
        "value": float(point.value),
        "label": point.label,
        "score": float(analysis.risk.score),
        "level": analysis.risk.level,
        "features": {
            key: analysis.features.get(key)
            for key in tracked
        },
    }


def run_monotonicity_validation(tolerance: float = 1e-6) -> Dict[str, Any]:
    """Run every safer-to-more-dangerous sweep and report score decreases."""

    series_records: List[Dict[str, Any]] = []
    total_violations = 0
    for sweep in build_monotonic_sweeps():
        points: List[Dict[str, Any]] = []
        for point in sweep.points:
            analysis = analyze_bisenet(
                point.scene.image,
                point.scene.mask,
                yolo_detections=point.scene.detections,
            )
            points.append(_point_record(point, analysis, sweep.tracked_features))

        violations: List[Dict[str, Any]] = []
        plateaus = 0
        for index in range(1, len(points)):
            previous = float(points[index - 1]["score"])
            current = float(points[index]["score"])
            if current + tolerance < previous:
                violations.append(
                    {
                        "from_label": points[index - 1]["label"],
                        "to_label": points[index]["label"],
                        "from_score": previous,
                        "to_score": current,
                        "decrease": float(previous - current),
                    }
                )
            elif abs(current - previous) <= tolerance:
                plateaus += 1
        total_violations += len(violations)
        series_records.append(
            {
                "name": sweep.name,
                "description": sweep.description,
                "expected": "non_decreasing",
                "passed": not violations,
                "plateau_transition_count": int(plateaus),
                "points": points,
                "violations": violations,
            }
        )

    return {
        "summary": {
            "passed": total_violations == 0,
            "series_count": len(series_records),
            "violation_count": int(total_violations),
        },
        "series": series_records,
    }


def build_sensitivity_scenes() -> List[ControlledScene]:
    """Return a compact panel spanning the major scoring components."""

    combined_mask = _centered_rect_mask("slippery_surface", 0.025)
    combined_mask[105:119, 135:185] = CLASS_IDS["step_threshold"]
    near_fifty_mask = _walkable_mask()
    near_fifty_mask[90:115, 120:200] = CLASS_IDS["step_threshold"]
    near_fifty_mask[125:160, 110:210] = CLASS_IDS["slippery_surface"]
    severe_step_mask = _walkable_mask()
    severe_step_mask[90:130, 100:220] = CLASS_IDS["step_threshold"]
    near_fifty_detections = _count_detections(5)
    near_fifty_detections.append(
        _yolo_box(
            (135, 175, 185, 200),
            class_name="electric_wire_power",
            confidence=0.90,
            class_id=1,
        )
    )
    return [
        ControlledScene("clean", _blank_image(), _walkable_mask(), []),
        ControlledScene(
            "one_obstacle",
            _blank_image(),
            _walkable_mask(),
            [_yolo_box((142, 160, 178, 210))],
        ),
        ControlledScene(
            "three_obstacles",
            _blank_image(),
            _walkable_mask(),
            _count_detections(3),
        ),
        ControlledScene(
            "electric_wire",
            _blank_image(),
            _walkable_mask(),
            [
                _yolo_box(
                    (135, 170, 185, 195),
                    class_name="electric_wire_power",
                    confidence=0.85,
                    class_id=1,
                )
            ],
        ),
        ControlledScene(
            "narrow_passage",
            _blank_image(),
            _passage_mask(0.12),
            [],
        ),
        ControlledScene(
            "slippery_surface",
            _blank_image(),
            _centered_rect_mask("slippery_surface", 0.04),
            [],
        ),
        ControlledScene(
            "step_threshold",
            _blank_image(),
            _centered_rect_mask("step_threshold", 0.025),
            [],
        ),
        ControlledScene("low_light", _blank_image(35), _walkable_mask(), []),
        ControlledScene(
            "combined_risk",
            _blank_image(55),
            combined_mask,
            [
                _yolo_box(
                    (132, 170, 188, 198),
                    class_name="electric_wire_power",
                    confidence=0.90,
                    class_id=1,
                )
            ],
        ),
        ControlledScene(
            "near_25_boundary",
            _blank_image(),
            _walkable_mask(),
            [
                _yolo_box(
                    (142, 160, 178, 210),
                    class_name="liquid_spot",
                    confidence=0.45,
                    class_id=15,
                )
            ],
        ),
        ControlledScene(
            "near_50_boundary",
            _blank_image(35),
            near_fifty_mask,
            near_fifty_detections,
        ),
        ControlledScene(
            "high_combined",
            _blank_image(),
            severe_step_mask.copy(),
            _count_detections(3)
            + [
                _yolo_box(
                    (140, 155, 180, 210),
                    class_name="liquid_spot",
                    confidence=0.60,
                    class_id=15,
                )
            ],
        ),
        ControlledScene(
            "near_75_boundary",
            _blank_image(70),
            severe_step_mask.copy(),
            _count_detections(5)
            + [
                _yolo_box(
                    (140, 155, 180, 210),
                    class_name="liquid_spot",
                    confidence=0.80,
                    class_id=15,
                )
            ],
        ),
        ControlledScene(
            "emergency_combined",
            _blank_image(20),
            severe_step_mask.copy(),
            _count_detections(5)
            + [
                _yolo_box(
                    (140, 155, 180, 210),
                    class_name="liquid_spot",
                    confidence=1.0,
                    class_id=15,
                )
            ],
        ),
    ]


def perturb_component_weights(
    base_weights: Mapping[str, float],
    component: str,
    fractional_delta: float,
) -> Dict[str, float]:
    """Change one component and rescale the others to preserve the total."""

    if component not in base_weights:
        raise KeyError("unknown risk component: {}".format(component))
    weights = {key: float(value) for key, value in base_weights.items()}
    if any(not np.isfinite(value) or value < 0.0 for value in weights.values()):
        raise ValueError("base weights must be finite and non-negative")
    total = float(sum(weights.values()))
    old_value = weights[component]
    new_value = old_value * (1.0 + float(fractional_delta))
    other_total = total - old_value
    if new_value < 0.0 or new_value > total or other_total <= 0.0:
        raise ValueError("weight perturbation is outside the valid range")
    scale = (total - new_value) / other_total
    for key in weights:
        weights[key] = new_value if key == component else weights[key] * scale
    return weights


def _distance_to_level_boundary(score: float) -> float:
    return float(min(abs(score - boundary) for boundary in RISK_LEVEL_THRESHOLDS))


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    values_array = np.asarray(values, dtype=np.float64)
    order = np.argsort(values_array, kind="mergesort")
    ranks = np.empty(values_array.size, dtype=np.float64)
    start = 0
    while start < order.size:
        end = start + 1
        while end < order.size and values_array[order[end]] == values_array[order[start]]:
            end += 1
        average_rank = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _spearman_rank_correlation(first: Sequence[float], second: Sequence[float]) -> float:
    first_ranks = _average_ranks(first)
    second_ranks = _average_ranks(second)
    first_std = float(first_ranks.std())
    second_std = float(second_ranks.std())
    if first_std == 0.0 or second_std == 0.0:
        return 1.0 if np.array_equal(first_ranks, second_ranks) else 0.0
    return float(np.corrcoef(first_ranks, second_ranks)[0, 1])


def run_weight_sensitivity(
    deltas: Sequence[float] = DEFAULT_WEIGHT_DELTAS,
    boundary_margin: float = 3.0,
) -> Dict[str, Any]:
    """Perturb each top-level component weight and compare scores and ranks."""

    scenes = build_sensitivity_scenes()
    base_weights = dict(RISK_COMPONENT_MAX_POINTS)
    baseline: Dict[str, Dict[str, Any]] = {}
    for scene in scenes:
        analysis = analyze_bisenet(
            scene.image,
            scene.mask,
            yolo_detections=scene.detections,
            risk_component_max_points=base_weights,
        )
        baseline[scene.name] = {
            "score": float(analysis.risk.score),
            "level": analysis.risk.level,
            "distance_to_level_boundary": _distance_to_level_boundary(
                analysis.risk.score
            ),
        }

    variants: List[Dict[str, Any]] = []
    baseline_scores = [baseline[scene.name]["score"] for scene in scenes]
    for component in base_weights:
        for delta in deltas:
            weights = perturb_component_weights(base_weights, component, float(delta))
            rows: List[Dict[str, Any]] = []
            variant_scores: List[float] = []
            for scene in scenes:
                analysis = analyze_bisenet(
                    scene.image,
                    scene.mask,
                    yolo_detections=scene.detections,
                    risk_component_max_points=weights,
                )
                base = baseline[scene.name]
                score = float(analysis.risk.score)
                level_changed = analysis.risk.level != base["level"]
                level_jump = abs(
                    RISK_LEVELS.index(analysis.risk.level)
                    - RISK_LEVELS.index(str(base["level"]))
                )
                rows.append(
                    {
                        "scenario": scene.name,
                        "baseline_score": float(base["score"]),
                        "variant_score": score,
                        "score_delta": float(score - float(base["score"])),
                        "absolute_delta": float(abs(score - float(base["score"]))),
                        "baseline_level": base["level"],
                        "variant_level": analysis.risk.level,
                        "distance_to_level_boundary": float(
                            base["distance_to_level_boundary"]
                        ),
                        "level_changed": bool(level_changed),
                        "non_boundary_level_changed": bool(
                            level_changed
                            and float(base["distance_to_level_boundary"])
                            > boundary_margin
                        ),
                        "level_jump": int(level_jump),
                    }
                )
                variant_scores.append(score)

            absolute_deltas = [float(row["absolute_delta"]) for row in rows]
            variants.append(
                {
                    "component": component,
                    "fractional_delta": float(delta),
                    "weights": weights,
                    "mean_absolute_score_delta": float(np.mean(absolute_deltas)),
                    "max_absolute_score_delta": float(max(absolute_deltas)),
                    "level_flip_count": int(
                        sum(bool(row["level_changed"]) for row in rows)
                    ),
                    "non_boundary_level_flip_count": int(
                        sum(bool(row["non_boundary_level_changed"]) for row in rows)
                    ),
                    "two_level_jump_count": int(
                        sum(int(row["level_jump"]) >= 2 for row in rows)
                    ),
                    "spearman_rank_correlation": _spearman_rank_correlation(
                        baseline_scores, variant_scores
                    ),
                    "scenarios": rows,
                }
            )

    summary = {
        "scenario_count": len(scenes),
        "variant_count": len(variants),
        "boundary_margin": float(boundary_margin),
        "maximum_absolute_score_delta": float(
            max(item["max_absolute_score_delta"] for item in variants)
        ),
        "minimum_spearman_rank_correlation": float(
            min(item["spearman_rank_correlation"] for item in variants)
        ),
        "total_level_flips": int(
            sum(item["level_flip_count"] for item in variants)
        ),
        "total_non_boundary_level_flips": int(
            sum(item["non_boundary_level_flip_count"] for item in variants)
        ),
        "total_two_level_jumps": int(
            sum(item["two_level_jump_count"] for item in variants)
        ),
    }
    criteria = {
        "maximum_absolute_score_delta_at_most_5": bool(
            summary["maximum_absolute_score_delta"] <= 5.0
        ),
        "minimum_rank_correlation_at_least_0_95": bool(
            summary["minimum_spearman_rank_correlation"] >= 0.95
        ),
        "no_non_boundary_level_flips": bool(
            summary["total_non_boundary_level_flips"] == 0
        ),
        "no_two_level_jumps": bool(summary["total_two_level_jumps"] == 0),
    }

    return {
        "summary": summary,
        "acceptance": {
            "passed": all(criteria.values()),
            "criteria": criteria,
            "note": "Internal engineering thresholds, not clinical validation.",
        },
        "base_weights": base_weights,
        "baseline": baseline,
        "variants": variants,
    }
