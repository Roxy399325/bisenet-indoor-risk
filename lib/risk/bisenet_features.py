#!/usr/bin/env python3
"""Extract and fuse interpretable indoor-risk features from one image."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


CLASS_IDS = {
    "walkable_surface": 0,
    "wall_boundary": 1,
    "non_walkable": 2,
    "obstacle": 3,
    "slippery_surface": 4,
    "step_threshold": 5,
}
CLASS_NAMES = [
    "walkable_surface",
    "wall_boundary",
    "non_walkable",
    "obstacle",
    "slippery_surface",
    "step_threshold",
]
IGNORE_ID = 255

CLASS_COLORS = np.array(
    [
        (75, 180, 70),
        (210, 120, 60),
        (95, 95, 95),
        (55, 55, 220),
        (50, 220, 240),
        (35, 140, 255),
    ],
    dtype=np.uint8,
)

_VALID_IDS = frozenset(range(len(CLASS_NAMES)))
_DANGER_CLASSES = (
    CLASS_IDS["walkable_surface"],
    CLASS_IDS["obstacle"],
    CLASS_IDS["slippery_surface"],
    CLASS_IDS["step_threshold"],
)

# These values follow the YOLO handoff package's ``config/default.json``.
# They are risk points (higher means more dangerous), not safety deductions.
# ``liquid_spot`` is intentionally absent in V2 because it is fused into the
# slippery component instead of being counted again as a passage category.
YOLO_CLASS_RISK_WEIGHTS = {
    "shoes_slippers": 8.0,
    "electric_wire_power": 12.0,
    "electric_wire": 12.0,
    "rug_mat_carpet": 10.0,
    "rug_edge": 10.0,
    "toy": 8.0,
    "door_stairs_threshold": 10.0,
    "threshold": 10.0,
}
YOLO_MIN_CORRIDOR_OVERLAP = 0.05

# V2 groups three correlated passage signals under one cap so a single item of
# furniture cannot independently saturate obstacle, occupancy, and width risk.
RISK_SCORING_VERSION = "v2_evidence_aware"
RISK_COMPONENT_MAX_POINTS = {
    "passage_obstruction": 45.0,
    "steps_thresholds": 25.0,
    "slippery_surface": 20.0,
    "low_light": 10.0,
}
PASSAGE_EVIDENCE_WEIGHTS = {
    "obstacles": 0.35,
    "corridor_occupancy": 0.40,
    "narrow_passage": 0.25,
}
CARPET_SLIPPERY_RELIABILITY_FACTOR = 0.25
_CARPET_YOLO_CLASSES = frozenset(("rug_mat_carpet", "rug_edge"))
_LIQUID_YOLO_CLASSES = frozenset(("liquid_spot",))


def _json_number(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass
class RiskResult:
    """Rule-based risk output."""

    score: float
    level: str
    scoring_version: str = RISK_SCORING_VERSION
    reasons: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    component_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": float(self.score),
            "level": self.level,
            "scoring_version": self.scoring_version,
            "reasons": list(self.reasons),
            "suggestions": list(self.suggestions),
            "component_scores": {
                key: float(value) for key, value in self.component_scores.items()
            },
        }


@dataclass
class BisenetAnalysis:
    """Analysis result plus arrays used by visualisation."""

    image_size: Tuple[int, int]
    mask_valid: bool
    features: Dict[str, Any]
    risk: RiskResult
    quality: Dict[str, Any]
    available_features: List[str]
    unavailable_features: List[str]
    cleaned_mask: np.ndarray = field(repr=False)
    corridor_mask: np.ndarray = field(repr=False)
    obstacle_mask: np.ndarray = field(repr=False)
    obstacle_instances: List[Dict[str, Any]] = field(default_factory=list)
    yolo_detections: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "mask_valid": bool(self.mask_valid),
            "features": {
                key: _json_number(value) for key, value in self.features.items()
            },
            "risk": self.risk.to_dict(),
            "quality": {
                key: _json_number(value) for key, value in self.quality.items()
            },
            "available_features": list(self.available_features),
            "unavailable_features": list(self.unavailable_features),
            "obstacles": [
                {key: _json_number(value) for key, value in instance.items()}
                for instance in self.obstacle_instances
            ],
            "yolo_detections": [dict(item) for item in self.yolo_detections],
        }


def _validate_inputs(image: np.ndarray, mask: np.ndarray) -> Tuple[int, int]:
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a HxWx3 numpy array")
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError("mask must be a HxW numpy array")
    if image.shape[:2] != mask.shape:
        raise ValueError(
            "image and mask sizes do not match: {} vs {}".format(
                image.shape[:2], mask.shape
            )
        )
    if not np.issubdtype(mask.dtype, np.integer):
        raise ValueError("mask must contain integer class ids")
    allowed = np.array(sorted(_VALID_IDS | {IGNORE_ID}))
    unknown = np.setdiff1d(np.unique(mask), allowed)
    if unknown.size:
        raise ValueError("mask contains unknown class ids: {}".format(unknown.tolist()))
    return int(mask.shape[0]), int(mask.shape[1])


def _minimum_component_area(height: int, width: int) -> int:
    return max(64, int(round(0.0002 * height * width)))


def _clean_class_mask(mask: np.ndarray, class_id: int, min_area: int) -> np.ndarray:
    """Clean one class without changing labels of other classes."""

    binary = (mask == class_id).astype(np.uint8)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    cleaned = np.zeros(binary.shape, dtype=bool)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            cleaned[labels == label] = True
    return cleaned


def _components(binary: np.ndarray, min_area: int) -> List[Dict[str, Any]]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), 8
    )
    components: List[Dict[str, Any]] = []
    height, width = binary.shape
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        box_width = int(stats[label, cv2.CC_STAT_WIDTH])
        box_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        component_mask = labels == label
        contours, _ = cv2.findContours(
            component_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contour_points: List[List[int]] = []
        if contours:
            contour = max(contours, key=cv2.contourArea)
            epsilon = max(1.0, 0.005 * cv2.arcLength(contour, True))
            contour = cv2.approxPolyDP(contour, epsilon, True)
            contour_points = [
                [int(point[0][0]), int(point[0][1])] for point in contour
            ]
        components.append(
            {
                "id": int(len(components) + 1),
                "area_px": area,
                "area_ratio": float(area / float(height * width)),
                "bbox": [x, y, box_width, box_height],
                "centroid": [
                    float(centroids[label][0]),
                    float(centroids[label][1]),
                ],
                "bottom_frame_ratio": float(
                    component_mask[int(height * 0.8) :, :].sum() / float(area)
                ),
                "contour": contour_points,
            }
        )
    return components


def _annotate_corridor_overlap(
    components: List[Dict[str, Any]],
    corridor: np.ndarray,
    corridor_valid: bool,
) -> int:
    """Attach per-region corridor overlap and return the in-corridor count."""

    in_corridor_count = 0
    for component in components:
        if not corridor_valid:
            component["corridor_overlap_px"] = None
            component["corridor_overlap_ratio"] = None
            component["in_corridor"] = None
            continue

        x, y, box_width, box_height = component["bbox"]
        local_mask = np.zeros((box_height, box_width), dtype=np.uint8)
        contour = np.asarray(component["contour"], dtype=np.int32)
        if contour.size:
            contour = contour - np.array([x, y], dtype=np.int32)
            cv2.fillPoly(local_mask, [contour], 1)
        overlap = int(
            np.count_nonzero(
                local_mask.astype(bool)
                & corridor[y : y + box_height, x : x + box_width]
            )
        )
        overlap_ratio = overlap / float(max(int(component["area_px"]), 1))
        in_corridor = overlap >= max(4, int(round(component["area_px"] * 0.05)))
        component["corridor_overlap_px"] = overlap
        component["corridor_overlap_ratio"] = float(overlap_ratio)
        component["in_corridor"] = bool(in_corridor)
        if in_corridor:
            in_corridor_count += 1
    return in_corridor_count


def _row_intervals(row: np.ndarray) -> List[Tuple[int, int]]:
    indices = np.flatnonzero(row)
    if indices.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[indices[0], indices[breaks + 1]]
    ends = np.r_[indices[breaks], indices[-1]]
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _central_interval(
    intervals: Sequence[Tuple[int, int]], center: float
) -> Optional[Tuple[int, int]]:
    if not intervals:
        return None
    return min(
        intervals,
        key=lambda interval: abs((interval[0] + interval[1]) / 2.0 - center),
    )


def _perspective_envelope(height: int, width: int) -> np.ndarray:
    """Return the default 20%-top/60%-bottom perspective corridor."""

    polygon = np.array(
        [
            [int(width * 0.40), int(height * 0.35)],
            [int(width * 0.60), int(height * 0.35)],
            [int(width * 0.80), height - 1],
            [int(width * 0.20), height - 1],
        ],
        dtype=np.int32,
    )
    envelope = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(envelope, [polygon], 1)
    return envelope.astype(bool)


def _build_corridor(
    walkable: np.ndarray,
    candidate: np.ndarray,
) -> Tuple[np.ndarray, bool, bool]:
    """Build a perspective-limited corridor from bottom-centre walkable pixels."""

    height, width = candidate.shape
    default_envelope = _perspective_envelope(height, width)
    fallback = default_envelope & candidate
    bottom_band_start = max(0, int(height * 0.90))
    seed: Optional[Tuple[int, int]] = None
    center_x = width / 2.0
    center_left = int(width * 0.40)
    center_right = max(center_left + 1, int(width * 0.60))

    for y in range(height - 1, bottom_band_start - 1, -1):
        center_pixels = np.flatnonzero(walkable[y, center_left:center_right])
        if center_pixels.size:
            xs = center_pixels + center_left
            x = int(xs[np.argmin(np.abs(xs - center_x))])
            seed = (x, y)
            break

    if seed is None:
        return fallback, False, True

    _, labels, _, _ = cv2.connectedComponentsWithStats(
        walkable.astype(np.uint8), 8
    )
    component_label = int(labels[seed[1], seed[0]])
    if component_label <= 0:
        return fallback, False, True

    component = labels == component_label
    corridor = np.zeros_like(candidate, dtype=bool)
    previous_center = center_x
    top_y = int(height * 0.35)
    rows_with_candidate = 0
    for y in range(height - 1, top_y - 1, -1):
        interval = _central_interval(_row_intervals(component[y]), previous_center)
        if interval is not None:
            observed_center = (interval[0] + interval[1]) / 2.0
            previous_center = 0.75 * previous_center + 0.25 * observed_center

        progress = (y - top_y) / float(max(height - 1 - top_y, 1))
        corridor_width = width * (0.20 + 0.40 * progress)
        left = max(0, int(round(previous_center - corridor_width / 2.0)))
        right = min(width, int(round(previous_center + corridor_width / 2.0)))
        if right > left:
            corridor[y, left:right] = candidate[y, left:right]
            if np.any(corridor[y]):
                rows_with_candidate += 1

    bottom_contact = bool(
        np.any(corridor[bottom_band_start:, center_left:center_right])
    )
    row_coverage = rows_with_candidate / float(max(height - top_y, 1))
    corridor_valid = bottom_contact and row_coverage >= 0.25
    return corridor, bool(corridor_valid), False


def _max_run_length(row: np.ndarray) -> int:
    intervals = _row_intervals(row)
    return max((end - start + 1 for start, end in intervals), default=0)


def _low_light_features(
    image: np.ndarray,
    roi: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if roi is None or roi.shape != gray.shape or int(np.count_nonzero(roi)) < 64:
        roi = np.zeros(gray.shape, dtype=bool)
        roi[int(gray.shape[0] * 0.40) :, :] = True
    pixels = gray[roi]
    mean_gray = float(pixels.mean())
    p10_gray = float(np.percentile(pixels, 10))
    dark_pixel_ratio = float(np.mean(pixels < 50))
    mean_risk = _clip01((100.0 - mean_gray) / 70.0)
    dark_area_risk = _clip01((dark_pixel_ratio - 0.15) / 0.50)
    score = float(0.70 * mean_risk + 0.30 * dark_area_risk)
    return {
        "gray_mean": mean_gray,
        "gray_p10": p10_gray,
        "low_light_dark_pixel_ratio": dark_pixel_ratio,
        "low_light_roi_ratio": float(np.count_nonzero(roi) / roi.size),
        "low_light_score": score,
        "low_light_flag": bool(score >= 0.50),
    }


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _risk_level(score: float) -> str:
    if score >= 75.0:
        return "emergency"
    if score >= 50.0:
        return "high"
    if score >= 25.0:
        return "medium"
    return "low"


def _detection_value(detection: Any, name: str) -> Any:
    if isinstance(detection, Mapping):
        if name == "xyxy" and name not in detection:
            return detection.get("bbox_xyxy")
        return detection.get(name)
    return getattr(detection, name, None)


def _normalize_yolo_detections(
    detections: Iterable[Any],
    shape: Tuple[int, int],
    corridor: np.ndarray,
    corridor_valid: bool,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Clip YOLO boxes and attach their overlap with the BiSeNet corridor."""

    height, width = shape
    records: List[Dict[str, Any]] = []
    occupied = np.zeros(shape, dtype=bool)
    for detection in detections:
        coords = _detection_value(detection, "xyxy")
        if coords is None or len(coords) != 4:
            raise ValueError("each YOLO detection must provide four xyxy coordinates")
        try:
            values = [float(value) for value in coords]
        except (TypeError, ValueError) as error:
            raise ValueError("YOLO xyxy coordinates must be numeric") from error
        if not np.all(np.isfinite(values)):
            raise ValueError("YOLO xyxy coordinates must be finite")
        x1, x2 = sorted((int(round(values[0])), int(round(values[2]))))
        y1, y2 = sorted((int(round(values[1])), int(round(values[3]))))
        x1, x2 = int(np.clip(x1, 0, width)), int(np.clip(x2, 0, width))
        y1, y2 = int(np.clip(y1, 0, height)), int(np.clip(y2, 0, height))
        if x2 <= x1 or y2 <= y1:
            continue

        confidence = float(_detection_value(detection, "confidence"))
        if not np.isfinite(confidence):
            raise ValueError("YOLO confidence must be finite")
        confidence = float(np.clip(confidence, 0.0, 1.0))
        class_id = int(_detection_value(detection, "class_id"))
        class_name = str(_detection_value(detection, "class_name"))
        box_area = float((x2 - x1) * (y2 - y1))

        if corridor_valid:
            local_corridor = corridor[y1:y2, x1:x2]
            overlap_pixels = int(local_corridor.sum())
            overlap_ratio: Optional[float] = overlap_pixels / box_area
            in_corridor: Optional[bool] = (
                overlap_ratio >= YOLO_MIN_CORRIDOR_OVERLAP
            )
            if in_corridor:
                occupied[y1:y2, x1:x2] |= local_corridor
        else:
            overlap_ratio = None
            in_corridor = None

        records.append(
            {
                "bbox_xyxy": [x1, y1, x2, y2],
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "corridor_overlap_ratio": overlap_ratio,
                "in_corridor": in_corridor,
            }
        )
    return records, occupied


def _base_yolo_class_name(detection: Mapping[str, Any]) -> str:
    return str(detection["class_name"]).split(" (", 1)[0]


def _yolo_category_risk_points(
    detections: Sequence[Dict[str, Any]],
) -> float:
    points = 0.0
    for detection in detections:
        if detection["in_corridor"] is not True:
            continue
        class_name = _base_yolo_class_name(detection)
        points += (
            YOLO_CLASS_RISK_WEIGHTS.get(class_name, 0.0)
            * float(detection["confidence"])
        )
    return float(min(25.0, points))


def _score_risk(
    features: Dict[str, Any],
    mask_valid: bool,
    yolo_detections: Sequence[Dict[str, Any]],
) -> RiskResult:
    bisenet_obstacle_count = float(features.get("obstacle_count", 0) or 0)
    yolo_obstacle_count = float(features.get("yolo_corridor_detection_count", 0) or 0)
    obstacle_count = max(bisenet_obstacle_count, yolo_obstacle_count)
    bisenet_obstacle_ratio = float(features.get("obstacle_area_ratio", 0.0) or 0.0)
    yolo_obstacle_ratio = float(features.get("yolo_corridor_occupancy", 0.0) or 0.0)
    obstacle_ratio = max(bisenet_obstacle_ratio, yolo_obstacle_ratio)
    obstacle_quantity = (
        0.5 * _clip01(obstacle_count / 5.0)
        + 0.5 * _clip01(obstacle_ratio / 0.25)
    )
    generic_obstacle_score = 25.0 * obstacle_quantity

    # A category-specific score strengthens the generic obstacle evidence but
    # does not stack on top of it, which avoids double-counting one object seen
    # by both BiSeNet and YOLO.
    category_score = float(features.get("yolo_category_risk_points", 0.0) or 0.0)
    obstacle_score = max(generic_obstacle_score, category_score)

    corridor_valid = bool(features.get("corridor_valid", False))
    corridor_occupancy = max(
        float(features.get("corridor_obstacle_occupancy", 0.0) or 0.0),
        float(features.get("yolo_corridor_occupancy", 0.0) or 0.0),
    )
    corridor_score = 20.0 * _clip01(corridor_occupancy / 0.35) if corridor_valid else 0.0

    width_ratio = features.get("narrowest_passage_width_ratio")
    if width_ratio is None or not corridor_valid:
        width_score = 0.0
    else:
        width_score = 20.0 * _clip01((0.40 - float(width_ratio)) / 0.30)

    passage_evidence = (
        PASSAGE_EVIDENCE_WEIGHTS["obstacles"] * _clip01(obstacle_score / 25.0)
        + PASSAGE_EVIDENCE_WEIGHTS["corridor_occupancy"]
        * _clip01(corridor_score / 20.0)
        + PASSAGE_EVIDENCE_WEIGHTS["narrow_passage"]
        * _clip01(width_score / 20.0)
    )
    passage_score = (
        RISK_COMPONENT_MAX_POINTS["passage_obstruction"]
        * _clip01(passage_evidence)
    )

    step_count = float(features.get("step_threshold_count", 0) or 0)
    step_ratio = float(features.get("step_threshold_area_ratio", 0.0) or 0.0)
    step_evidence = max(
        _clip01(step_count / 3.0),
        _clip01(step_ratio / 0.05),
    )
    step_score = RISK_COMPONENT_MAX_POINTS["steps_thresholds"] * step_evidence

    slippery_ratio = float(features.get("corridor_slippery_ratio", 0.0) or 0.0)
    raw_slippery_evidence = _clip01(
        float(features.get("slippery_surface_score", 0.0) or 0.0)
    )
    corridor_classes = {
        _base_yolo_class_name(item)
        for item in yolo_detections
        if item["in_corridor"] is True
    }
    carpet_present = bool(corridor_classes & _CARPET_YOLO_CLASSES)
    liquid_confidence = max(
        (
            float(item["confidence"])
            for item in yolo_detections
            if item["in_corridor"] is True
            and _base_yolo_class_name(item) in _LIQUID_YOLO_CLASSES
        ),
        default=0.0,
    )
    carpet_discount_applied = carpet_present and liquid_confidence <= 0.0
    slippery_reliability_factor = (
        CARPET_SLIPPERY_RELIABILITY_FACTOR
        if carpet_discount_applied
        else 1.0
    )
    slippery_evidence = max(raw_slippery_evidence, _clip01(liquid_confidence))
    slippery_evidence *= slippery_reliability_factor
    slippery_score = (
        RISK_COMPONENT_MAX_POINTS["slippery_surface"]
        * _clip01(slippery_evidence)
    )
    low_light_evidence = _clip01(
        float(features.get("low_light_score", 0.0) or 0.0)
    )
    low_light_score = RISK_COMPONENT_MAX_POINTS["low_light"] * low_light_evidence

    features.update(
        {
            "scoring_version": RISK_SCORING_VERSION,
            "passage_obstruction_evidence": float(passage_evidence),
            "slippery_surface_raw_evidence": float(raw_slippery_evidence),
            "slippery_surface_reliability_factor": float(
                slippery_reliability_factor
            ),
            "slippery_surface_carpet_discount_applied": bool(
                carpet_discount_applied
            ),
            "yolo_liquid_spot_confidence": float(liquid_confidence),
        }
    )

    component_scores = {
        "passage_obstruction": passage_score,
        "steps_thresholds": step_score,
        "slippery_surface": slippery_score,
        "low_light": low_light_score,
    }
    raw_score = float(sum(component_scores.values()))
    observability_penalty = (
        max(0.0, 25.0 - raw_score)
        if not corridor_valid or not mask_valid
        else 0.0
    )
    component_scores["observability_penalty"] = observability_penalty
    score = float(round(min(100.0, sum(component_scores.values())), 2))
    reasons: List[str] = []
    suggestions: List[str] = []

    if obstacle_count > 0:
        reasons.append("检测到{}个疑似通行障碍物。".format(int(obstacle_count)))
        suggestions.append("清理或移除通道及活动区域内的障碍物。")
    corridor_yolo = [
        item for item in yolo_detections if item["in_corridor"] is True
    ]
    if corridor_yolo:
        category_counts: Dict[str, int] = {}
        for item in corridor_yolo:
            name = str(item["class_name"])
            category_counts[name] = category_counts.get(name, 0) + 1
        summary = "、".join(
            "{}×{}".format(name, count)
            for name, count in sorted(category_counts.items())
        )
        reasons.append("YOLO在通道代理区域检测到：{}。".format(summary))
        suggestions.append("优先清理或固定YOLO识别出的具体障碍物。")
    if corridor_valid and corridor_occupancy >= 0.05:
        reasons.append("通道代理区域存在障碍物占用。")
    if width_ratio is not None and corridor_valid and float(width_ratio) < 0.25:
        reasons.append("通道代理区域的相对最窄宽度较小。")
        suggestions.append("扩大通行空间，避免家具或杂物压缩通道。")
    if step_count > 0:
        reasons.append("检测到{}个疑似台阶或门槛区域，结果仅供提示。".format(int(step_count)))
        suggestions.append("检查台阶或门槛，并增加醒目标识、扶手或坡道。")
    if corridor_valid and (slippery_ratio >= 0.02 or liquid_confidence > 0.0):
        if carpet_discount_applied:
            reasons.append(
                "检测到疑似湿滑区域，但与通道内地毯证据同时出现，"
                "已降低该项评分可信度。"
            )
            suggestions.append("检查地毯是否平整、固定和干燥，并复核周边地面。")
        else:
            reasons.append("通道代理区域存在疑似湿滑表面。")
            suggestions.append("及时清洁并干燥地面，必要时增加防滑措施。")
    if bool(features.get("low_light_flag", False)):
        reasons.append("图像存在低光照风险。")
        suggestions.append("增加室内照明或配置夜间感应灯。")
    if not corridor_valid:
        reasons.append("无法可靠识别底部中心通道，当前风险结果可信度有限。")
        suggestions.append("重新采集图像或调整摄像头视角，确保地面和底部中心区域可见。")

    if not mask_valid:
        reasons.append("有效分割像素比例不足，无法完成可靠环境判断。")
        suggestions.append("检查模型输出和输入图像，重新进行分割。")

    level = _risk_level(score)
    if not reasons:
        reasons.append("未发现明显的 BiSeNet 可见环境风险。")
    if not suggestions:
        suggestions.append("保持通道整洁、地面干燥并确保照明充足。")

    return RiskResult(
        score=score,
        level=level,
        scoring_version=RISK_SCORING_VERSION,
        reasons=reasons,
        suggestions=list(dict.fromkeys(suggestions)),
        component_scores=component_scores,
    )


def analyze_bisenet(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    yolo_detections: Optional[Iterable[Any]] = None,
    min_component_area: Optional[int] = None,
    min_valid_pixel_ratio: float = 0.80,
) -> BisenetAnalysis:
    """Extract features and fuse optional YOLO detections into the risk score."""

    height, width = _validate_inputs(image, mask)
    mask = mask.astype(np.uint8, copy=False)
    min_area = int(min_component_area or _minimum_component_area(height, width))
    valid = mask != IGNORE_ID
    valid_count = int(valid.sum())
    valid_pixel_ratio = float(valid_count / float(height * width))
    mask_valid = valid_count > 0 and valid_pixel_ratio >= min_valid_pixel_ratio
    denominator = float(max(valid_count, 1))

    class_masks = {
        class_id: _clean_class_mask(mask, class_id, min_area)
        for class_id in range(len(CLASS_NAMES))
    }
    cleaned_mask = np.full(mask.shape, IGNORE_ID, dtype=np.uint8)
    # Later assignments have higher priority when morphology creates overlap.
    for class_id in (
        CLASS_IDS["non_walkable"],
        CLASS_IDS["wall_boundary"],
        CLASS_IDS["walkable_surface"],
        CLASS_IDS["obstacle"],
        CLASS_IDS["slippery_surface"],
        CLASS_IDS["step_threshold"],
    ):
        cleaned_mask[class_masks[class_id]] = class_id

    # Preserve original categorical pixels for ratios. Cleaned masks are used
    # for components and geometry so one-pixel noise does not become a hazard.
    features: Dict[str, Any] = {
        "valid_pixel_ratio": valid_pixel_ratio,
    }
    for class_name, class_id in CLASS_IDS.items():
        pixel_count = int(np.count_nonzero((mask == class_id) & valid))
        features[class_name + "_pixel_count"] = pixel_count
        features[class_name + "_area_ratio"] = float(pixel_count / denominator)

    obstacle_mask = class_masks[CLASS_IDS["obstacle"]]
    obstacle_components = _components(obstacle_mask, min_area)
    obstacle_pixels = int(obstacle_mask.sum())
    lower_frame = np.zeros_like(obstacle_mask, dtype=bool)
    lower_frame[int(height * 0.80) :, :] = True
    features.update(
        {
            "obstacle_count": int(len(obstacle_components)),
            "obstacle_region_count": int(len(obstacle_components)),
            "largest_obstacle_ratio": float(
                max((item["area_ratio"] for item in obstacle_components), default=0.0)
            ),
            "obstacle_total_cleaned_ratio": float(obstacle_pixels / denominator),
            "obstacle_bottom_contact_ratio": float(
                (obstacle_mask & lower_frame).sum() / float(max(obstacle_pixels, 1))
            ),
        }
    )

    candidate = np.zeros_like(mask, dtype=bool)
    for class_id in _DANGER_CLASSES:
        candidate |= class_masks[class_id]
    corridor_mask, corridor_valid, corridor_fallback = _build_corridor(
        class_masks[CLASS_IDS["walkable_surface"]],
        candidate,
    )
    corridor_pixels = int(corridor_mask.sum())
    corridor_denominator = float(max(corridor_pixels, 1))
    corridor_obstacle_mask = obstacle_mask & corridor_mask
    corridor_obstacle_count = _annotate_corridor_overlap(
        obstacle_components,
        corridor_mask,
        corridor_valid,
    )
    yolo_available = yolo_detections is not None
    normalized_yolo, yolo_occupied = _normalize_yolo_detections(
        [] if yolo_detections is None else yolo_detections,
        (height, width),
        corridor_mask,
        corridor_valid,
    )
    full_obstacle_pixels = int(obstacle_mask.sum())

    yolo_class_counts: Dict[str, int] = {}
    for detection in normalized_yolo:
        if detection["in_corridor"] is True:
            name = str(detection["class_name"])
            yolo_class_counts[name] = yolo_class_counts.get(name, 0) + 1

    features.update(
        {
            "corridor_valid": bool(corridor_valid),
            "corridor_fallback_used": bool(corridor_fallback),
            "corridor_area_ratio": float(corridor_pixels / denominator)
            if corridor_valid
            else None,
            "corridor_obstacle_count": int(corridor_obstacle_count)
            if corridor_valid
            else None,
            "corridor_obstacle_occupancy": float(
                corridor_obstacle_mask.sum() / corridor_denominator
            )
            if corridor_valid
            else None,
            "corridor_slippery_ratio": float(
                (class_masks[CLASS_IDS["slippery_surface"]] & corridor_mask).sum()
                / corridor_denominator
            )
            if corridor_valid
            else None,
            "corridor_step_ratio": float(
                (class_masks[CLASS_IDS["step_threshold"]] & corridor_mask).sum()
                / corridor_denominator
            )
            if corridor_valid
            else None,
            "obstacle_channel_overlap_ratio": float(
                corridor_obstacle_mask.sum() / float(max(full_obstacle_pixels, 1))
            )
            if corridor_valid
            else None,
            "yolo_detection_count": int(len(normalized_yolo))
            if yolo_available
            else None,
            "yolo_corridor_detection_count": int(
                sum(item["in_corridor"] is True for item in normalized_yolo)
            )
            if yolo_available and corridor_valid
            else None,
            "yolo_corridor_occupancy": float(
                yolo_occupied.sum() / corridor_denominator
            )
            if yolo_available and corridor_valid
            else None,
            "yolo_class_counts": yolo_class_counts if yolo_available else None,
            "yolo_category_risk_points": _yolo_category_risk_points(
                normalized_yolo
            )
            if yolo_available and corridor_valid
            else None,
        }
    )
    global_slippery_ratio = float(features["slippery_surface_area_ratio"])
    corridor_slippery_ratio = float(features["corridor_slippery_ratio"] or 0.0)
    features["slippery_surface_score"] = float(
        0.40 * _clip01(global_slippery_ratio / 0.25)
        + 0.60 * _clip01(corridor_slippery_ratio / 0.25)
        if corridor_valid
        else _clip01(global_slippery_ratio / 0.25)
    )

    step_components = _components(class_masks[CLASS_IDS["step_threshold"]], min_area)
    features["step_threshold_count"] = int(len(step_components))

    passage_widths: List[int] = []
    if corridor_valid:
        for y in range(int(height * 0.35), height):
            width_px = _max_run_length(
                corridor_mask[y] & class_masks[CLASS_IDS["walkable_surface"]][y]
            )
            if width_px >= 2:
                passage_widths.append(width_px)
    if passage_widths:
        narrowest_px = int(round(float(np.percentile(passage_widths, 5))))
        features["narrowest_passage_width_px"] = narrowest_px
        features["narrowest_passage_width_ratio"] = float(narrowest_px / float(width))
    else:
        features["narrowest_passage_width_px"] = None
        features["narrowest_passage_width_ratio"] = None

    light_roi = corridor_mask if corridor_valid else None
    features.update(_low_light_features(image, light_roi))
    risk = _score_risk(features, mask_valid, normalized_yolo)

    quality = {
        "valid_pixel_ratio": features["valid_pixel_ratio"],
        "min_valid_pixel_ratio": float(min_valid_pixel_ratio),
        "mask_valid": bool(mask_valid),
        "min_component_area_px": int(min_area),
        "corridor_valid": bool(corridor_valid),
        "corridor_fallback_used": bool(corridor_fallback),
        "step_threshold_reliability": "low",
        "slippery_surface_reliability": "medium",
        "risk_scoring_version": RISK_SCORING_VERSION,
        "risk_component_max_points": dict(RISK_COMPONENT_MAX_POINTS),
        "passage_scoring_method": "weighted_composite_capped_at_45",
        "passage_evidence_weights": dict(PASSAGE_EVIDENCE_WEIGHTS),
        "slippery_carpet_discount_factor": float(
            CARPET_SLIPPERY_RELIABILITY_FACTOR
        ),
        "specific_obstacle_categories_available": bool(yolo_available),
        "yolo_corridor_overlap_threshold": float(YOLO_MIN_CORRIDOR_OVERLAP),
        "yolo_scoring_method": "max_evidence_inside_passage_composite",
        "obstacle_count_semantics": "connected_regions_not_object_instances",
        "low_light_method": "corridor_or_lower_frame_heuristic",
        "absolute_metric_scale_available": False,
    }
    available = [
        key
        for key, value in features.items()
        if value is not None
    ]
    unavailable = [
        "depth_change_score",
        "narrowest_passage_width_m",
    ]
    if not yolo_available:
        unavailable.append("specific_obstacle_categories")
    unavailable.extend(
        key for key, value in features.items() if value is None and key not in unavailable
    )
    return BisenetAnalysis(
        image_size=(height, width),
        mask_valid=mask_valid,
        features=features,
        risk=risk,
        quality=quality,
        available_features=available,
        unavailable_features=unavailable,
        cleaned_mask=cleaned_mask,
        corridor_mask=corridor_mask,
        obstacle_mask=obstacle_mask,
        obstacle_instances=obstacle_components,
        yolo_detections=normalized_yolo,
    )


def _draw_text_block(image: np.ndarray, lines: Iterable[str]) -> None:
    y = 28
    for line in lines:
        cv2.putText(
            image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (255, 255, 255), 3, cv2.LINE_AA
        )
        cv2.putText(
            image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (20, 20, 20), 1, cv2.LINE_AA
        )
        y += 27


def render_analysis_overlay(
    image: np.ndarray,
    analysis: BisenetAnalysis,
    *,
    class_colors: np.ndarray = CLASS_COLORS,
    alpha: float = 0.35,
) -> np.ndarray:
    """Render mask, corridor, obstacle contours and risk text on an image."""

    if image.ndim != 3 or image.shape[:2] != analysis.corridor_mask.shape:
        raise ValueError("image shape does not match analysis")
    overlay = image.copy()
    color_mask = np.zeros_like(image, dtype=np.uint8)
    for class_id in range(len(CLASS_NAMES)):
        color_mask[analysis.cleaned_mask == class_id] = class_colors[class_id]
    blended = cv2.addWeighted(image, 1.0 - alpha, color_mask, alpha, 0.0)
    valid_mask = analysis.cleaned_mask != IGNORE_ID
    overlay[valid_mask] = blended[valid_mask]

    corridor_color = np.zeros_like(image, dtype=np.uint8)
    corridor_color[analysis.corridor_mask] = (255, 190, 0)
    corridor_pixels = analysis.corridor_mask
    if np.any(corridor_pixels):
        overlay[corridor_pixels] = np.clip(
            overlay[corridor_pixels].astype(np.float32) * 0.55
            + corridor_color[corridor_pixels].astype(np.float32) * 0.45,
            0,
            255,
        ).astype(np.uint8)

    contours, _ = cv2.findContours(
        analysis.obstacle_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
    corridor_contours, _ = cv2.findContours(
        analysis.corridor_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(overlay, corridor_contours, -1, (255, 190, 0), 2)

    for detection in analysis.yolo_detections:
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        in_corridor = detection["in_corridor"]
        color = (0, 0, 255) if in_corridor is True else (160, 160, 160)
        cv2.rectangle(overlay, (x1, y1), (max(x1, x2 - 1), max(y1, y2 - 1)), color, 2)
        label = "{} {:.0%}".format(
            detection["class_name"], float(detection["confidence"])
        )
        cv2.putText(
            overlay,
            label,
            (x1, max(16, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            label,
            (x1, max(16, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )

    width_ratio = analysis.features.get("narrowest_passage_width_ratio")
    width_text = "N/A" if width_ratio is None else "{:.1%}".format(width_ratio)
    obstacle_count = max(
        int(analysis.features.get("obstacle_count", 0) or 0),
        int(analysis.features.get("yolo_corridor_detection_count", 0) or 0),
    )
    corridor_occupancy = max(
        float(analysis.features.get("corridor_obstacle_occupancy", 0.0) or 0.0),
        float(analysis.features.get("yolo_corridor_occupancy", 0.0) or 0.0),
    )
    lines = [
        "Risk: {} ({:.1f}/100)".format(analysis.risk.level, analysis.risk.score),
        "Obstacles: {} | Corridor occupancy: {:.1%}".format(
            obstacle_count,
            corridor_occupancy,
        ),
        "Narrow width: {}".format(width_text),
    ]
    _draw_text_block(overlay, lines)
    return overlay
