#!/usr/bin/env python3
"""Ultralytics YOLO adapter used by the single-image risk pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class YoloDetection:
    """Stable, model-agnostic representation of one YOLO detection."""

    xyxy: Tuple[int, int, int, int]
    class_id: int
    class_name: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox_xyxy": [int(value) for value in self.xyxy],
            "class_id": int(self.class_id),
            "class_name": str(self.class_name),
            "confidence": float(self.confidence),
        }


class UltralyticsYoloAdapter:
    """Load one Ultralytics checkpoint and return :class:`YoloDetection` values."""

    def __init__(
        self,
        model_path: str,
        *,
        device: Optional[str] = None,
        confidence: float = 0.25,
        iou: float = 0.70,
    ) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError("Could not find YOLO checkpoint: {}".format(path))
        if path.stat().st_size == 0:
            raise ValueError(
                "YOLO checkpoint is empty (likely a placeholder): {}".format(path)
            )
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not 0.0 <= iou <= 1.0:
            raise ValueError("iou must be in [0, 1]")

        # Keep the optional dependency out of mask-only analysis and unit tests.
        from ultralytics import YOLO

        self.model = YOLO(str(path))
        self.device = device
        self.confidence = float(confidence)
        self.iou = float(iou)

    def predict(self, image_bgr: np.ndarray) -> List[YoloDetection]:
        if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3:
            raise ValueError("image_bgr must be a HxWxC numpy array")

        kwargs = {
            "source": image_bgr,
            "conf": self.confidence,
            "iou": self.iou,
            "verbose": False,
        }
        if self.device is not None:
            kwargs["device"] = self.device
        result = self.model.predict(**kwargs)[0]
        if result.boxes is None:
            return []

        detections: List[YoloDetection] = []
        names = result.names
        for box in result.boxes:
            class_id = int(box.cls.item())
            if isinstance(names, dict):
                class_name = names.get(class_id, "class_{}".format(class_id))
            else:
                class_name = names[class_id]
            coords = tuple(int(round(value)) for value in box.xyxy[0].tolist())
            detections.append(
                YoloDetection(
                    xyxy=coords,
                    class_id=class_id,
                    class_name=str(class_name),
                    confidence=float(box.conf.item()),
                )
            )
        return detections

