"""YOLO26-seg inference helpers for the end-to-end pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np

Modality = Literal["rgb", "thermal"]

DEFAULT_FIXED_IOU = 0.7
DEFAULT_FIXED_IMGSZ = 400
DEFAULT_RGB_INFER_CONF = 0.25
DEFAULT_THERMAL_INFER_CONF = 0.2


def _person_class_ids(names: dict[int, str]) -> set[int]:
    out: set[int] = set()
    for idx, label in names.items():
        if str(label).strip().lower() == "person":
            out.add(int(idx))
    return out


def snap_imgsz(imgsz: int, *, stride: int = 32) -> int:
    """Round image size up to a multiple of model stride (matches Ultralytics snapping)."""
    s = max(1, int(stride))
    v = max(s, int(imgsz))
    return int(math.ceil(v / s) * s)


def _resize_mask_bool(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    m = np.asarray(mask)
    if m.shape[0] == h and m.shape[1] == w:
        return m.astype(bool)
    rsz = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return rsz > 0


def extract_pred_masks(
    result: Any,
    model: Any,
    h: int,
    w: int,
    *,
    min_conf: float = 0.0,
) -> tuple[list[np.ndarray], list[float]]:
    """Person instance masks and scores from one Ultralytics result."""
    if result.boxes is None or len(result.boxes) == 0:
        return [], []
    names = model.names if hasattr(model, "names") else {}
    person_ids = _person_class_ids(names if isinstance(names, dict) else {})
    if not person_ids:
        return [], []

    cls = result.boxes.cls.detach().cpu().numpy().astype(int)
    conf = result.boxes.conf.detach().cpu().numpy()
    if result.masks is None:
        return [], []

    mask_data = result.masks.data.detach().cpu().numpy()
    out_masks: list[np.ndarray] = []
    out_scores: list[float] = []
    for i in range(len(cls)):
        if cls[i] not in person_ids:
            continue
        score = float(conf[i])
        if score < float(min_conf):
            continue
        m = _resize_mask_bool(mask_data[i], h, w)
        if m.any():
            out_masks.append(m)
            out_scores.append(score)
    return out_masks, out_scores


def run_yolo_predict(
    model: Any,
    image_rgb: np.ndarray,
    *,
    conf: float,
    iou: float,
    imgsz: int,
    device: str | None = None,
) -> Any:
    kwargs: dict[str, Any] = dict(
        conf=float(conf),
        iou=float(iou),
        imgsz=snap_imgsz(int(imgsz)),
        verbose=False,
    )
    if device is not None:
        kwargs["device"] = device
    return model.predict(np.asarray(image_rgb, dtype=np.uint8), **kwargs)[0]


@dataclass(frozen=True)
class YoloInferenceSettings:
    conf: float
    iou: float
    imgsz: int


def default_inference_settings(modality: Modality) -> YoloInferenceSettings:
    conf = DEFAULT_RGB_INFER_CONF if modality == "rgb" else DEFAULT_THERMAL_INFER_CONF
    return YoloInferenceSettings(
        conf=float(conf),
        iou=float(DEFAULT_FIXED_IOU),
        imgsz=int(DEFAULT_FIXED_IMGSZ),
    )
