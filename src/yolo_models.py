"""Load YOLO26-seg person segmentation checkpoints."""

from __future__ import annotations

from pathlib import Path

_YOLO_PT_SUFFIXES = {".pt", ".pth"}
_TASK = "segment"


def load_yolo_seg_model(checkpoint: Path):
    from ultralytics import YOLO

    path = checkpoint.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if path.suffix.lower() not in _YOLO_PT_SUFFIXES:
        raise ValueError(f"Expected a .pt or .pth checkpoint, got {path.name}")

    model = YOLO(str(path), task=_TASK)
    if isinstance(model.model, str):
        raise RuntimeError(f"Failed to load YOLO26-seg weights from {checkpoint}")
    return model
