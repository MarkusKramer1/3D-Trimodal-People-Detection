"""Load YOLO26-seg person segmentation checkpoints (.ptrom or TensorRT .engine)."""

from __future__ import annotations

import shutil
from pathlib import Path

_YOLO_PT_SUFFIXES = {".pt", ".pth"}
_ENGINE_SUFFIX = ".engine"
_TASK = "segment"


def normalize_yolo_checkpoint_path(checkpoint: Path) -> Path:
    """Ultralytics loads ``.pt`` / ``.pth``; ``.ptrom`` weights are copied to a cached ``.pt``."""
    path = checkpoint.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if path.suffix.lower() in _YOLO_PT_SUFFIXES:
        return path

    cache_dir = path.parent / "._yolo_load_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        mtime_ns = 0
    cached = cache_dir / f"{path.stem}_{mtime_ns}.pt"
    if not cached.is_file():
        shutil.copy2(path, cached)
    return cached.resolve()


def resolve_yolo_inference_path(
    checkpoint: Path,
    *,
    imgsz: int,
    prefer_tensorrt: bool = True,
) -> Path:
    path = checkpoint.expanduser().resolve()
    if not prefer_tensorrt or path.suffix.lower() == _ENGINE_SUFFIX:
        return path

    from src.yolo_inference import snap_imgsz

    tag = f"imgsz{snap_imgsz(int(imgsz))}_fp16"
    candidates = [
        path.with_name(f"{path.stem}_{tag}{_ENGINE_SUFFIX}"),
        path.with_suffix(_ENGINE_SUFFIX),
    ]
    try:
        pt_path = normalize_yolo_checkpoint_path(path)
        candidates.append(pt_path.with_suffix(_ENGINE_SUFFIX))
    except FileNotFoundError:
        pass

    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
    return path


def load_yolo_seg_model(
    checkpoint: Path,
    *,
    imgsz: int = 400,
    prefer_tensorrt: bool = True,
):
    from ultralytics import YOLO

    resolved = resolve_yolo_inference_path(
        checkpoint,
        imgsz=imgsz,
        prefer_tensorrt=prefer_tensorrt,
    )
    if resolved.suffix.lower() == _ENGINE_SUFFIX:
        return YOLO(str(resolved), task=_TASK)

    load_path = normalize_yolo_checkpoint_path(resolved)
    model = YOLO(str(load_path), task=_TASK)
    if isinstance(model.model, str):
        raise RuntimeError(f"Failed to load YOLO26-seg weights from {checkpoint}")
    return model
