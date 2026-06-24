"""Resolve bundled MMDetection3D submodule paths."""

from __future__ import annotations

import sys
from pathlib import Path

PAPER_REF_ROOT = Path(__file__).resolve().parents[1]
MMDET3D_SUBMODULE = PAPER_REF_ROOT / "third_party/mmdetection3d"


def resolve_mmdet3d_root(explicit: Path | None = None) -> Path:
    """Return the MMDetection3D repo root (bundled submodule by default)."""
    if explicit is not None:
        root = Path(explicit).resolve()
        if (root / "mmdet3d").is_dir():
            return root
        raise RuntimeError(f"Not an MMDetection3D checkout: {root}")

    if (MMDET3D_SUBMODULE / "mmdet3d").is_dir():
        return MMDET3D_SUBMODULE.resolve()

    legacy = PAPER_REF_ROOT.parent / "mmdetection3d"
    if (legacy / "mmdet3d").is_dir():
        return legacy.resolve()

    raise RuntimeError(
        "MMDetection3D submodule not found. Initialize and install it:\n"
        "  git submodule update --init third_party/mmdetection3d\n"
        "  bash scripts/setup_env.sh"
    )


def ensure_mmdet3d_on_path(mmdet3d_root: Path | None = None) -> Path:
    """Insert the MMDetection3D repo root on ``sys.path`` if needed."""
    root = resolve_mmdet3d_root(mmdet3d_root)
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    return root


def sync_pointpillar_project(mmdet3d_root: Path | None = None) -> Path:
    """Copy ``pointpillar/`` into ``mmdetection3d/projects/PointPillar`` once."""
    import shutil

    root = resolve_mmdet3d_root(mmdet3d_root)
    dst = root / "projects" / "PointPillar"
    if not dst.is_dir():
        shutil.copytree(PAPER_REF_ROOT / "pointpillar", dst)
    return dst
