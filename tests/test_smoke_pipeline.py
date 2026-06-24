"""Smoke and regression tests for the paper reference pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PAPER_REF_ROOT = Path(__file__).resolve().parents[1]
if str(PAPER_REF_ROOT) not in sys.path:
    sys.path.insert(0, str(PAPER_REF_ROOT))

from src.fast_pipeline import FastPipeline  # noqa: E402


@pytest.fixture
def paper_ref_root() -> Path:
    return PAPER_REF_ROOT


def test_fast_pipeline_in_memory_paint(paper_ref_root: Path) -> None:
    pytest.importorskip("ultralytics")
    pipeline = FastPipeline(
        data_root=paper_ref_root / "sample_data",
        load_detector=False,
        device="cpu",
        prefer_tensorrt=False,
    )
    result = pipeline.run_frame("example_scene", 0)
    assert result.paint.features.shape[1] == 5
    assert int(result.paint.is_virtual.sum()) > 0
    assert result.inline_mask_counts.get("rgb", 0) >= 0
    assert result.timings.total_s >= 0.0


def test_pipeline_scene_cache(paper_ref_root: Path) -> None:
    pipeline = FastPipeline(
        data_root=paper_ref_root / "sample_data",
        load_detector=False,
        prefer_tensorrt=False,
    )
    cache = pipeline._scene_cache("example_scene")
    assert cache.subset == "example_scene"
    assert len(cache.links) > 0
