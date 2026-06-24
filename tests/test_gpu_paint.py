"""GPU vs CPU seed painting parity."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PAPER_REF_ROOT = Path(__file__).resolve().parents[1]
if str(PAPER_REF_ROOT) not in sys.path:
    sys.path.insert(0, str(PAPER_REF_ROOT))

from src.paint_config import config_to_virtual_paint, load_experiment_config  # noqa: E402
from src.fast_pipeline import FastPipeline  # noqa: E402
from src.virtual_point_painting import SequencePaintCache, augment_point_cloud  # noqa: E402


@pytest.fixture
def paper_ref_root() -> Path:
    return PAPER_REF_ROOT


def test_gpu_paint_matches_cpu_with_same_masks(paper_ref_root: Path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    pytest.importorskip("ultralytics")
    cfg = config_to_virtual_paint(load_experiment_config(paper_ref_root / "configs/seed_painting.yaml"))
    data_root = paper_ref_root / "sample_data"
    cache = SequencePaintCache.load(data_root, "example_scene")
    pipeline = FastPipeline(data_root=data_root, load_detector=False, device="cpu", prefer_tensorrt=False)
    inline = {}
    for mod in ("rgb", "thermal"):
        inline[mod], _ = pipeline._run_yolo_modality("example_scene", 0, mod)  # type: ignore[arg-type]

    cpu, err_cpu = augment_point_cloud(
        data_root, "example_scene", 0, cfg, cache=cache, inline_masks_by_modality=inline, device="cpu"
    )
    gpu, err_gpu = augment_point_cloud(
        data_root, "example_scene", 0, cfg, cache=cache, inline_masks_by_modality=inline, device="cuda:0"
    )
    assert err_cpu is None and cpu is not None
    assert err_gpu is None and gpu is not None
    assert cpu.features.shape == gpu.features.shape
    assert int(cpu.is_virtual.sum()) == int(gpu.is_virtual.sum())


def test_gpu_featurize_matches_cpu(paper_ref_root: Path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    cfg = config_to_virtual_paint(load_experiment_config(paper_ref_root / "configs/seed_painting.yaml"))
    data_root = paper_ref_root / "sample_data"
    pipeline = FastPipeline(data_root=data_root, load_detector=False, device="cpu", prefer_tensorrt=False)
    inline = {}
    for mod in ("rgb", "thermal"):
        inline[mod], _ = pipeline._run_yolo_modality("example_scene", 0, mod)  # type: ignore[arg-type]

    from src.geometry import lidar_to_camera_from_calibrations
    from src.virtual_point_painting import (
        _build_xmod_from_paint_plans,
        _precompute_seed_scorer_frame,
        get_rectified_view,
        load_frame_lidar_cached,
        project_sweep_to_image,
    )

    cache = pipeline._scene_cache("example_scene")
    link = cache.links[0]
    by_token = cache.by_token
    calibs = cache.calibs
    xyz, intensity = load_frame_lidar_cached(cache, 0)
    lid_s = by_token[link["lidar_token"]]
    rgb_s = by_token[link["rgb_token"]]
    th_s = by_token[link["thermal_token"]]
    lid_cal = calibs[str(lid_s["calibrated_sensor_token"])]
    paint_plans = []
    for mod, sample in (("rgb", rgb_s), ("thermal", th_s)):
        cam_cal = calibs[str(sample["calibrated_sensor_token"])]
        view, _ = get_rectified_view(cache, sample, cam_cal)
        T_lc = lidar_to_camera_from_calibrations(lid_cal, cam_cal)
        proj = project_sweep_to_image(xyz, T_lc, view.K, view.h_rect, view.w_rect)
        masks = inline[mod]
        paint_plans.append(dict(modality=mod, masks=masks, masks_all=masks, projection=proj))
    xmod = _build_xmod_from_paint_plans(paint_plans, int(xyz.shape[0]))
    scorer_plans = [
        dict(
            modality=p["modality"],
            projection=p["projection"],
            instance_masks=p["masks"],
            context_masks=p["masks_all"],
        )
        for p in paint_plans
    ]
    pre_cpu = _precompute_seed_scorer_frame(
        cfg=cfg,
        xyz=xyz,
        intensity=intensity,
        modality_plans=scorer_plans,
        xmod_by_mod=xmod,
        score_device=None,
    )
    pre_gpu = _precompute_seed_scorer_frame(
        cfg=cfg,
        xyz=xyz,
        intensity=intensity,
        modality_plans=scorer_plans,
        xmod_by_mod=xmod,
        score_device=torch.device("cuda:0") if torch.cuda.is_available() else None,
    )
    assert pre_cpu.keys() == pre_gpu.keys()
    for key in pre_cpu:
        _arrays, probs_cpu = pre_cpu[key]
        _arrays2, probs_gpu = pre_gpu[key]
        np.testing.assert_allclose(probs_cpu, probs_gpu, rtol=1e-3, atol=1e-3)
