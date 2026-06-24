"""Single-process, in-memory pipeline: YOLO26-seg -> MLP seed painting -> PointPillar."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from src.paint_config import config_to_virtual_paint, load_experiment_config
from src.virtual_point_painting import (
    SequencePaintCache,
    VirtualPaintResult,
    augment_point_cloud,
    load_frame_lidar_cached,
    load_rectified_image_cached,
)
from src.yolo_models import load_yolo_seg_model
from src.yolo_inference import (
    Modality,
    YoloInferenceSettings,
    default_inference_settings,
    extract_pred_masks,
    run_yolo_predict,
)

from src.pointpillar_infer import DetectionResult, PointPillarRuntime


@dataclass
class FrameTimings:
    yolo_rgb_s: float = 0.0
    yolo_thermal_s: float = 0.0
    yolo_wall_s: float = 0.0
    paint_s: float = 0.0
    detect_s: float = 0.0

    @property
    def total_s(self) -> float:
        yolo = self.yolo_wall_s if self.yolo_wall_s > 0.0 else (self.yolo_rgb_s + self.yolo_thermal_s)
        return yolo + self.paint_s + self.detect_s


@dataclass
class FastPipelineResult:
    paint: VirtualPaintResult
    detections: Any | None
    inline_mask_counts: dict[str, int] = field(default_factory=dict)
    inline_masks: dict[str, list[tuple[int, np.ndarray, float]]] = field(default_factory=dict)
    images: dict[str, tuple[np.ndarray, int, int] | None] = field(default_factory=dict)
    timings: FrameTimings = field(default_factory=FrameTimings)


class FastPipeline:
    """Loads all models once; runs one frame without intermediate JSON/bin artifacts."""

    def __init__(
        self,
        *,
        data_root: Path,
        seed_config: Path | None = None,
        rgb_checkpoint: Path | None = None,
        thermal_checkpoint: Path | None = None,
        pointpillar_config: Path | None = None,
        pointpillar_checkpoint: Path | None = None,
        mmdet3d_root: Path | None = None,
        device: str = "cuda:0",
        load_detector: bool = True,
        rgb_settings: YoloInferenceSettings | None = None,
        thermal_settings: YoloInferenceSettings | None = None,
        parallel_yolo: bool = True,
        prefer_tensorrt: bool = True,
    ) -> None:
        root = Path(__file__).resolve().parents[1]
        self.data_root = Path(data_root).resolve()
        cfg_path = seed_config or root / "configs/seed_painting.yaml"
        self.paint_cfg = config_to_virtual_paint(load_experiment_config(cfg_path))

        rgb_ckpt = rgb_checkpoint or root / "checkpoints/best_rgb_yolo.ptrom"
        th_ckpt = thermal_checkpoint or root / "checkpoints/best_thermal_yolo.ptrom"
        self.rgb_settings = rgb_settings or default_inference_settings("rgb")
        self.thermal_settings = thermal_settings or default_inference_settings("thermal")

        self._rgb_yolo = load_yolo_seg_model(
            rgb_ckpt,
            prefer_tensorrt=prefer_tensorrt,
            imgsz=self.rgb_settings.imgsz,
        )
        self._thermal_yolo = load_yolo_seg_model(
            th_ckpt,
            prefer_tensorrt=prefer_tensorrt,
            imgsz=self.thermal_settings.imgsz,
        )
        self._device = device
        self._parallel_yolo = parallel_yolo
        self._prefer_tensorrt = prefer_tensorrt

        self._detector: PointPillarRuntime | None = None
        if load_detector:
            self._detector = PointPillarRuntime(
                config_path=pointpillar_config,
                checkpoint_path=pointpillar_checkpoint,
                mmdet3d_root=mmdet3d_root,
                device=device,
            )

        self._scene_caches: dict[str, SequencePaintCache] = {}

    def _scene_cache(self, scene: str) -> SequencePaintCache:
        if scene not in self._scene_caches:
            self._scene_caches[scene] = SequencePaintCache.load(self.data_root, scene)
        return self._scene_caches[scene]

    def _yolo_settings(self, modality: Modality) -> YoloInferenceSettings:
        return self.rgb_settings if modality == "rgb" else self.thermal_settings

    def _yolo_model(self, modality: Modality) -> Any:
        return self._rgb_yolo if modality == "rgb" else self._thermal_yolo

    def _infer_masks_from_image(
        self,
        modality: Modality,
        loaded: tuple[np.ndarray, int, int] | None,
    ) -> list[tuple[int, np.ndarray, float]]:
        if loaded is None:
            return []
        image_rgb, h, w = loaded
        settings = self._yolo_settings(modality)
        res = run_yolo_predict(
            self._yolo_model(modality),
            image_rgb,
            conf=settings.conf,
            iou=settings.iou,
            imgsz=settings.imgsz,
            device=self._device,
        )
        masks, scores = extract_pred_masks(res, self._yolo_model(modality), h, w, min_conf=settings.conf)
        return [(i, m, float(s)) for i, (m, s) in enumerate(zip(masks, scores))]

    def _run_yolo_modality(
        self,
        scene: str,
        frame_idx: int,
        modality: Modality,
    ) -> tuple[list[tuple[int, np.ndarray, float]], float]:
        t0 = time.perf_counter()
        loaded = load_rectified_image_cached(self._scene_cache(scene), frame_idx, modality)
        masks = self._infer_masks_from_image(modality, loaded)
        return masks, time.perf_counter() - t0

    def _load_rectified_parallel(
        self,
        scene: str,
        frame_idx: int,
        modalities: list[Modality],
    ) -> dict[str, tuple[np.ndarray, int, int] | None]:
        cache = self._scene_cache(scene)
        if self._parallel_yolo and len(modalities) > 1:
            with ThreadPoolExecutor(max_workers=len(modalities)) as pool:
                futs = {
                    mod: pool.submit(load_rectified_image_cached, cache, frame_idx, mod)
                    for mod in modalities
                }
                return {mod: futs[mod].result() for mod in modalities}
        return {mod: load_rectified_image_cached(cache, frame_idx, mod) for mod in modalities}

    def run_frame(
        self,
        scene: str,
        frame_idx: int = 0,
        *,
        score_thr: float = 0.1,
        modalities: tuple[Literal["rgb", "thermal"], ...] = ("rgb", "thermal"),
    ) -> FastPipelineResult:
        timings = FrameTimings()
        inline: dict[str, list[tuple[int, np.ndarray, float]]] = {}
        images: dict[str, tuple[np.ndarray, int, int] | None] = {}
        yolo_mods: list[Modality] = list(modalities)

        if yolo_mods:
            t_wall = time.perf_counter()
            cache = self._scene_cache(scene)
            with ThreadPoolExecutor(max_workers=2) as pool:
                lidar_fut = pool.submit(load_frame_lidar_cached, cache, frame_idx)
                images_fut = pool.submit(self._load_rectified_parallel, scene, frame_idx, yolo_mods)
                images = images_fut.result()
                for mod in yolo_mods:
                    t0 = time.perf_counter()
                    inline[mod] = self._infer_masks_from_image(mod, images[mod])
                    elapsed = time.perf_counter() - t0
                    if mod == "rgb":
                        timings.yolo_rgb_s = elapsed
                    else:
                        timings.yolo_thermal_s = elapsed
                lidar_fut.result()
            timings.yolo_wall_s = time.perf_counter() - t_wall
        else:
            load_frame_lidar_cached(self._scene_cache(scene), frame_idx)

        t0 = time.perf_counter()
        paint, err = augment_point_cloud(
            self.data_root,
            scene,
            frame_idx,
            self.paint_cfg,
            cache=self._scene_cache(scene),
            inline_masks_by_modality=inline,
            device=self._device,
        )
        timings.paint_s = time.perf_counter() - t0
        if paint is None:
            raise RuntimeError(err or "Seed painting failed")

        detections = None
        if self._detector is not None:
            t0 = time.perf_counter()
            detections = self._detector.detect(paint.features, score_thr=score_thr)
            timings.detect_s = time.perf_counter() - t0

        return FastPipelineResult(
            paint=paint,
            detections=detections,
            inline_mask_counts={k: len(v) for k, v in inline.items()},
            inline_masks=dict(inline),
            images=images,
            timings=timings,
        )
