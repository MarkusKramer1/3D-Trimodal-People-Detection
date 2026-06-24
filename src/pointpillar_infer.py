"""In-memory PointPillar inference (no disk I/O per frame)."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.mmdet3d_paths import ensure_mmdet3d_on_path, sync_pointpillar_project

PAPER_REF_ROOT = Path(__file__).resolve().parents[1]

_SETUP_HINT = (
    "Install the OpenMMLab stack and MMDetection3D submodule:\n"
    "  git submodule update --init third_party/mmdetection3d\n"
    "  bash scripts/setup_env.sh"
)


@dataclass
class DetectionResult:
    labels_3d: list[int]
    scores_3d: list[float]
    bboxes_3d: list[list[float]]
    box_type_3d: str = "LiDAR"
    class_names: tuple[str, ...] = ("Pedestrian",)


def _pred_to_detection(pred: Any, score_thr: float) -> DetectionResult:
    if isinstance(pred, dict):
        scores = np.asarray(pred["scores_3d"], dtype=np.float64)
        keep = scores >= score_thr
        return DetectionResult(
            labels_3d=np.asarray(pred["labels_3d"], dtype=int)[keep].tolist(),
            scores_3d=scores[keep].astype(float).tolist(),
            bboxes_3d=np.asarray(pred["bboxes_3d"], dtype=np.float64)[keep].tolist(),
        )

    inst = pred.pred_instances_3d
    scores = inst.scores_3d.detach().cpu().numpy()
    keep = scores >= score_thr
    boxes = inst.bboxes_3d.tensor.detach().cpu().numpy()[keep]
    labels = inst.labels_3d.detach().cpu().numpy()[keep]
    scores = scores[keep]
    return DetectionResult(
        labels_3d=labels.astype(int).tolist(),
        scores_3d=scores.astype(float).tolist(),
        bboxes_3d=boxes.astype(float).tolist(),
    )


def _points_for_detector(
    features: np.ndarray,
    *,
    point_cloud_range: np.ndarray | None = None,
) -> np.ndarray:
    """PointPillar expects ``(N, 4)`` xyz+intensity; painted features may be ``(N, 5)``."""
    arr = np.asarray(features, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError(f"Expected (N, >=4) features, got {arr.shape}.")
    if point_cloud_range is not None and arr.shape[0] > 0:
        pcr = np.asarray(point_cloud_range, dtype=np.float32).reshape(6)
        keep = (
            (arr[:, 0] >= pcr[0])
            & (arr[:, 0] <= pcr[3])
            & (arr[:, 1] >= pcr[1])
            & (arr[:, 1] <= pcr[4])
            & (arr[:, 2] >= pcr[2])
            & (arr[:, 2] <= pcr[5])
        )
        arr = arr[keep]
    return np.ascontiguousarray(arr[:, :4])


class PointPillarRuntime:
    """Loads the detector once; runs ``model.test_step`` on in-memory points."""

    def __init__(
        self,
        *,
        config_path: Path | None = None,
        checkpoint_path: Path | None = None,
        mmdet3d_root: Path | None = None,
        device: str = "cuda:0",
    ) -> None:
        self.config_path = (
            config_path or PAPER_REF_ROOT / "pointpillar/configs/inference.py"
        ).resolve()
        self.checkpoint_path = (checkpoint_path or PAPER_REF_ROOT / "checkpoints/epoch_200.pth").resolve()
        self.mmdet3d_root = ensure_mmdet3d_on_path(mmdet3d_root)
        self.device = device
        sync_pointpillar_project(self.mmdet3d_root)

        try:
            from mmengine.config import Config  # noqa: E402
            from mmengine.dataset import Compose  # noqa: E402
            from mmdet3d.apis import init_model  # noqa: E402
            from mmdet3d.structures import get_box_type  # noqa: E402
        except ImportError as exc:
            raise RuntimeError(_SETUP_HINT) from exc

        self._model = init_model(
            str(self.config_path),
            str(self.checkpoint_path),
            device=device,
        )
        cfg = Config.fromfile(str(self.config_path))
        pipeline_cfg = deepcopy(cfg.test_dataloader.dataset.pipeline)
        load_idx = next(
            i for i, step in enumerate(pipeline_cfg) if step.get("type") == "LoadPointsFromFile"
        )
        pipeline_cfg[load_idx]["type"] = "LoadPointsFromDict"
        self._box_type_3d, self._box_mode_3d = get_box_type(cfg.test_dataloader.dataset.box_type_3d)
        self._pipeline = Compose(pipeline_cfg)
        self._point_cloud_range = np.asarray(
            cfg.model.data_preprocessor.voxel_layer.point_cloud_range,
            dtype=np.float32,
        )
        self._axis_align = np.eye(4, dtype=np.float32)
        self._model.eval()
        import torch  # noqa: E402

        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

    def detect(
        self,
        features: np.ndarray,
        *,
        score_thr: float = 0.1,
    ) -> DetectionResult:
        """Run 3D detection on ``(N, 5)`` painted features without writing a ``.bin`` file."""
        import torch  # noqa: E402
        from mmengine.dataset import pseudo_collate  # noqa: E402

        points = _points_for_detector(features, point_cloud_range=self._point_cloud_range)
        data = self._pipeline(
            dict(
                points=points,
                timestamp=1,
                axis_align_matrix=self._axis_align,
                box_type_3d=self._box_type_3d,
                box_mode_3d=self._box_mode_3d,
            )
        )
        batch = pseudo_collate([data])
        use_cuda = str(self.device).startswith("cuda") and torch.cuda.is_available()
        with torch.inference_mode():
            if use_cuda:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pred = self._model.test_step(batch)[0]
            else:
                pred = self._model.test_step(batch)[0]
        return _pred_to_detection(pred, score_thr)
