# 3D Trimodal People Detection

Standalone reference for **YOLO26-seg (person) → fused MLP seed painting → PointPillar** on synchronized RGB + thermal + LiDAR frames.

## Pipeline

```
LiDAR + RGB + thermal frame
        │
        ▼
  YOLO26-seg (RGB + thermal, parallel)
        │
        ▼
  MLP seed painting (fused 33-D scorer)
        │
        ▼
  PointPillar 3D detector
        │
        ▼
  3D bounding boxes
```

Entry point: `scripts/run_pipeline.py` → `src/fast_pipeline.py`

## Quick start

Clone with submodules:

```bash
git clone --recurse-submodules git@github.com:MarkusKramer1/3D-Trimodal-People-Detection.git
cd 3D-Trimodal-People-Detection
```

Or, if already cloned:

```bash
git submodule update --init third_party/mmdetection3d
```

```bash
bash scripts/setup_conda_env.sh
conda activate people_detection

python scripts/demo_visualize.py --device cuda:0
python scripts/run_pipeline.py --device cuda:0
```

Verify the environment:

```bash
python scripts/verify_env.py --device cuda:0 --run-pipeline
```

## CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--data-root` | `sample_data/` | Dataset root with scene folders |
| `--scene` | `example_scene` | Scene name |
| `--frame-idx` | `0` | Frame index |
| `--score-thr` | `0.1` | 3D detection score threshold |
| `--device` | `cuda:0` | PyTorch device |
| `--save-detections` | — | Optional JSON output path (`run_pipeline.py` only) |

## Checkpoints (Git LFS)

| File | Role |
|------|------|
| `checkpoints/best_rgb_yolo.ptrom` | YOLO26-seg RGB |
| `checkpoints/best_thermal_yolo.ptrom` | YOLO26-seg thermal |
| `checkpoints/mlp_fused.pkl` | Fused 33-D MLP seed scorer |
| `checkpoints/epoch_200.pth` | PointPillar pedestrian detector |

```bash
git lfs install && git lfs pull
```

## Tests

Unit tests for YOLO + painting run without the full MMDet3D stack (`load_detector=False`). Full pipeline smoke test:

```bash
python scripts/verify_env.py --device cuda:0 --run-pipeline
pytest tests/ -q
```

## Citation

See [CITATION.bib](CITATION.bib).
