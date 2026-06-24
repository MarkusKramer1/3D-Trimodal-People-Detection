#!/usr/bin/env bash
# Install OpenMMLab + bundled MMDetection3D submodule into the active Python env.
set -euo pipefail

PAPER_REF_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MMDET3D_ROOT="$PAPER_REF_ROOT/third_party/mmdetection3d"

echo "==> 3D Trimodal People Detection environment setup"
echo "    PAPER_REF_ROOT=$PAPER_REF_ROOT"
echo "    MMDET3D_ROOT=$MMDET3D_ROOT"
echo "    Python: $(python -c 'import sys; print(sys.executable)')"

if [[ ! -d "$MMDET3D_ROOT/mmdet3d" ]]; then
  echo "Initializing MMDetection3D submodule …"
  git -C "$PAPER_REF_ROOT" submodule update --init third_party/mmdetection3d
fi

if [[ ! -d "$MMDET3D_ROOT/mmdet3d" ]]; then
  echo "ERROR: MMDetection3D submodule missing at $MMDET3D_ROOT" >&2
  exit 1
fi

TORCH_VER="$(python - <<'PY'
import torch
v = torch.__version__.split("+")[0]
parts = v.split(".")
print(f"{parts[0]}.{parts[1]}")
PY
)"
CUDA_TAG="$(python - <<'PY'
import torch
if not torch.cuda.is_available():
    print("cpu")
else:
    cap = torch.version.cuda or "121"
    cap = cap.replace(".", "")
    print(f"cu{cap}")
PY
)"
echo "    PyTorch: $(python -c 'import torch; print(torch.__version__)')  mmcv tag: torch${TORCH_VER} ${CUDA_TAG}"

if python - <<'PY'
import torch
major, minor = (int(x) for x in torch.__version__.split(".")[:2])
raise SystemExit(0 if (major, minor) <= (2, 1) else 1)
PY
then
  echo "WARNING: PyTorch > 2.1 — prebuilt mmcv wheels may be unavailable." >&2
  echo "         Recommended: bash scripts/setup_conda_env.sh  (conda env with torch 2.1.2+cu121)" >&2
fi

echo "==> OpenMMLab stack (mmengine, mmcv, mmdet)"
pip install "numpy>=1.24,<2" "setuptools>=70,<82"
pip install -U "mmengine>=0.10" "mmdet>=3.0.0,<3.3.0"

MMCV_WHEEL="https://download.openmmlab.com/mmcv/dist/${CUDA_TAG}/torch${TORCH_VER}/index.html"
if ! pip install "mmcv>=2.0.0rc4,<2.2.0" -f "$MMCV_WHEEL"; then
  echo "ERROR: Could not install mmcv for torch${TORCH_VER} / ${CUDA_TAG}." >&2
  echo "       Use the conda installer instead:" >&2
  echo "         bash scripts/setup_conda_env.sh && conda activate people_detection" >&2
  exit 1
fi

echo "==> MMDetection3D runtime deps"
pip install ninja scikit-image open3d "opencv-python<4.10" "plyfile<1.1" \
  trimesh tensorboard numba "matplotlib>=3.8"

echo "==> Install bundled MMDetection3D"
pip install "$MMDET3D_ROOT" --no-build-isolation
pushd "$MMDET3D_ROOT" >/dev/null
python setup.py develop --no-deps
popd >/dev/null

echo "==> PointPillar custom project"
PP_DST="$MMDET3D_ROOT/projects/PointPillar"
if [[ ! -d "$PP_DST" ]]; then
  cp -r "$PAPER_REF_ROOT/pointpillar" "$PP_DST"
fi

echo "==> point painting Python deps"
pip install -r "$PAPER_REF_ROOT/requirements.txt"

echo "==> Verify imports"
python "$PAPER_REF_ROOT/scripts/verify_env.py" --device cpu

cat <<EOF

Done.

  cd $PAPER_REF_ROOT
  python scripts/demo_visualize.py --device cuda:0
  python scripts/run_pipeline.py --device cuda:0

EOF
