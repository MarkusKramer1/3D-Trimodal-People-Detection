#!/usr/bin/env bash
# Create the paintfusion conda env (YOLO26-seg + MLP seed painting + PointPillars).
set -euo pipefail

PAPER_REF_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MMDET3D_ROOT="$PAPER_REF_ROOT/third_party/mmdetection3d"
ENV_NAME="${ENV_NAME:-paintfusion}"
CUDA_INDEX="${CUDA_INDEX:-https://download.pytorch.org/whl/cu121}"

echo "==> paintfusion conda environment"
echo "    PAPER_REF_ROOT=$PAPER_REF_ROOT"
echo "    MMDET3D_ROOT=$MMDET3D_ROOT"
echo "    ENV_NAME=$ENV_NAME"

if [[ ! -d "$MMDET3D_ROOT/mmdet3d" ]]; then
  echo "Initializing MMDetection3D submodule …"
  git -C "$PAPER_REF_ROOT" submodule update --init third_party/mmdetection3d
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda env '$ENV_NAME' already exists. Remove it first:"
  echo "  conda env remove -n $ENV_NAME"
  exit 1
fi

conda env create -f "$PAPER_REF_ROOT/environment.yml"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "==> PyTorch 2.1.2 + CUDA 12.1 (required for mmcv 2.1 / mmdet3d 1.4)"
pip install "numpy<2"
pip install torch==2.1.2 torchvision==0.16.2 --index-url "$CUDA_INDEX"

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
PY

echo "==> OpenMMLab stack"
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0rc4,<2.2.0"
mim install "mmdet>=3.0.0,<3.3.0"

echo "==> MMDetection3D runtime deps"
pip install ninja scikit-image open3d "opencv-python<4.10" "plyfile<1.1" \
  trimesh tensorboard numba "matplotlib>=3.8"

echo "==> Install bundled MMDetection3D submodule"
pip install "$MMDET3D_ROOT" --no-build-isolation
pushd "$MMDET3D_ROOT" >/dev/null
python setup.py develop --no-deps
popd >/dev/null

echo "==> YOLO + point painting Python deps"
pip install -r "$PAPER_REF_ROOT/requirements.txt"

echo "==> Verify imports"
python "$PAPER_REF_ROOT/scripts/verify_env.py" --device cpu

cat <<EOF

Done.

  conda activate $ENV_NAME
  cd $PAPER_REF_ROOT
  python scripts/demo_visualize.py --device cuda:0

For a full smoke test including PointPillar:

  python scripts/verify_env.py --device cuda:0 --run-pipeline

EOF
