#!/usr/bin/env bash
# Finish conda setup when the people_detection env already exists (e.g. after a partial run).
set -euo pipefail

PAPER_REF_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MMDET3D_ROOT="$PAPER_REF_ROOT/third_party/mmdetection3d"
ENV_NAME="${ENV_NAME:-people_detection}"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

if [[ ! -d "$MMDET3D_ROOT/mmdet3d" ]]; then
  git -C "$PAPER_REF_ROOT" submodule update --init third_party/mmdetection3d
fi

pushd "$MMDET3D_ROOT" >/dev/null
python setup.py develop --no-deps
popd >/dev/null

PP_DST="$MMDET3D_ROOT/projects/PointPillar"
if [[ ! -d "$PP_DST" ]]; then
  cp -r "$PAPER_REF_ROOT/pointpillar" "$PP_DST"
fi

pip install -r "$PAPER_REF_ROOT/requirements.txt"
python "$PAPER_REF_ROOT/scripts/verify_env.py" --device cpu

echo "Finished setup for $ENV_NAME"
