#!/bin/bash
# Builds a personal copy of the P2 env on DeltaAI GH200 from the prebuilt TE wheel in wheels/
# (no TE compile, ~5 min). Same versions as the shared env (docs/precision.md).
# Usage (repo root, DeltaAI login or gh node):  bash scripts/install_env_from_wheel.sh [VENV_DIR]
# Then:  export DIPDESK_VENV=<VENV_DIR>; source scripts/env.sh
set -euo pipefail
VENV=${1:-$HOME/envs/dipdesk}
[ "$(uname -m)" = aarch64 ] || { echo "This wheel is aarch64/sm_90 (DeltaAI GH200) only." >&2; exit 1; }
module load python/miniforge3_pytorch/2.10.0
[ -d "$VENV" ] || python -m venv --system-site-packages "$VENV"
source "$VENV/bin/activate"
pip install "transformers==4.57.6" "peft==0.21.2" accelerate "transformer_engine_cu12==2.20.2" \
    wheels/transformer_engine_torch-2.20.2-cp312-cp312-linux_aarch64.whl
# --no-deps: keep cuDNN 9.27 (TE needs >= 9.12; torch pins 9.10), recorded deviation.
pip install --no-deps "transformer_engine==2.20.2" "nvidia-cudnn-cu12==9.27.0.42"
echo "Installed. Next: export DIPDESK_VENV=$VENV; source scripts/env.sh"
