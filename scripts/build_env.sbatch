#!/bin/bash
# Builds the DipDesk P2 venv (TE 2.20.2 + PEFT + transformers) on DeltaAI. This is the recipe that
# worked on 2026-10-31 (TE was compiled on the login node; a compute node works the same way).
# Usage: sbatch scripts/build_env.sbatch   (from the repo root), or run the body by hand.
#SBATCH --account=bcrc-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=00:45:00
#SBATCH --job-name=p2-build-env
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
module load python/miniforge3_pytorch/2.10.0
ENV=/projects/bcrc/nvarma2/envs
SDK=/opt/nvidia/hpc_sdk/Linux_aarch64/25.5
[ -d $ENV/dipdesk ] || python -m venv --system-site-packages $ENV/dipdesk
source $ENV/dipdesk/bin/activate
pip install "transformers>=4.46,<5" peft accelerate
# TE 2.20 needs cuDNN >= 9.12 headers; the module ships 9.10. --no-deps keeps pip from reverting it
# to torch's pin (torch 2.10 runs fine on 9.27; recorded deviation, docs/precision.md).
pip install --no-deps "nvidia-cudnn-cu12==9.27.0.42"
CUDNN=$ENV/dipdesk/lib/python3.12/site-packages/nvidia/cudnn
rm -rf $ENV/cudnn && mkdir -p $ENV/cudnn/lib && ln -s $CUDNN/include $ENV/cudnn/include
for f in $CUDNN/lib/*.so.9; do ln -sf $f $ENV/cudnn/lib/; ln -sf $f $ENV/cudnn/lib/$(basename $f .9); done
export CUDA_HOME=$SDK/cuda/12.9 CUDNN_PATH=$ENV/cudnn CUDNN_HOME=$ENV/cudnn
export CPATH=$ENV/cudnn/include:$SDK/math_libs/12.9/include:$SDK/comm_libs/12.9/nccl/include
export LIBRARY_PATH=$ENV/cudnn/lib:$SDK/math_libs/12.9/lib64:$SDK/comm_libs/12.9/nccl/lib
export NVTE_FRAMEWORK=pytorch MAX_JOBS=16 NVTE_CUDA_ARCHS=90 CC=gcc CXX=g++
pip install --no-build-isolation "transformer_engine[pytorch]==2.20.2"
pip install --no-deps "nvidia-cudnn-cu12==9.27.0.42"   # TE's install re-pins 9.10; restore 9.27
source scripts/env.sh
python -c "import torch, transformer_engine.pytorch as te; print('TE OK', torch.backends.cudnn.version(), te.is_fp8_available())"
