#!/bin/bash
# Builds the DipDesk P2 venv (TE + PEFT + transformers) on a GH200 node.
# Usage: sbatch scripts/build_env.sbatch   (from the repo root)
#SBATCH --account=bcrc-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120g
#SBATCH --time=01:30:00
#SBATCH --job-name=p2-build-env
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
module load python/miniforge3_pytorch/2.10.0
ENV=/projects/bcrc/nvarma2/envs
source $ENV/dipdesk/bin/activate
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/25.5/cuda/12.9
export CUDNN_PATH=$ENV/cudnn CUDNN_HOME=$ENV/cudnn
export CPATH=$ENV/cudnn/include:${CPATH:-} LIBRARY_PATH=$ENV/cudnn/lib:${LIBRARY_PATH:-}
export NVTE_FRAMEWORK=pytorch MAX_JOBS=32 NVTE_CUDA_ARCHS=90 CC=gcc CXX=g++
pip install --no-build-isolation -v "transformer_engine[pytorch]==2.20.2" 2>&1 | tail -40
python -c "import transformer_engine.pytorch as te, torch; print('TE OK', te.__version__ if hasattr(te,'__version__') else '', torch.cuda.get_device_name())"
