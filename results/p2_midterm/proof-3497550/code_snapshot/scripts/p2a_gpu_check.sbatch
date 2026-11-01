#!/bin/bash
# P2-A GPU acceptance: precision tests + 3-way toy parity on one GH200 GPU.
# Usage: sbatch scripts/p2a_gpu_check.sbatch   (from the repo root)
#SBATCH --account=bcrc-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=00:20:00
#SBATCH --job-name=p2a-precision
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
module load python/miniforge3_pytorch/2.10.0
OUT=results/p2a_parity/gpu-${SLURM_JOB_ID}
mkdir -p "$OUT"
{ hostname; nvidia-smi; python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())"; } > "$OUT/environment.txt"
python -m pytest -q 2>&1 | tee "$OUT/pytest.txt"
python scripts/precision_parity.py --out "$OUT" --steps 200
