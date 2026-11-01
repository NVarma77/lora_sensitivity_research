#!/bin/bash
# P2 midterm pilots: Qwen2.5-1.5B + LoRA r=16, seed 1, one mode per GPU, in parallel.
# fp8_te doubles as the FP8+LoRA test run (param_audit.json: only LoRA weights update).
# Usage (repo root): sbatch --gpus-per-node=N scripts/p2_pilots.sbatch UPDATES MODE1 [MODE2 ...]
#   e.g. sbatch --gpus-per-node=3 scripts/p2_pilots.sbatch 200 fp32 bf16 fp16
#SBATCH --account=bcrc-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=400g
#SBATCH --time=02:00:00
#SBATCH --job-name=p2-pilots
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
source scripts/env.sh
UPDATES=$1; shift
MODES=("$@")
OUT=results/p2_pilots/gpu-${SLURM_JOB_ID}
mkdir -p "$OUT/logs"
scripts/version_lock.sh "$OUT/environment.txt"
mkdir -p "$OUT/code_snapshot" && cp -r src scripts "$OUT/code_snapshot/" && git diff HEAD > "$OUT/code_snapshot/uncommitted.diff"
for i in "${!MODES[@]}"; do
  mode=${MODES[$i]}
  ( CUDA_VISIBLE_DEVICES=$i python src/pilot_runner.py --mode $mode --updates $UPDATES --out "$OUT" \
      > "$OUT/logs/$mode.log" 2>&1
    echo "$mode exit $?: $(tail -1 "$OUT/logs/$mode.log")" ) &
done
wait
python scripts/plot_pilots.py "$OUT"/P2PILOT-* --out "$OUT/figures"
echo DONE
