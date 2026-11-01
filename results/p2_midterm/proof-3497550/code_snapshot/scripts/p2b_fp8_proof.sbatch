#!/bin/bash
# P2-B FP8 proof + smoke tests on one GH200 (handbook §8.5, X02, T08/T10).
# Usage: sbatch scripts/p2b_fp8_proof.sbatch   (from the repo root)
#SBATCH --account=bcrc-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=01:30:00
#SBATCH --job-name=p2b-fp8-proof
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
source scripts/env.sh
OUT=results/p2b_fp8/gpu-${SLURM_JOB_ID}
mkdir -p "$OUT"
scripts/version_lock.sh "$OUT/environment.txt"
mkdir -p "$OUT/code_snapshot" && cp -r src scripts "$OUT/code_snapshot/" && git diff HEAD > "$OUT/code_snapshot/uncommitted.diff"
python -m pytest -q 2>&1 | tee "$OUT/pytest.txt"

echo "== toy + Qwen-shape te.Linear proof"
python scripts/fp8_toy_proof.py --out "$OUT/toy" 2>&1 | tail -40

echo "== Qwen LoRA smokes with profiler (fp8_te, then bf16_te negative control)"
for MODE in fp8_te bf16_te; do
  python src/pilot_runner.py --mode $MODE --updates 30 --eval-every 15 --profile-update 10 --out "$OUT/smoke" \
    --run-id smoke-$MODE-$SLURM_JOB_ID 2>&1 | grep -vE "^\[train\]" | tail -8
done

echo "== nsys capture of one FP8 update (open the .nsys-rep in Nsight Systems for a GUI screenshot)"
nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop --trace=cuda,nvtx --sample=none \
  -o "$OUT/nsys_fp8_update" --force-overwrite true \
  python src/pilot_runner.py --mode fp8_te --updates 12 --eval-every 100 --profile-update 10 --nsys-range \
    --out "$OUT/smoke" --run-id nsys-fp8_te-$SLURM_JOB_ID 2>&1 | tail -5
nsys stats --report cuda_gpu_kern_sum --format csv --output "$OUT/nsys_fp8_update" "$OUT/nsys_fp8_update.nsys-rep" >/dev/null 2>&1
ls "$OUT"/nsys_fp8_update*

echo "== short fp32 / bf16 / fp16 smokes"
for MODE in fp16 bf16 fp32; do
  python src/pilot_runner.py --mode $MODE --updates 10 --eval-every 10 --out "$OUT/smoke" \
    --run-id smoke-$MODE-$SLURM_JOB_ID 2>&1 | grep -vE "^\[train\]" | tail -4
done

echo "== X02 first pass: one layer, one batch"
python scripts/layer_error_check.py --out "$OUT/x02" 2>&1 | tail -60
echo DONE
