#!/bin/bash
# P2 midterm bundle on one GH200, sequential (the interactive QOS allows one job per user and
# 1-GPU requests start fastest):
#   1. FP8 proof chain: toy/Qwen-shape te.Linear proof, profiled fp8_te + bf16_te smokes, nsys, X02 layer check
#   2. pilots: bf16, fp16, bf16_te, fp8_te (= FP8+LoRA test run; param_audit.json), fp32 (slowest, last)
# Usage (repo root): sbatch scripts/p2_midterm.sbatch UPDATES STAGE
#   STAGE = proof | pilots | fp32 | all   (short stages backfill much faster on ghx4-interactive)
#SBATCH --account=bcrc-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110g
#SBATCH --time=00:45:00
#SBATCH --job-name=p2-midterm
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
source scripts/env.sh
UPDATES=${1:-200}
STAGE=${2:-all}
ROOT=results/p2_midterm/${STAGE}-${SLURM_JOB_ID}
PILOTS=$ROOT/pilots PROOF=$ROOT/fp8_proof
mkdir -p "$PILOTS/logs" "$PROOF"
scripts/version_lock.sh "$ROOT/environment.txt"
mkdir -p "$ROOT/code_snapshot" && cp -r src scripts "$ROOT/code_snapshot/" && git diff HEAD > "$ROOT/code_snapshot/uncommitted.diff"
python -m pytest -q > "$ROOT/pytest.txt" 2>&1; tail -1 "$ROOT/pytest.txt"

pilot() {  # GPU MODE...
  local gpu=$1; shift
  for mode in "$@"; do
    CUDA_VISIBLE_DEVICES=$gpu python src/pilot_runner.py --mode $mode --updates $UPDATES --out "$PILOTS" \
      > "$PILOTS/logs/$mode.log" 2>&1
    echo "[pilot] $mode exit $?: $(tail -1 "$PILOTS/logs/$mode.log")"
  done
}

proof() {
  export CUDA_VISIBLE_DEVICES=0
  python scripts/fp8_toy_proof.py --out "$PROOF/toy" > "$PROOF/toy.log" 2>&1
  echo "[proof] toy exit $?: $(grep -m1 Verdict "$PROOF/toy/kernel_evidence.md" 2>/dev/null)"
  for MODE in fp8_te bf16_te; do
    python src/pilot_runner.py --mode $MODE --updates 30 --eval-every 15 --profile-update 10 --out "$PROOF/smoke" \
      --run-id smoke-$MODE > "$PROOF/smoke-$MODE.log" 2>&1
    echo "[proof] smoke $MODE exit $?: $(tail -1 "$PROOF/smoke-$MODE.log")"
  done
  nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop --trace=cuda,nvtx --sample=none \
    -o "$PROOF/nsys_fp8_update" --force-overwrite true \
    python src/pilot_runner.py --mode fp8_te --updates 12 --eval-every 100 --profile-update 10 --nsys-range \
      --out "$PROOF/smoke" --run-id nsys-fp8_te > "$PROOF/nsys.log" 2>&1
  echo "[proof] nsys exit $?"
  nsys stats --report cuda_gpu_kern_sum --format csv --output "$PROOF/nsys_fp8_update" \
    "$PROOF/nsys_fp8_update.nsys-rep" >> "$PROOF/nsys.log" 2>&1
  python scripts/fp8_profile_report.py "$PROOF/smoke/smoke-fp8_te" "$PROOF/smoke/smoke-bf16_te" --out "$PROOF" \
    --nsys "$(ls "$PROOF"/nsys_fp8_update*cuda_gpu_kern_sum*.csv 2>/dev/null | head -1)" > "$PROOF/report.log" 2>&1
  echo "[proof] report exit $?: $(grep Verdict "$PROOF/X02_kernel_evidence.md" 2>/dev/null)"
  python scripts/layer_error_check.py --out "$PROOF/x02" > "$PROOF/x02.log" 2>&1
  echo "[proof] x02 exit $?"
}

case $STAGE in
  proof) proof ;;
  pilots) pilot 0 bf16 fp16 bf16_te fp8_te ;;
  fp32) pilot 0 fp32 ;;
  all) proof; pilot 0 bf16 fp16 bf16_te fp8_te fp32 ;;
esac
if ls "$PILOTS"/P2PILOT-* >/dev/null 2>&1; then
  python scripts/plot_pilots.py "$PILOTS"/P2PILOT-* --out "$PILOTS/figures" > "$PILOTS/figures.log" 2>&1
  echo "[plots] exit $?"
fi
echo DONE
