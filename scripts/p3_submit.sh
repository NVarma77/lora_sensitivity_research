#!/bin/bash
# Person 3 job submitter (run from repo root on a DeltaAI login node).
#   bash scripts/p3_submit.sh toy          # P3-A gate: DDP correctness at 1/2/4 GPUs (~5 min each)
#   bash scripts/p3_submit.sh smoke        # P3-B gate: Qwen LoRA r16 BF16, 50 updates, 1 vs 4 GPUs
#   bash scripts/p3_submit.sh x06smoke     # full-FT memory check, 30 updates, FP32 + BF16, 1 GPU
#   bash scripts/p3_submit.sh x06          # full FT vs LoRA r16, FP32 + BF16, 200 updates, 1 GPU
#   bash scripts/p3_submit.sh x05          # scaling: LoRA r16, bf16 / bf16_te / fp8_te at 1/2/4 GPUs, 3 repeats
#   bash scripts/p3_submit.sh x01          # seed-2 share of the 12-cell matrix (after protocol freeze)
# Override partition for quick checks:  PART=ghx4-interactive bash scripts/p3_submit.sh toy
set -euo pipefail
PART=${PART:-ghx4}
OUT=results/p3_runs
sub() {  # N TIME NAME SCRIPT ARGS...
  local n=$1 t=$2 name=$3; shift 3
  sbatch -p "$PART" --gpus-per-node="$n" --cpus-per-task=$((16*n)) --mem=$((110*n))g --time="$t" \
         --job-name="$name" scripts/p3_ddp_job.sbatch "$@"
}
case "${1:-}" in
  toy)
    for N in 1 2 4; do sub $N 00:10:00 p3-toy-N$N tests/ddp_toy_check.py --out results/p3_toy_ddp; done ;;
  smoke)
    for N in 1 4; do
      sub $N 00:45:00 p3-smoke-N$N src/p3_ddp_runner.py --precision bf16 --lora_rank 16 --seed 0 \
          --max_updates 50 --eval_every 25 --out $OUT/smoke --run_id P3SMOKE-bf16-r16-s0-N$N
    done ;;
  x06smoke)
    for P in fp32 bf16; do
      sub 1 00:45:00 p3-x06smoke-$P src/p3_ddp_runner.py --method full --precision $P --max_updates 30 \
          --eval_every 15 --out $OUT/x06 --run_id X06SMOKE-full-$P-s0
    done ;;
  x06)
    for P in fp32 bf16; do
      sub 1 04:00:00 p3-x06-full-$P src/p3_ddp_runner.py --method full --lr 1e-5 --precision $P \
          --max_updates 200 --eval_every 50 --out $OUT/x06 --run_id X06-full-$P-s0
      sub 1 02:00:00 p3-x06-lora-$P src/p3_ddp_runner.py --method lora --lora_rank 16 --precision $P \
          --max_updates 200 --eval_every 50 --out $OUT/x06 --run_id X06-lora16-$P-s0
    done ;;
  x05)
    BASE="--method lora --lora_rank 16 --seed 0 --bench_warmup 20 --bench_timed 150 --out results/p3_x05"
    for P in ${PRECS:-bf16 bf16_te fp8_te}; do
      for N in 1 2 4; do
        for REP in 1 2 3; do
          sub $N 00:45:00 X05-$P-N$N-rep$REP src/p3_ddp_runner.py $BASE --precision $P --run_id X05-$P-N$N-rep$REP
        done
        sub $N 00:45:00 X05-$P-N$N-prof src/p3_ddp_runner.py $BASE --precision $P --profile --bench_timed 30 \
            --run_id X05-$P-N$N-profile
      done
    done
    for N in 2 4; do   # communication control: same work, no gradient all-reduce
      sub $N 00:45:00 X05-bf16-N$N-nosync src/p3_ddp_runner.py $BASE --precision bf16 --no_sync_all \
          --run_id X05-bf16-N$N-nosync
    done ;;
  x01)
    for P in fp32 bf16 fp16 fp8_te; do
      for R in 8 16 32; do
        sub 1 06:00:00 X01-$P-r$R-s2 src/p3_ddp_runner.py --method lora --lora_rank $R --precision $P --seed 2 \
            --max_updates 800 --eval_every 100 --out $OUT/x01 --run_id X01-$P-r$R-s2
      done
    done ;;
  *) sed -n '2,10p' "$0"; exit 1 ;;
esac
