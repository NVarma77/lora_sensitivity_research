#!/bin/bash
# Person 3 — everything for the midterm in ONE interactive 4-GPU allocation (~30-40 min).
# Run on a GH200 compute node from the repo root, inside:
#   srun --account=bcrc-dtai-gh --partition=ghx4-interactive --nodes=1 --gpus-per-node=4 \
#        --cpus-per-task=64 --mem=400g --time=00:45:00 --pty bash
#   bash scripts/p3_tonight.sh
# Every step logs to results/p3_tonight_<job>/ and the script keeps going if one step fails.
# At the end it writes results/p3_tonight_<job>.tar.gz  -> send that file back.
set -u
source scripts/env.sh || { echo "env.sh failed — see docs/p3_distributed.md (own env)"; exit 1; }
export OMP_NUM_THREADS=8 NCCL_DEBUG=WARN
JOB=${SLURM_JOB_ID:-local}
R=results/p3_tonight_$JOB; mkdir -p $R/logs
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $R/STEPS.txt; }
run() {  # name cmd...
  local name=$1; shift
  log "START $name"; local t0=$SECONDS
  "$@" > $R/logs/$name.log 2>&1; local rc=$?
  log "END   $name rc=$rc $((SECONDS-t0))s :: $(grep -E 'PASS|FAIL|COMPLETED|FAILED|passed|failed|Error' $R/logs/$name.log | tail -1)"
}
TR="python -m torch.distributed.run --standalone --nnodes=1"

# 0. environment + topology
{ hostname; nvidia-smi -L; nvidia-smi topo -m; python -c "import torch,transformers,peft;print(torch.__version__,transformers.__version__,peft.__version__,torch.cuda.device_count())";
  python -c "import transformer_engine as te;print('TE',te.__version__)"; } > $R/logs/00_env.log 2>&1
scripts/version_lock.sh $R/environment.txt > /dev/null 2>&1
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw,clocks.sm --format=csv -l 2 > $R/system.csv &
SMI=$!; trap 'kill $SMI 2>/dev/null' EXIT
log "GPUs: $(nvidia-smi -L | wc -l)"

# 1. unit tests (whole repo)
run 01_pytest python -m pytest -q

# 2. P3-A gate: DDP correctness on 1/2/4 GH200 (NCCL)
for N in 1 2 4; do run 02_toy_N$N $TR --nproc_per_node=$N tests/ddp_toy_check.py --out $R/toy; done

# 3. P3-B gate: Qwen2.5-1.5B LoRA r16 BF16, 50 updates, 1 vs 4 GPUs (same global batch 32x512)
for N in 1 4; do
  run 03_smoke_N$N $TR --nproc_per_node=$N src/p3_ddp_runner.py --precision bf16 --lora_rank 16 --seed 0 \
      --max_updates 50 --eval_every 25 --out $R/smoke --run_id P3SMOKE-bf16-r16-s0-N$N
done
python scripts/p3_compare_runs.py $R/smoke/P3SMOKE-bf16-r16-s0-N1 $R/smoke/P3SMOKE-bf16-r16-s0-N4 --tol 2e-3 \
  > $R/smoke_parity.txt 2>&1; log "parity: $(tail -1 $R/smoke_parity.txt)"

# 4. X05 mini scaling: LoRA r16, BF16 / TE-BF16 / FP8 at 1/2/4 GPUs (10 warm-up + 40 timed updates)
for P in bf16 bf16_te fp8_te; do
  for N in 1 2 4; do
    run 04_x05_${P}_N$N $TR --nproc_per_node=$N src/p3_ddp_runner.py --precision $P --lora_rank 16 --seed 0 \
        --bench_warmup 10 --bench_timed 40 --out $R/x05 --run_id X05MINI-$P-N$N
  done
done
run 04_x05_bf16_N4_nosync $TR --nproc_per_node=4 src/p3_ddp_runner.py --precision bf16 --lora_rank 16 --seed 0 \
    --bench_warmup 10 --bench_timed 40 --no_sync_all --out $R/x05 --run_id X05MINI-bf16-N4-nosync
python scripts/p3_plot_scaling.py $R/x05 --out $R/x05_summary > $R/logs/05_plot.log 2>&1

# 5. X06 memory smoke: full fine-tuning, 1 GPU, 30 updates, BF16 then FP32 (skipped if time is short)
if [ $SECONDS -lt 1900 ]; then
  for P in bf16 fp32; do
    run 06_x06smoke_$P $TR --nproc_per_node=1 src/p3_ddp_runner.py --method full --precision $P --max_updates 30 \
        --eval_every 15 --out $R/x06 --run_id X06SMOKE-full-$P-s0
  done
else
  log "SKIP X06 smoke (time)"
fi

# 6. pack results (no checkpoints; small files only)
kill $SMI 2>/dev/null
tar czf $R.tar.gz $R
log "DONE -> $R.tar.gz  ($(du -h $R.tar.gz | cut -f1))"
