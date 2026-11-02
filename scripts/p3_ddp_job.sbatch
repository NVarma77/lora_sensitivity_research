#!/bin/bash
# Person 3 generic DeltaAI launcher: ONE torchrun per node, N workers = GPUs requested (1/2/4).
# Uses the team env (scripts/env.sh). GPU count, CPUs and memory are chosen at submit time, e.g.
#   sbatch --gpus-per-node=4 --cpus-per-task=64 --mem=440g scripts/p3_ddp_job.sbatch src/p3_ddp_runner.py --precision bf16
#   sbatch --gpus-per-node=2 scripts/p3_ddp_job.sbatch tests/ddp_toy_check.py
# Partition: ghx4 (batch) by default; short checks can use  sbatch -p ghx4-interactive ...  (1 job/user, <=45 min).
#SBATCH --account=bcrc-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110g
#SBATCH --time=00:45:00
#SBATCH --job-name=p3-ddp
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
source scripts/env.sh || exit 1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} NCCL_DEBUG=${NCCL_DEBUG:-WARN}

NGPU="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
SCRIPT="$1"; shift
echo "== job ${SLURM_JOB_ID} on $(hostname)  GPUs=${NGPU}  ${SCRIPT} $*"
nvidia-smi -L
nvidia-smi topo -m > "logs/topo-${SLURM_JOB_ID}.txt" 2>&1 || true
scripts/version_lock.sh "logs/environment-${SLURM_JOB_ID}.txt" > /dev/null 2>&1 || true

# GPU telemetry (utilisation, memory, power, SM clock) every 2 s -> logs/system-<job>.csv
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw,clocks.sm \
           --format=csv -l 2 > "logs/system-${SLURM_JOB_ID}.csv" &
SMI_PID=$!
trap 'kill ${SMI_PID} 2>/dev/null || true' EXIT

srun python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${NGPU}" "${SCRIPT}" "$@"
