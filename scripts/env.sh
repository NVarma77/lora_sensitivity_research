# Shared environment for P2 jobs (source from the repo root). Pinned stack, see results/*/environment.txt.
# Defaults are the shared bcrc env (needs delta_bcrc group). Without that access, build your own with
# scripts/install_env_from_wheel.sh and set DIPDESK_VENV (and optionally HF_HOME / DIPDESK_CACHE).
module load python/miniforge3_pytorch/2.10.0 >/dev/null 2>&1
DIPDESK_VENV=${DIPDESK_VENV:-/projects/bcrc/nvarma2/envs/dipdesk}
if [ "$DIPDESK_VENV" = /projects/bcrc/nvarma2/envs/dipdesk ]; then
    _cudnn_lib=/projects/bcrc/nvarma2/envs/cudnn/lib
else
    _cudnn_lib=$DIPDESK_VENV/lib/python3.12/site-packages/nvidia/cudnn/lib
fi
if [ ! -r "$DIPDESK_VENV/bin/activate" ]; then
    echo "env.sh: cannot read $DIPDESK_VENV/bin/activate (no delta_bcrc access?)." >&2
    echo "        Run scripts/install_env_from_wheel.sh and export DIPDESK_VENV=<its dir>." >&2
    return 1 2>/dev/null || exit 1
fi
source "$DIPDESK_VENV/bin/activate"
export HF_HOME=${HF_HOME:-/work/nvme/bcrc/nvarma2/hf_cache} HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export DIPDESK_CACHE=${DIPDESK_CACHE:-/work/nvme/bcrc/nvarma2/dipdesk_cache}
export LD_LIBRARY_PATH=$_cudnn_lib:${LD_LIBRARY_PATH:-}
unset _cudnn_lib
export TOKENIZERS_PARALLELISM=false
# torch 2.10 needs its bundled NCCL (ncclCommShrink); the HPC SDK / aws-ofi NCCL paths that some
# shells inherit are older and break `import torch`. Put torch's copy first.
export LD_LIBRARY_PATH=/sw/user/python/miniforge3-pytorch-2.10.0/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
