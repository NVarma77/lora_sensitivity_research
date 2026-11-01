# Shared environment for P2 jobs (source from the repo root). Pinned stack, see results/*/environment.txt.
module load python/miniforge3_pytorch/2.10.0 >/dev/null 2>&1
source /projects/bcrc/nvarma2/envs/dipdesk/bin/activate
export HF_HOME=/work/nvme/bcrc/nvarma2/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export DIPDESK_CACHE=/work/nvme/bcrc/nvarma2/dipdesk_cache
export LD_LIBRARY_PATH=/projects/bcrc/nvarma2/envs/cudnn/lib:${LD_LIBRARY_PATH:-}
export TOKENIZERS_PARALLELISM=false
# torch 2.10 needs its bundled NCCL (ncclCommShrink); the HPC SDK / aws-ofi NCCL paths that some
# shells inherit are older and break `import torch`. Put torch's copy first.
export LD_LIBRARY_PATH=/sw/user/python/miniforge3-pytorch-2.10.0/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
