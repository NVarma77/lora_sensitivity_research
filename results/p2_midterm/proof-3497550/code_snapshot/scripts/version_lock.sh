#!/bin/bash
# Handbook §8.5 step 10: pip freeze, module list, nvidia-smi, CUDA/TE API. Usage: version_lock.sh OUTFILE
{
  echo "# $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-none}"
  echo "## module list"; module list 2>&1
  echo "## nvidia-smi"; nvidia-smi
  echo "## nvcc"; nvcc --version 2>&1 | tail -2
  echo "## python stack"
  python - <<'PY'
import torch, transformers, peft
print("torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version())
print("transformers", transformers.__version__, "peft", peft.__version__)
try:
    import transformer_engine as te, transformer_engine.pytorch as tep, inspect
    print("transformer_engine", te.__version__)
    print("te.autocast", inspect.signature(tep.autocast) if hasattr(tep, "autocast") else "absent")
    print("fp8 support", tep.fp8.check_fp8_support())
except Exception as e:
    print("transformer_engine import failed:", repr(e))
print("device", torch.cuda.get_device_name() if torch.cuda.is_available() else None)
PY
  echo "## pip freeze"; pip freeze 2>/dev/null
} > "$1" 2>&1
