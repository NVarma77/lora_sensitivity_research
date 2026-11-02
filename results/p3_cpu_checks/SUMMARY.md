# P3 CPU (Gloo) verification — 2026-11-02 — torch 2.15.0+cu132, transformers 5.22.1, peft 0.21.2

## pytest (whole repo, incl. P2 precision tests)
35 passed in 2.74s

## tests/ddp_toy_check.py (T11/T12/T13, fp64 toy model)
N=1 A=16 PASS=True max|dgrad|=2.2e-15 checksum=fe5790d89189
N=2 A=8 PASS=True max|dgrad|=2.2e-15 checksum=fe5790d89189
N=4 A=4 PASS=True max|dgrad|=2.2e-15 checksum=fe5790d89189

## p3_ddp_runner.py, tiny random Qwen2 + PEFT LoRA r8 (q,v), synthetic tokens, team precision.py, 1 vs 4 workers
PASS  tiny-fp32-N4  N=4  max|Δtrain|=9.23e-08 over 10 updates  max|Δval|=2.26e-07 over 3 evals  status=COMPLETED ranks_agree=True
PASS  tiny-bf16-N4  N=4  max|Δtrain|=0.00e+00 over 10 updates  max|Δval|=1.33e-07 over 3 evals  status=COMPLETED ranks_agree=True
PASS  tiny-fp16-N4  N=4  max|Δtrain|=4.74e-06 over 10 updates  max|Δval|=1.92e-06 over 3 evals  status=COMPLETED ranks_agree=True

Not verified here: CUDA/NCCL, TE (bf16_te/fp8_te), real Qwen2.5-1.5B, timings (CPU timings are meaningless).
