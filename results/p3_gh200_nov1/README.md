# P3 first GH200 run — job 3503458, gh151, Nov 1 2026 (1 GPU, ghx4-interactive)
Env: own copy built with scripts/install_env_from_wheel.sh (torch 2.10.0+cu129, transformers 4.57.6, peft 0.21.2, TE 2.20.2, NCCL 2.27.5).
STEPS.txt includes two earlier attempts that failed at import (torchrun used system Python; shared env not group-readable) — fixed.

| Step | Result |
|---|---|
| pytest | 35 passed |
| toy DDP (NCCL, N=1) | PASS, checksum fe5790d89189 (identical to CPU) |
| LoRA r16 BF16 smoke, 50 updates | COMPLETED; val 2.4003 → 2.3033 (25) → 2.2840 (50); peak 24.83 GiB |
| X05 mini BF16 / TE BF16 / FP8, 40 timed updates | 504 / 699 / 824 ms median step; 32.5k / 23.4k / 19.9k tok/s; ~24.8–24.9 GiB |
| X06 smoke full FT BF16, 30 updates | COMPLETED; val 2.4003 → 2.1901; 0.65 s/step; peak 46.33 GiB |
| X06 smoke full FT FP32, 30 updates | COMPLETED; val 2.3997 → 2.1827; 3.75 s/step; peak 52.20 GiB |

Not yet run: 2- and 4-GPU (node unavailable). One seed, short budgets.
