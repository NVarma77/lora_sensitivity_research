# P2-A toy parity (cuda, NVIDIA GH200 120GB, torch 2.10.0+cu129, seed 0, 200 updates x 4 microbatches)

| mode | init loss | init loss Δ vs FP32 | init grad rel L2 vs FP32 | init grad cosine | final train loss | val loss | val Δ vs FP32 | attempted | successful | skipped | final scale |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | 5.687815 | +0.00e+00 | 0.00e+00 | 1.000000 | 4.184103 | 4.184813 | +0.00e+00 | 200 | 200 | 0 | n/a |
| bf16 | 5.687879 | +6.34e-05 | 3.20e-03 | 0.999995 | 4.183995 | 4.184664 | -1.49e-04 | 200 | 200 | 0 | n/a |
| fp16 | 5.687819 | +3.81e-06 | 3.97e-04 | 1.000000 | 4.184106 | 4.184804 | -8.94e-06 | 200 | 200 | 0 | 65536.0 |

Toy model only: this checks the precision switch, not Qwen/LoRA behaviour.
