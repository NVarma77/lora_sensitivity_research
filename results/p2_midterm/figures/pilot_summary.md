# P2 precision pilots: summary

Stand-in runner (src/pilot_runner.py), not the shared X01 trainer. One seed, short budget: these show the precision switch works on the real model; they are not precision comparisons.

| mode | status | attempted | successful | skipped | final train loss (mean last 10) | final val loss | median tokens/s | peak alloc GiB | only LoRA updated | run id |
|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | COMPLETED | 200 | 200 | 0 | 2.3337 | 2.1899 | 6,242 | 29.1 | True | `P2PILOT-qwen15b-wt103-r16-fp32-s1-g1-3497606` |
| bf16 | COMPLETED | 200 | 200 | 0 | 2.3343 | 2.1908 | 28,661 | 24.8 | True | `P2PILOT-qwen15b-wt103-r16-bf16-s1-g1-3497564` |
| fp16 | COMPLETED | 200 | 200 | 0 | 2.3337 | 2.1899 | 29,217 | 24.8 | True | `P2PILOT-qwen15b-wt103-r16-fp16-s1-g1-3497564` |
| bf16_te | COMPLETED | 200 | 200 | 0 | 2.3343 | 2.1905 | 24,824 | 24.9 | True | `P2PILOT-qwen15b-wt103-r16-bf16_te-s1-g1-3497564` |
| fp8_te | COMPLETED | 200 | 200 | 0 | 2.3442 | 2.2007 | 20,965 | 24.9 | True | `P2PILOT-qwen15b-wt103-r16-fp8_te-s1-g1-3497564` |

Tokens/s = valid target tokens per effective update / wall step time (updates > 5, includes optimizer and Python overhead, excludes eval). FP32 has TF32 disabled.
