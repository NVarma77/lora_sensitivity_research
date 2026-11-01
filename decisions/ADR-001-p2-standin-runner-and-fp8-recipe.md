# ADR-001: P2 stand-in runner and pilot FP8 recipe

- Date: 2026-10-31
- Owner: Person 2. Reviewers: Person 1 (trainer/TE boundary), Person 3 (launch/metrics).
- Status: proposed (needs team sign-off before the Nov 5 protocol freeze, handbook §4.3)

## Context
The midterm (end of Oct 26–Nov 1) needs real Qwen2.5-1.5B + LoRA pilots in FP32/BF16/FP16 and an
FP8+LoRA test run. P1's shared trainer (data.py/model.py/lora.py/train.py, te_conversion.py) was not
available to P2 on Oct 31.

## Decision
1. P2 uses `src/pilot_runner.py`, a minimal stand-in that follows §4.1/§5.4 where cheap. Its runs are
   labelled `P2PILOT-*` and are not X01 cells. It is deleted when P1's trainer can call
   `build_precision_policy`. Deviations from §4.1: training data = article-ordered prefix (8M tokens)
   of WikiText-103 train; validation = first 128 blocks; microbatch 8 x accumulation 4 (global 32);
   200-update pilot budget.
2. TE layer conversion prototype lives in `fp8_backend.convert_linears_to_te`: all 196 decoder
   nn.Linear (q/k/v/o/gate/up/down x 28) → te.Linear, weights copied, lm_head (tied) left eager.
   LoRA attaches through PEFT 0.21 `TeLinear`; adapters run in BF16 autocast, not FP8 (§8.4 split).
3. Pilot FP8 recipe: `DelayedScaling(Format.HYBRID, amax_history_len=16, amax_compute_algo="max",
   margin=0)` via `transformer_engine.pytorch.autocast`, nested inside torch BF16 autocast.
4. Frozen base loaded as FP32 in every mode (open decision 1 in docs/precision.md).

## Options considered
Wait for P1 (misses midterm); toy/submodule-only FP8 (handbook §16.6 contingency; does not give
Qwen LoRA curves); fork P1's code (forbidden by §16.2).

## Impact on comparability
None on X01 if P1's trainer replaces the stand-in before Nov 5. If the team instead adopts the
stand-in, the deviations above must be resolved or frozen in a protocol revision.
