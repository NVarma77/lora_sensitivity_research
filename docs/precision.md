# Precision policies (P2-A, P2-B)

Owner: Person 2. Handbook refs: §5.2 (contract), §8 (policies), §13.1 (FP16 audit), §16.1–16.2 (tasks).

## Use

```python
from precision import build_precision_policy

policy = build_precision_policy(cfg.precision.mode)   # "fp32" | "bf16" | "fp16" | "bf16_te" | "fp8_te"
for effective_batch in stream:
    opt.zero_grad(set_to_none=True)
    for micro in microbatches:
        with policy.forward_context():                # forward + loss only
            loss_sum, n = forward_loss(model, micro)
        policy.backward(loss_sum / global_valid_tokens)
    policy.unscale_once_if_needed(opt)                # once, after ALL microbatches
    grad_norm = clip_grad_norm_(trainable, 1.0)       # sees true gradients
    rec = policy.optimizer_step_and_update_scaler(opt)
    if rec.did_step:
        scheduler.step()                              # never on a skipped update
    log_train_row(record_as_row(rec), grad_norm=grad_norm)

policy.write_manifest(run_dir / "precision.json", model, opt)
checkpoint["precision"] = policy.state_dict()          # includes FP16 scaler state
```

Rules for callers:
- Do not call `torch.autocast` or `GradScaler` anywhere else (a test enforces this for `src/`).
- Calling `backward()` after `unscale_once_if_needed()` in the same effective batch raises.
- Do not call `model.half()` / `model.bfloat16()`; master weights and AdamW moments stay FP32.
- Use a non-fused optimizer with FP16 (fused AdamW hides skipped steps; rejected with an error).
- Under DDP (P3): gradients are all-reduced before unscale, so every rank sees the same
  inf/NaN result and skips together. Do not unscale inside `no_sync()` microbatches.

TE modes need the model's linears converted to `te.Linear` *before* LoRA is attached
(`fp8_backend.convert_linears_to_te(model)`; P1's `te_conversion.py` should take this over).
PEFT 0.21 then wraps them in `peft.tuners.lora.te.TeLinear`. `bf16_te` is the backend-matched
control for `fp8_te`: same converted modules, same weights, FP8 recipe off.

## Dtype policy matrix

| | fp32 | bf16 | fp16 | bf16_te | fp8_te |
|---|---|---|---|---|---|
| Autocast | off | `bfloat16` | `float16` | `bfloat16` + TE (FP8 off) | `bfloat16` + TE FP8 |
| Backbone decoder linears (q/k/v/o/gate/up/down) | FP32 (TF32 off) | BF16 | FP16 | te.Linear BF16 | te.Linear FP8: E4M3 fwd, E5M2 grads (HYBRID, delayed scaling, amax history 16, max) |
| LoRA adapters (lora_A/lora_B), lm_head, attention SDPA | FP32 | BF16 | FP16 | BF16 | BF16 |
| RMSNorm/LayerNorm, softmax, log-softmax, loss | FP32 | FP32 (autocast fp32 list) | FP32 (autocast fp32 list) | FP32 | FP32 |
| Loss reduction | FP32 (`logits.float()` + sum) | FP32 | FP32 | FP32 | FP32 |
| Trainable params (master) | FP32 | FP32 | FP32 | FP32 | FP32 |
| Gradients stored on params | FP32 | FP32 | FP32 (scaled until unscale) | FP32 | FP32 |
| AdamW moments | FP32 | FP32 | FP32 | FP32 | FP32 |
| Loss scaling | none | none | dynamic `GradScaler`, init 2^16, x2 growth / 2000 steps, x0.5 backoff | none | none (TE per-tensor FP8 scales) |
| Nonfinite gradients | update skipped, logged | update skipped, logged | update skipped by scaler, scale backed off, logged | skipped, logged | skipped, logged |
| TF32 (`allow_tf32`, matmul precision) | off / `highest` | off / `highest` | off / `highest` | off | off |

TF32 is disabled globally for every mode so the autocast dtype is the only arithmetic
difference between policies. Frozen backbone dtype depends on how `model.py` loads the
checkpoint (see open decision 1).

## Logged fields (train.csv)

`attempted_step, successful_update, optimizer_step_skipped, grad_finite, scale_before,
scale_after` come from `StepRecord`; `grad_norm` comes from the caller's clip. Skips are
detected from an optimizer step hook (the optimizer actually ran), not inferred from the
scale shrinking. Scale fields are NaN (not 0) for modes without a scaler.

## Tests

```bash
module load python/miniforge3_pytorch/2.10.0
python -m pytest -q                       # CPU, login node OK
sbatch scripts/p2a_gpu_check.sbatch       # 1 GPU, ~5 min: tests + parity table on GH200
```

`tests/test_precision.py` covers: identical API across modes, autocast really changes
compute dtype, unscale exactly once per effective batch, FP16 grads match FP32 after
unscale, loss scaling rescues FP16 underflow, forced overflow skips and backs off with
parameters and optimizer state untouched, FP32/BF16 never step on nonfinite grads,
skip-then-recover counters, TF32 off, scaler/counter checkpoint round trip, manifest
contents, NaN-not-zero logging, and no direct AMP calls outside `precision.py`.

`scripts/precision_parity.py` trains the toy LM identically in all three modes and writes
`parity_table.md` (fixed-batch loss and gradient error vs FP32, final train/val loss,
attempted/successful/skipped updates), `parity_steps.csv` and `manifest_<mode>.json`.

## Open decisions (need team ADR before the Nov 5 freeze)

1. Frozen base weights: load Qwen (stored BF16) as FP32 for all modes and let autocast cast
   down (proposed; ~6 GB), recording source dtype. Affects `model.py` (P1).
2. FP32/BF16 nonfinite gradients: currently skip and log. Agree when this becomes
   `FAILED_NUMERICAL`.
3. FP16 instability rule (pre-registered, §8.3): e.g. K consecutive skips or skipped fraction
   above X% of attempts. `policy.consecutive_skips` / `skipped_steps` expose the counters.
4. FP16 attempt budget: how many extra attempts are allowed to reach 800 successful updates.
5. TF32 off for all modes (implemented; confirm).

## Environment (DeltaAI GH200, built 2026-10-31)

`scripts/env.sh` activates `/projects/bcrc/nvarma2/envs/dipdesk`: a venv on top of
`python/miniforge3_pytorch/2.10.0` (torch 2.10.0+cu129) with transformers 4.57.6, peft 0.21.2 and
Transformer Engine 2.20.2 built from source for sm_90. Building TE needed three fixes:
cuDNN headers via `CUDNN_PATH`, HPC SDK `math_libs/12.9` on `CPATH`/`LIBRARY_PATH`, and cuDNN ≥ 9.12
(TE static assert). **Deviation:** the venv overrides torch's pinned cuDNN 9.10.2 with
`nvidia-cudnn-cu12==9.27.0.42` (installed `--no-deps`). Every mode runs on this same stack; it is
recorded in each job's `environment.txt`. Reproduce the build with the commands in
`scripts/build_env.sbatch` plus the paths above.

## Stand-in runner

`src/pilot_runner.py` is a P2-only Qwen2.5-1.5B + LoRA loop used for the midterm pilots because
P1's trainer was not available. Its outputs are pilots, not X01 cells. Delete it once P1's
`train.py` can call `build_precision_policy` (see ADR-001).
