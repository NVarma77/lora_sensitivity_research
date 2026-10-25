# Precision policies (P2-A)

Owner: Person 2. Handbook refs: §5.2 (contract), §8.1–8.3 (policies), §13.1 (FP16 audit), §16.1 (task).

## Use

```python
from precision import build_precision_policy

policy = build_precision_policy(cfg.precision.mode)   # "fp32" | "bf16" | "fp16"
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

`fp8_te` and `bf16_te` are reserved mode names; they raise `NotImplementedError` until P2-B.

## Dtype policy matrix

| | fp32 | bf16 | fp16 |
|---|---|---|---|
| Autocast | off | `bfloat16` | `float16` |
| Matmul / linear compute | FP32 (TF32 off) | BF16 | FP16 |
| LayerNorm, softmax, log-softmax, loss | FP32 | FP32 (autocast fp32 list) | FP32 (autocast fp32 list) |
| Loss reduction | FP32 (`logits.float()` + sum) | FP32 | FP32 |
| Trainable params (master) | FP32 | FP32 | FP32 |
| Gradients stored on params | FP32 | FP32 | FP32 (scaled until unscale) |
| AdamW moments | FP32 | FP32 | FP32 |
| Loss scaling | none | none | dynamic `GradScaler`, init 2^16, x2 growth / 2000 steps, x0.5 backoff |
| Nonfinite gradients | update skipped, logged | update skipped, logged | update skipped by scaler, scale backed off, logged |
| TF32 (`allow_tf32`, matmul precision) | off / `highest` | off / `highest` | off / `highest` |

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
