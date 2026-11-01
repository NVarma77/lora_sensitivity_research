"""Acceptance tests for task P2-A (handbook §16.1, test T07).

Run on CPU by default; on a GPU node the same tests run on CUDA.
"""

import copy
import json
import math
from pathlib import Path

import pytest
import torch

from precision import (
    MODES,
    PrecisionProtocolError,
    build_precision_policy,
    record_as_row,
)
from toy_models import make_batches, make_toy, toy_loss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SRC = Path(__file__).resolve().parents[1] / "src"


def train_steps(policy, model, opt, batches, accum, max_norm=1.0, corrupt_grads=False):
    """Mirror of the handbook §7.3 loop; returns one StepRecord per effective batch."""
    records = []
    for i in range(0, len(batches), accum):
        micro = batches[i : i + accum]
        n_tokens = sum(b[:, 1:].numel() for b in micro)
        opt.zero_grad(set_to_none=True)
        for ids in micro:
            with policy.forward_context():
                loss_sum, _ = toy_loss(model, ids)
            policy.backward(loss_sum / n_tokens)
        if corrupt_grads:
            next(model.parameters()).grad[0].fill_(math.inf)
        policy.unscale_once_if_needed(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        records.append(policy.optimizer_step_and_update_scaler(opt))
    return records


def setup(mode, seed=0, **policy_kwargs):
    policy = build_precision_policy(mode, device_type=DEVICE, **policy_kwargs)
    model = make_toy(seed, DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
    return policy, model, opt


def flat_grads(model):
    return torch.cat([p.grad.flatten().float() for p in model.parameters()])


def rel_l2(x, ref):
    return float((x - ref).norm() / max(float(ref.norm()), 1e-12))


# ----- mode vocabulary --------------------------------------------------------


def test_mode_vocabulary_is_frozen():
    assert {"fp32", "bf16", "fp16", "bf16_te", "fp8_te"} == set(MODES)
    with pytest.raises(ValueError):
        build_precision_policy("fp8")


# ----- same API, all modes ----------------------------------------------------


@pytest.mark.parametrize("mode", ["fp32", "bf16", "fp16"])
def test_identical_api_runs_and_counts_updates(mode):
    policy, model, opt = setup(mode)
    records = train_steps(policy, model, opt, make_batches(6, device=DEVICE), accum=2)
    assert len(records) == 3
    assert all(r.did_step and r.grad_finite for r in records)
    assert [r.successful_update for r in records] == [1, 2, 3]
    assert policy.attempted_steps == policy.successful_updates == 3
    assert policy.skipped_steps == 0
    for p in model.parameters():
        assert p.dtype == torch.float32, "master weights must stay FP32"
        assert torch.isfinite(p).all()


def test_scaler_only_in_fp16():
    assert build_precision_policy("fp32", device_type=DEVICE).scaler is None
    assert build_precision_policy("bf16", device_type=DEVICE).scaler is None
    assert build_precision_policy("fp16", device_type=DEVICE).scaler is not None
    assert build_precision_policy("bf16", device_type=DEVICE).current_scale() is None


def test_autocast_actually_changes_compute_dtype():
    # Guards against "strangely identical curves" (handbook §18.7).
    model = make_toy(0, DEVICE)
    ids = make_batches(1, device=DEVICE)[0]
    seen = {}
    for mode, expected in [("fp32", torch.float32), ("bf16", torch.bfloat16), ("fp16", torch.float16)]:
        policy = build_precision_policy(mode, device_type=DEVICE)
        with policy.forward_context():
            seen[mode] = model.up(model.embed(ids)).dtype
        assert seen[mode] == expected, (mode, seen[mode])


# ----- accumulation and unscale (T07) -----------------------------------------


def test_fp16_unscales_once_after_full_accumulation(monkeypatch):
    policy, model, opt = setup("fp16")
    calls = []
    real_unscale = policy.scaler.unscale_
    monkeypatch.setattr(policy.scaler, "unscale_", lambda o: (calls.append(1), real_unscale(o)))

    # Two effective batches of four microbatches each.
    train_steps(policy, model, opt, make_batches(8, device=DEVICE), accum=4)
    assert len(calls) == 2


def test_backward_after_unscale_is_rejected():
    policy, model, opt = setup("fp16")
    ids = make_batches(1, device=DEVICE)[0]
    with policy.forward_context():
        loss, n = toy_loss(model, ids)
    policy.backward(loss / n)
    policy.unscale_once_if_needed(opt)
    with policy.forward_context():
        loss, n = toy_loss(model, ids)
    with pytest.raises(PrecisionProtocolError):
        policy.backward(loss / n)


def test_fp16_unscaled_grads_match_fp32_reference():
    batches = make_batches(4, device=DEVICE)
    grads = {}
    for mode in ("fp32", "fp16"):
        policy, model, opt = setup(mode)
        opt.zero_grad(set_to_none=True)
        n_tokens = sum(b[:, 1:].numel() for b in batches)
        for ids in batches:
            with policy.forward_context():
                loss, _ = toy_loss(model, ids)
            policy.backward(loss / n_tokens)
        policy.unscale_once_if_needed(opt)
        grads[mode] = flat_grads(model)
    # FP16 rounding gives small differences; a missing or double unscale
    # would be off by a factor of the scale (65536).
    assert rel_l2(grads["fp16"], grads["fp32"]) < 1e-2


def test_scaler_rescues_fp16_gradient_underflow():
    # A gradient of 1e-9 is below the smallest FP16 subnormal (~6e-8).
    torch.manual_seed(0)
    layer = torch.nn.Linear(8, 8).to(DEVICE)
    x = torch.randn(4, 8, device=DEVICE)

    def grad_with(scale):
        layer.zero_grad(set_to_none=True)
        with torch.autocast(DEVICE, dtype=torch.float16):
            out = layer(x)
        (out.float().sum() * 1e-9 * scale).backward()
        return layer.weight.grad.clone() / scale

    unscaled = grad_with(1.0)
    scaled = grad_with(2.0**16)
    assert unscaled.abs().max() == 0, "expected FP16 underflow without loss scaling"
    reference = x.sum(0).expand(8, 8) * 1e-9
    assert rel_l2(scaled, reference) < 1e-2


# ----- nonfinite handling -----------------------------------------------------


def test_fp16_overflow_skips_update_and_backs_off_scale():
    policy, model, opt = setup("fp16", init_scale=1024.0)
    before = copy.deepcopy(model.state_dict())
    (record,) = train_steps(policy, model, opt, make_batches(2, device=DEVICE), accum=2, corrupt_grads=True)

    assert not record.did_step and record.optimizer_step_skipped
    assert not record.grad_finite
    assert record.successful_update == 0 and record.attempted_step == 1
    assert record.scale_before == 1024.0 and record.scale_after == 512.0
    assert policy.consecutive_skips == 1
    assert len(opt.state) == 0, "optimizer state must not be touched on a skipped step"
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k]), f"{k} changed on a skipped step"


@pytest.mark.parametrize("mode", ["fp32", "bf16"])
def test_nonfinite_grads_never_step_without_scaler(mode):
    policy, model, opt = setup(mode)
    before = copy.deepcopy(model.state_dict())
    (record,) = train_steps(policy, model, opt, make_batches(2, device=DEVICE), accum=2, corrupt_grads=True)
    assert not record.did_step and not record.grad_finite
    assert policy.skipped_steps == 1 and policy.successful_updates == 0
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k])


def test_skip_then_recover_counts():
    policy, model, opt = setup("fp16", init_scale=1024.0)
    batches = make_batches(4, device=DEVICE)
    train_steps(policy, model, opt, batches[:2], accum=2, corrupt_grads=True)
    (record,) = train_steps(policy, model, opt, batches[2:], accum=2)
    assert record.did_step
    assert (record.attempted_step, record.successful_update) == (2, 1)
    assert policy.consecutive_skips == 0 and policy.skipped_steps == 1


def test_fused_optimizer_rejected_for_fp16():
    policy = build_precision_policy("fp16", device_type=DEVICE)
    model = make_toy(0, DEVICE)
    try:
        opt = torch.optim.AdamW(model.parameters(), fused=True)
    except RuntimeError:
        pytest.skip("fused AdamW unavailable on this device")
    with pytest.raises(ValueError, match="fused"):
        policy.optimizer_step_and_update_scaler(opt)


# ----- TF32, checkpointing, manifest ------------------------------------------


@pytest.mark.parametrize("mode", ["fp32", "bf16", "fp16"])
def test_tf32_disabled_for_every_mode(mode):
    build_precision_policy(mode, device_type=DEVICE)
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False
    assert torch.get_float32_matmul_precision() == "highest"


def test_state_dict_round_trip_preserves_scaler_and_counters():
    policy, model, opt = setup("fp16", init_scale=1024.0, growth_interval=3)
    batches = make_batches(8, device=DEVICE)
    train_steps(policy, model, opt, batches[:2], accum=2, corrupt_grads=True)
    train_steps(policy, model, opt, batches[2:], accum=2)
    state = policy.state_dict()

    resumed = build_precision_policy("fp16", device_type=DEVICE, init_scale=1024.0, growth_interval=3)
    resumed.load_state_dict(state)
    assert resumed.current_scale() == policy.current_scale()
    assert resumed.scaler.state_dict() == policy.scaler.state_dict()
    assert (resumed.attempted_steps, resumed.successful_updates, resumed.skipped_steps) == (4, 3, 1)

    with pytest.raises(ValueError):
        build_precision_policy("bf16", device_type=DEVICE).load_state_dict(state)


@pytest.mark.parametrize("mode", ["fp32", "bf16", "fp16"])
def test_manifest_records_policy(mode, tmp_path):
    policy, model, opt = setup(mode)
    train_steps(policy, model, opt, make_batches(2, device=DEVICE), accum=2)
    m = policy.write_manifest(tmp_path / "precision.json", model, opt)

    assert json.loads((tmp_path / "precision.json").read_text()) == m
    assert m["mode"] == mode
    assert m["autocast_enabled"] == (mode != "fp32")
    assert (m["grad_scaler"] is not None) == (mode == "fp16")
    assert m["tf32"]["cuda_matmul_allow_tf32"] is False
    assert m["versions"]["torch"] == torch.__version__
    assert set(m["parameter_dtypes"]["trainable"]) == {"torch.float32"}
    assert m["optimizer"]["state_dtypes"]["exp_avg:torch.float32"] > 0


def test_record_row_uses_nan_not_zero_for_missing_scale():
    policy, model, opt = setup("bf16")
    (record,) = train_steps(policy, model, opt, make_batches(1, device=DEVICE), accum=1)
    row = record_as_row(record)
    assert math.isnan(row["scale_before"]) and math.isnan(row["scale_after"])


# ----- single code path (handbook §1.3 rule 8, §5.2) --------------------------


def test_no_direct_amp_calls_outside_precision_module():
    offenders = []
    for path in SRC.rglob("*.py"):
        if path.name == "precision.py":
            continue
        text = path.read_text()
        # fp8_backend.py defines the te.autocast wrapper; only precision.py may call it.
        if path.name != "fp8_backend.py" and ("autocast(" in text or "GradScaler(" in text):
            offenders.append(path.name)
    assert offenders == []
