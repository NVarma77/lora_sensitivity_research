"""Precision policies for the shared DipDesk trainer (handbook §5.2, §8, §13.1).

Every precision decision in the training loop goes through a PrecisionPolicy.
No other module should call torch.autocast or GradScaler directly.

Intended use inside one effective (global) batch, following handbook §7.3:

    policy = build_precision_policy(config.precision)
    for effective_batch in stream:
        opt.zero_grad(set_to_none=True)
        for micro in microbatches:
            with policy.forward_context():
                loss = compute_loss(model, micro)
            policy.backward(loss)
        policy.unscale_once_if_needed(opt)          # after ALL microbatches
        grad_norm = clip_grad_norm_(params, 1.0)    # on true (unscaled) grads
        step = policy.optimizer_step_and_update_scaler(opt)
        if step.did_step:
            scheduler.step()
"""

from __future__ import annotations

import contextlib
import json
import math
import platform
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

# torch AMP modes (P2-A) and Transformer Engine modes (P2-B). bf16_te is the
# backend-matched control for fp8_te: same converted te.Linear modules, FP8 off
# (handbook §3.5, §8.5 step 8). Both TE modes run everything else under BF16
# autocast, so the TE FP8 recipe is the only difference between them.
TORCH_AMP_MODES = ("fp32", "bf16", "fp16")
TE_MODES = ("bf16_te", "fp8_te")
IMPLEMENTED_MODES = TORCH_AMP_MODES + TE_MODES
MODES = IMPLEMENTED_MODES

_AUTOCAST_DTYPE = {
    "fp32": None,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "bf16_te": torch.bfloat16,
    "fp8_te": torch.bfloat16,
}


class PrecisionProtocolError(RuntimeError):
    """The training loop called the policy in an order that breaks the protocol."""


@dataclass
class StepRecord:
    """Per-attempt telemetry; field names match train.csv (handbook §5.4, §13.1)."""

    attempted_step: int
    successful_update: int
    did_step: bool
    optimizer_step_skipped: bool
    grad_finite: bool
    scale_before: float | None
    scale_after: float | None


class PrecisionPolicy:
    def __init__(
        self,
        mode: str,
        device_type: str = "cuda",
        allow_tf32: bool = False,
        init_scale: float = 2.0**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        fp8_recipe=None,
    ):
        if mode not in IMPLEMENTED_MODES:
            raise ValueError(f"unknown precision mode {mode!r}; expected one of {MODES}")

        self.mode = mode
        self.device_type = device_type
        self.autocast_dtype = _AUTOCAST_DTYPE[mode]
        self.allow_tf32 = allow_tf32
        _configure_tf32(allow_tf32)

        self.uses_te = mode in TE_MODES
        self.fp8_enabled = mode == "fp8_te"
        self.fp8_recipe_config = None
        self._te_recipe = None
        if self.uses_te:
            import fp8_backend

            fp8_backend.require_te()
            if self.fp8_enabled:
                self.fp8_recipe_config = fp8_recipe or fp8_backend.FP8RecipeConfig()
                self._te_recipe = fp8_backend.build_recipe(self.fp8_recipe_config)

        self.scaler = None
        if mode == "fp16":
            self.scaler = torch.amp.GradScaler(
                device_type,
                init_scale=init_scale,
                growth_factor=growth_factor,
                backoff_factor=backoff_factor,
                growth_interval=growth_interval,
            )
        self._scaler_config = dict(
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
        )

        self.attempted_steps = 0
        self.successful_updates = 0
        self.skipped_steps = 0
        self.consecutive_skips = 0
        self._unscaled = False
        self._hooked_optimizers: dict[int, list[int]] = {}

    # ----- forward / backward -------------------------------------------------

    def forward_context(self):
        """Wrap the forward pass and loss computation; backward stays outside.

        TE modes nest the TE context inside BF16 autocast: te.Linear GEMMs use
        FP8 (fp8_te) or BF16 (bf16_te); every other op follows torch autocast.
        Backward runs outside and inherits the recorded recipe (§8.5 step 5).
        """
        if self.autocast_dtype is None:
            return contextlib.nullcontext()
        if not self.uses_te:
            return torch.autocast(self.device_type, dtype=self.autocast_dtype)
        import fp8_backend

        stack = contextlib.ExitStack()
        stack.enter_context(torch.autocast(self.device_type, dtype=self.autocast_dtype))
        stack.enter_context(fp8_backend.te_autocast(enabled=self.fp8_enabled, recipe=self._te_recipe))
        return stack

    def backward(self, loss: torch.Tensor) -> None:
        if self._unscaled:
            raise PrecisionProtocolError(
                "backward() called after unscale_once_if_needed(); unscale must "
                "happen once, after every microbatch of the effective batch"
            )
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    # ----- optimizer step -----------------------------------------------------

    def unscale_once_if_needed(self, optimizer: torch.optim.Optimizer) -> None:
        """Unscale grads in place so clipping sees true values. Idempotent."""
        if self._unscaled:
            return
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)
        self._unscaled = True

    def optimizer_step_and_update_scaler(self, optimizer: torch.optim.Optimizer) -> StepRecord:
        """Step unless grads are nonfinite. Never steps silently on inf/NaN."""
        counter = self._step_counter(optimizer)
        self.unscale_once_if_needed(optimizer)
        grad_finite = _grads_finite(optimizer)
        scale_before = self.current_scale()
        steps_before = counter[0]

        if self.scaler is not None:
            # GradScaler skips optimizer.step() itself when it found inf/NaN.
            self.scaler.step(optimizer)
            self.scaler.update()
        elif grad_finite:
            optimizer.step()

        # A skip is detected from the optimizer actually running (a step hook),
        # not inferred from the scale shrinking (handbook §13.1).
        did_step = counter[0] > steps_before
        self.attempted_steps += 1
        if did_step:
            self.successful_updates += 1
            self.consecutive_skips = 0
        else:
            self.skipped_steps += 1
            self.consecutive_skips += 1
        self._unscaled = False

        return StepRecord(
            attempted_step=self.attempted_steps,
            successful_update=self.successful_updates,
            did_step=did_step,
            optimizer_step_skipped=not did_step,
            grad_finite=grad_finite,
            scale_before=scale_before,
            scale_after=self.current_scale(),
        )

    def current_scale(self) -> float | None:
        return float(self.scaler.get_scale()) if self.scaler is not None else None

    def _step_counter(self, optimizer: torch.optim.Optimizer) -> list[int]:
        key = id(optimizer)
        if key not in self._hooked_optimizers:
            if self.scaler is not None and getattr(optimizer, "_step_supports_amp_scaling", False):
                # Fused optimizers receive found_inf and skip inside step(), so
                # the step hook would fire on skipped updates too.
                raise ValueError("fp16 skip detection does not support fused optimizers; use fused=False")
            counter = [0]

            def _count(opt, args, kwargs):
                counter[0] += 1

            optimizer.register_step_post_hook(_count)
            self._hooked_optimizers[key] = counter
        return self._hooked_optimizers[key]

    # ----- checkpointing ------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "mode": self.mode,
            "attempted_steps": self.attempted_steps,
            "successful_updates": self.successful_updates,
            "skipped_steps": self.skipped_steps,
            "consecutive_skips": self.consecutive_skips,
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
        }

    def load_state_dict(self, state: dict) -> None:
        if state["mode"] != self.mode:
            raise ValueError(f"checkpoint precision {state['mode']!r} != policy {self.mode!r}")
        self.attempted_steps = state["attempted_steps"]
        self.successful_updates = state["successful_updates"]
        self.skipped_steps = state["skipped_steps"]
        self.consecutive_skips = state["consecutive_skips"]
        if self.scaler is not None:
            self.scaler.load_state_dict(state["scaler"])

    # ----- manifest -----------------------------------------------------------

    def manifest(self, model: torch.nn.Module | None = None, optimizer=None) -> dict:
        """Machine-readable record of what this policy actually configured."""
        m = {
            "mode": self.mode,
            "backend": (
                "transformer_engine" if self.uses_te else "torch_amp" if self.autocast_dtype is not None else "eager"
            ),
            "device_type": self.device_type,
            "autocast_enabled": self.autocast_dtype is not None,
            "autocast_dtype": str(self.autocast_dtype) if self.autocast_dtype else None,
            "grad_scaler": self._scaler_config if self.scaler is not None else None,
            "tf32": {
                "requested_allow_tf32": self.allow_tf32,
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
            },
            "versions": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
                "device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            },
        }
        if self.uses_te:
            import fp8_backend

            m["transformer_engine"] = fp8_backend.backend_manifest(self.fp8_recipe_config, self.fp8_enabled)
            if model is not None:
                m["transformer_engine"]["module_inventory"] = fp8_backend.te_module_inventory(model)
        if model is not None:
            m["parameter_dtypes"] = _param_dtype_summary(model)
        if optimizer is not None:
            m["optimizer"] = {
                "class": type(optimizer).__name__,
                # Empty until the first successful step creates the state.
                "state_dtypes": _optimizer_state_dtypes(optimizer),
            }
        return m

    def write_manifest(self, path: str | Path, model=None, optimizer=None) -> dict:
        m = self.manifest(model, optimizer)
        Path(path).write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")
        return m


def build_precision_policy(mode: str, **kwargs) -> PrecisionPolicy:
    """Precision contract from handbook §5.2.

    The returned policy exposes forward_context(), .scaler (None unless fp16)
    and manifest(), which together cover (context, scaler_or_none, policy_manifest).
    """
    return PrecisionPolicy(mode, **kwargs)


# ----- helpers ----------------------------------------------------------------


def _configure_tf32(allow: bool) -> None:
    # Global switches: applied identically for every mode so that the autocast
    # dtype is the only arithmetic difference between policies.
    torch.backends.cuda.matmul.allow_tf32 = allow
    torch.backends.cudnn.allow_tf32 = allow
    torch.set_float32_matmul_precision("high" if allow else "highest")


def _grads_finite(optimizer: torch.optim.Optimizer) -> bool:
    grads = [p.grad for g in optimizer.param_groups for p in g["params"] if p.grad is not None]
    if not grads:
        return True
    return bool(torch.stack([torch.isfinite(g).all() for g in grads]).all())


def _param_dtype_summary(model: torch.nn.Module) -> dict:
    trainable, frozen = Counter(), Counter()
    for p in model.parameters():
        (trainable if p.requires_grad else frozen)[str(p.dtype)] += p.numel()
    return {"trainable": dict(trainable), "frozen": dict(frozen)}


def _optimizer_state_dtypes(optimizer: torch.optim.Optimizer) -> dict:
    dtypes = Counter()
    for state in optimizer.state.values():
        for name, value in state.items():
            if torch.is_tensor(value):
                dtypes[f"{name}:{value.dtype}"] += value.numel()
    return dict(dtypes)


def record_as_row(record: StepRecord) -> dict:
    """StepRecord -> train.csv columns, with NaN rather than a silent zero for N/A."""
    row = asdict(record)
    for key in ("scale_before", "scale_after"):
        if row[key] is None:
            row[key] = math.nan
    return row
