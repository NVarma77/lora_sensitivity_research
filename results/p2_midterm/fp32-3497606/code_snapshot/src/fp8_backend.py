"""Transformer Engine FP8 backend (handbook §8.4-8.6, §16.2, task P2-B).

Owns: the FP8 recipe, the TE autocast context, layer eligibility rules, the
coverage manifest and kernel classification for the profiler proof.

`convert_linears_to_te` is the known-working TE layer prototype that P2 hands
to P1 (§15.2). P1's te_conversion.py should replace it in the shared trainer;
until then the P2 stand-in runner (src/pilot_runner.py) uses it directly.

Only precision.py should call `te_autocast`; models and trainers go through
PrecisionPolicy.forward_context().
"""

from __future__ import annotations

import contextlib
import inspect
import re
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

try:
    import transformer_engine as _te_root
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe as te_recipe

    TE_AVAILABLE = True
    TE_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on the node's environment
    te = te_recipe = _te_root = None
    TE_AVAILABLE = False
    TE_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

# TE's FP8 GEMMs need both GEMM dims divisible by 16 (and the token dim by 8/16)
# in the TE versions the handbook cites (§8.5 step 2). Checked per layer here;
# the token dim is fixed by seq_len * microbatch in the trainer config.
FP8_DIM_MULTIPLE = 16


@dataclass(frozen=True)
class FP8RecipeConfig:
    """Pilot recipe (§14.2 `hybrid_delayed_scaling`): E4M3 forward, E5M2 backward."""

    name: str = "delayed_scaling"
    fp8_format: str = "HYBRID"
    amax_history_len: int = 16
    amax_compute_algo: str = "max"
    margin: int = 0


def require_te() -> None:
    if not TE_AVAILABLE:
        raise RuntimeError(f"Transformer Engine is not importable: {TE_IMPORT_ERROR}")


def build_recipe(cfg: FP8RecipeConfig = FP8RecipeConfig()):
    require_te()
    if cfg.name != "delayed_scaling":
        raise ValueError(f"unsupported FP8 recipe {cfg.name!r}")
    return te_recipe.DelayedScaling(
        fp8_format=getattr(te_recipe.Format, cfg.fp8_format),
        amax_history_len=cfg.amax_history_len,
        amax_compute_algo=cfg.amax_compute_algo,
        margin=cfg.margin,
    )


def te_autocast(enabled: bool, recipe=None):
    """TE FP8 context; `te.autocast` in current TE, `fp8_autocast` in older releases."""
    require_te()
    if hasattr(te, "autocast"):
        return te.autocast(enabled=enabled, recipe=recipe)
    return te.fp8_autocast(enabled=enabled, fp8_recipe=recipe)  # pragma: no cover


def te_autocast_api_name() -> str:
    require_te()
    return "transformer_engine.pytorch.autocast" if hasattr(te, "autocast") else "transformer_engine.pytorch.fp8_autocast"


def fp8_support() -> dict:
    """What TE itself reports about FP8 on this device."""
    require_te()
    if hasattr(te, "is_fp8_available"):  # TE >= 2.x public API
        try:
            ok, reason = te.is_fp8_available(return_reason=True)
        except TypeError:
            ok, reason = te.is_fp8_available(), ""
    elif hasattr(te, "fp8"):  # older TE
        ok, reason = te.fp8.check_fp8_support()
    else:
        return {"fp8_available": None, "reason": "no FP8 support query in this TE version"}
    return {"fp8_available": bool(ok), "reason": reason or ""}


# ----- eligibility and conversion --------------------------------------------


def is_fp8_eligible(module: nn.Module) -> tuple[bool, str]:
    if not isinstance(module, nn.Linear):
        return False, "not nn.Linear"
    for name, dim in (("in_features", module.in_features), ("out_features", module.out_features)):
        if dim % FP8_DIM_MULTIPLE:
            return False, f"{name}={dim} not divisible by {FP8_DIM_MULTIPLE}"
    return True, "ok"


def default_te_filter(name: str, module: nn.Module) -> bool:
    """Backbone decoder projections only. lm_head is tied to the embedding in
    Qwen2.5-1.5B, so it stays an eager nn.Linear (recorded in coverage)."""
    return bool(re.search(r"\.layers\.\d+\.(self_attn|mlp)\.", name))


def convert_linears_to_te(model: nn.Module, name_filter=default_te_filter) -> list[dict]:
    """Replace eligible nn.Linear modules with te.Linear holding the same tensors.

    Weights are copied (same [out, in] layout), dtype and requires_grad are
    preserved. Returns one coverage row per nn.Linear in the model.
    Call before attaching LoRA so PEFT dispatches to its TeLinear wrapper.
    """
    require_te()
    rows = []
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        eligible, reason = is_fp8_eligible(module)
        wanted = name_filter(name, module)
        rows.append(
            {
                "module": name,
                "in_features": module.in_features,
                "out_features": module.out_features,
                "bias": module.bias is not None,
                "weight_dtype": str(module.weight.dtype),
                "fp8_shape_eligible": eligible,
                "eligibility_reason": reason,
                "converted_to_te": bool(eligible and wanted),
            }
        )
        if eligible and wanted:
            targets.append(name)

    for name in targets:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        old: nn.Linear = getattr(parent, child)
        new = te.Linear(
            old.in_features,
            old.out_features,
            bias=old.bias is not None,
            params_dtype=old.weight.dtype,
            device=old.weight.device,
        )
        with torch.no_grad():
            new.weight.copy_(old.weight)
            if old.bias is not None:
                new.bias.copy_(old.bias)
        new.weight.requires_grad_(old.weight.requires_grad)
        if old.bias is not None:
            new.bias.requires_grad_(old.bias.requires_grad)
        setattr(parent, child, new)
    return rows


def te_module_inventory(model: nn.Module) -> dict:
    counts = {"te_linear": 0, "eager_linear": 0}
    for _, m in model.named_modules():
        if TE_AVAILABLE and isinstance(m, te.Linear):
            counts["te_linear"] += 1
        elif isinstance(m, nn.Linear):
            counts["eager_linear"] += 1
    return counts


def backend_manifest(recipe_cfg: FP8RecipeConfig | None, fp8_enabled: bool) -> dict:
    require_te()
    m = {
        "te_version": getattr(_te_root, "__version__", "unknown"),
        "autocast_api": te_autocast_api_name(),
        "autocast_signature": str(inspect.signature(te.autocast)) if hasattr(te, "autocast") else None,
        "fp8_enabled": fp8_enabled,
        "recipe": asdict(recipe_cfg) if (fp8_enabled and recipe_cfg) else None,
        "fp8_dim_multiple": FP8_DIM_MULTIPLE,
        "notes": (
            "FP8 applies only to te.Linear GEMMs (forward E4M3, backward grads E5M2 under HYBRID). "
            "Adapters (PEFT TeLinear lora_A/lora_B), norms, attention, softmax, embeddings, lm_head "
            "and loss run under torch BF16 autocast or FP32."
        ),
    }
    try:
        m.update(fp8_support())
    except Exception as exc:  # pragma: no cover
        m["fp8_available"] = None
        m["reason"] = f"check failed: {exc}"
    return m


# ----- kernel evidence ---------------------------------------------------------

# Patterns seen in cuBLASLt / TE kernel names for FP8 work on Hopper. The
# profiler proof does not rely on these alone: it also diffs the kernel set
# against the TE-BF16 negative control (scripts/fp8_profile_report.py).
# cuBLASLt nvjet kernels encode operand types in the prefix: nvjet_tst_* (BF16 here), nvjet_qq*/qr*
# (FP8 operands; seen only when the FP8 recipe is on, forward and backward GEMMs respectively).
_FP8_KERNEL_PATTERNS = re.compile(r"e4m3|e5m2|fp8|float8|nvjet_q[qr]|quantize|cast_transpose|amax", re.IGNORECASE)
_GEMM_PATTERNS = re.compile(r"gemm|nvjet|xmma|cutlass|sm90_", re.IGNORECASE)


def classify_kernel(name: str) -> str:
    fp8 = bool(_FP8_KERNEL_PATTERNS.search(name))
    gemm = bool(_GEMM_PATTERNS.search(name))
    if fp8 and gemm:
        return "fp8_gemm"
    if fp8:
        return "fp8_cast_or_scaling"
    if gemm:
        return "other_gemm"
    return "other"
