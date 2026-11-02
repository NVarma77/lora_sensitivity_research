"""Integration seams between Person 3's DDP runner and the team's modules (handbook §5.5, rule 8).

Wired to the code actually in the repo (branch p2-midterm-fp8, Oct 31):
  * precision.build_precision_policy(mode, device_type=...) -> PrecisionPolicy   (Person 2)
      modes: fp32 | bf16 | fp16 | bf16_te | fp8_te
  * fp8_backend.convert_linears_to_te(model) -> coverage list                     (Person 2 prototype;
      Person 1's te_conversion.py replaces it later — change ONLY convert_for_te below)
  * pilot_runner.build_dataset(tokenizer, seq_len, max_train_tokens, val_blocks)  (P2 stand-in data path;
      swap for Person 1's data.py when it lands — change ONLY pilot_data below)

There is deliberately NO built-in precision fallback: every autocast/GradScaler/TE call goes through
precision.py (enforced by tests/test_precision.py::test_no_direct_amp_calls_outside_precision_module).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

TE_MODES = ("bf16_te", "fp8_te")
ALL_MODES = ("fp32", "bf16", "fp16", "bf16_te", "fp8_te")
ALIASES = {"fp8": "fp8_te"}


def canonical_mode(mode: str) -> str:
    m = ALIASES.get(mode, mode)
    if m not in ALL_MODES:
        raise ValueError(f"unknown precision {mode!r}; expected one of {ALL_MODES}")
    return m


# ============================================================================ precision
class PolicyAdapter:
    """Uniform interface the runner uses, on top of Person 2's PrecisionPolicy."""

    def __init__(self, mode: str, team_policy):
        self.mode = mode
        self.team = team_policy

    @property
    def scaler(self):
        return self.team.scaler

    def forward_context(self):
        return self.team.forward_context()

    def backward(self, loss: torch.Tensor) -> None:
        self.team.backward(loss)

    def unscale(self, opt) -> None:
        self.team.unscale_once_if_needed(opt)

    def step(self, opt) -> tuple[bool, Optional[float]]:
        """Returns (did_step, scale_after). Skips detected by P2's optimizer-step hook (§13.1)."""
        rec = self.team.optimizer_step_and_update_scaler(opt)
        return bool(rec.did_step), rec.scale_after

    @property
    def consecutive_skips(self) -> int:
        return getattr(self.team, "consecutive_skips", 0)

    def manifest(self, model=None, opt=None) -> dict:
        try:
            return {"source": "team precision.py", **self.team.manifest(model, opt)}
        except Exception as e:  # manifest must never kill a run
            return {"source": "team precision.py", "mode": self.mode, "manifest_error": repr(e)}


def build_precision_policy(mode: str, device: torch.device) -> PolicyAdapter:
    """Always Person 2's policy (handbook: no AMP/TE calls outside precision.py)."""
    import precision as team_precision  # Person 2: src/precision.py
    return PolicyAdapter(canonical_mode(mode),
                         team_policy=team_precision.build_precision_policy(canonical_mode(mode), device_type=device.type))


def convert_for_te(model: torch.nn.Module, mode: str) -> Optional[list]:
    """TE modes: convert decoder nn.Linear -> te.Linear BEFORE attaching LoRA (as in P2's pilots)."""
    if canonical_mode(mode) not in TE_MODES:
        return None
    import fp8_backend  # Person 2 prototype; later Person 1's te_conversion.py
    fp8_backend.require_te()
    return fp8_backend.convert_linears_to_te(model)


def lora_attachment_report(model) -> Optional[list]:
    try:
        import pilot_runner
        return pilot_runner.lora_attachment_report(model)
    except Exception:
        return None


# ============================================================================ data
@dataclass
class TokenData:
    train: torch.Tensor     # [n_train, seq_len] int64
    val: torch.Tensor       # [n_val, seq_len] int64
    meta: dict


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


def synthetic_data(vocab: int, seq_len: int, n_train: int = 512, n_val: int = 37, seed: int = 7) -> TokenData:
    g = torch.Generator().manual_seed(seed)
    tr = torch.randint(0, vocab, (n_train, seq_len), generator=g)
    va = torch.randint(0, vocab, (n_val, seq_len), generator=g)
    return TokenData(tr, va, {"dataset": "synthetic", "train_sha256": _sha(tr), "val_sha256": _sha(va)})


def pilot_data(tokenizer, seq_len: int, max_train_tokens: int, val_blocks: int,
               is_main: bool, barrier: Callable[[], None]) -> TokenData:
    """Same cached WikiText-103 tensors as P2's pilots (pinned revisions, DIPDESK_CACHE).
    Rank 0 builds the cache if missing; other ranks wait, then load it."""
    import pilot_runner
    if is_main:
        pilot_runner.build_dataset(tokenizer, seq_len, max_train_tokens, val_blocks)
    barrier()
    b = pilot_runner.build_dataset(tokenizer, seq_len, max_train_tokens, val_blocks)
    return TokenData(torch.as_tensor(b["train"]).long(), torch.as_tensor(b["val"]).long(),
                     {"dataset": f"{pilot_runner.DATASET_ID}/{pilot_runner.DATASET_CONFIG}@{pilot_runner.DATASET_REVISION[:8]}",
                      "source": "pilot_runner.build_dataset (P2 stand-in; replace with P1 data.py)",
                      "key": b.get("key"), "train_sha256": b.get("train_sha256"), "val_sha256": b.get("val_sha256")})


def default_model_revision() -> Optional[str]:
    try:
        import pilot_runner
        return pilot_runner.MODEL_REVISION
    except Exception:
        return None
