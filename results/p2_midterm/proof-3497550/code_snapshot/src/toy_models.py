"""Tiny deterministic models for precision tests and the 3-way parity check."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyLM(nn.Module):
    """Embedding -> pre-norm MLP block -> LM head, trained with token cross-entropy.

    Covers the op types that behave differently under autocast (matmul,
    layer norm, softmax/log-softmax) without needing Qwen.
    """

    def __init__(self, vocab: int = 64, d_model: int = 32, d_ff: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.up = nn.Linear(d_model, d_ff)
        self.down = nn.Linear(d_ff, d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(ids)
        h = h + self.down(F.gelu(self.up(self.norm(h))))
        return self.head(h)


def toy_loss(model: ToyLM, ids: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Sum of next-token NLL and valid-target count (handbook §12.1)."""
    logits = model(ids[:, :-1])
    targets = ids[:, 1:]
    loss_sum = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1), reduction="sum"
    )
    return loss_sum, targets.numel()


def make_toy(seed: int = 0, device: str = "cpu", **kwargs) -> ToyLM:
    torch.manual_seed(seed)
    return ToyLM(**kwargs).to(device)


def make_batches(n: int, batch: int = 4, seq: int = 17, vocab: int = 64, seed: int = 1, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, vocab, (batch, seq), generator=g).to(device) for _ in range(n)]
