"""DipDesk — distributed execution layer (Person 3, handbook §5.2 / §9).

Implements the handbook's distributed contract:

    build_execution_context(world_size, effective_global_batch, microbatch)
        -> ExecutionContext(rank, local_rank, accumulation, sampler, reductions)

Design rules this module enforces (handbook §9.4–9.5):
  * Effective global batch B = M (microbatch/GPU) x N (workers) x A (accumulation) is FIXED
    across world sizes. B=32, M=2  ->  N=1: A=16 | N=2: A=8 | N=4: A=4.
  * The SAME 32 global sample IDs make up update u regardless of N (GlobalBatchSampler).
  * Loss is normalised by the GLOBAL number of valid target tokens, and multiplied by N to
    cancel DDP's gradient averaging, so the synchronised gradient equals the serial
    single-GPU gradient of  sum(token NLL) / T_global.
  * Intermediate micro-batches run under DDP.no_sync(); only the last one all-reduces.
  * Validation sums NLL and token counts across ranks (never averages per-rank means).
  * FP16 skipped-step decisions are made identically on all ranks.

Works with NCCL (GPU) and Gloo (CPU) so the correctness tests can run anywhere.
"""
from __future__ import annotations

import contextlib
import hashlib
import math
import os
import time
from datetime import timedelta
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Sequence

import torch
import torch.distributed as dist


# ----------------------------------------------------------------------------- process group
@dataclass
class DistEnv:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    backend: str

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1


def init_distributed(force_cpu: bool = False) -> DistEnv:
    """Initialise torch.distributed from torchrun's env vars (or fall back to 1 process).

    One torchrun launcher per node starts the workers; each worker binds to LOCAL_RANK.
    """
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    use_cuda = torch.cuda.is_available() and not force_cpu
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if world_size > 1 and not dist.is_initialized():
        # generous timeout: rank 0 may tokenise/cache WikiText-103 while others wait at a barrier
        kwargs = {"backend": backend, "timeout": timedelta(minutes=int(os.environ.get("DIPDESK_PG_TIMEOUT_MIN", 60)))}
        if use_cuda:
            kwargs["device_id"] = device  # binds NCCL communicator eagerly (PyTorch >= 2.3)
        try:
            dist.init_process_group(**kwargs)
        except TypeError:  # older PyTorch without device_id
            kwargs.pop("device_id", None)
            dist.init_process_group(**kwargs)
    return DistEnv(rank, local_rank, world_size, device, backend)


def cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def rank0_print(*args, **kwargs) -> None:
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs, flush=True)


# ----------------------------------------------------------------------------- global batches
class GlobalBatchSampler:
    """Deterministic global batches that do NOT depend on world size.

    For each seed we build one permutation of all training example indices (epochs are
    concatenated permutations, each seeded from (seed, epoch)). Update u always consumes
    positions [u*B, (u+1)*B) of that stream, so 1, 2 and 4 GPUs see the identical 32 IDs.
    The cursor (= number of global batches handed out) is saved in checkpoints for resume.
    """

    def __init__(self, num_examples: int, global_batch: int, seed: int, start_update: int = 0):
        if num_examples < global_batch:
            raise ValueError(f"num_examples={num_examples} < global_batch={global_batch}")
        self.n = num_examples
        self.B = global_batch
        self.seed = seed
        self.cursor = start_update
        self._epoch_cache: dict[int, torch.Tensor] = {}

    def _perm(self, epoch: int) -> torch.Tensor:
        if epoch not in self._epoch_cache:
            g = torch.Generator().manual_seed(self.seed * 100_003 + epoch)
            self._epoch_cache = {epoch: torch.randperm(self.n, generator=g)}
        return self._epoch_cache[epoch]

    def global_ids(self, update: int) -> List[int]:
        start = update * self.B
        out: List[int] = []
        while len(out) < self.B:
            epoch, offset = divmod(start + len(out), self.n)
            take = min(self.B - len(out), self.n - offset)
            out.extend(self._perm(epoch)[offset: offset + take].tolist())
        return out

    def next(self) -> List[int]:
        ids = self.global_ids(self.cursor)
        self.cursor += 1
        return ids

    def state_dict(self) -> dict:
        return {"cursor": self.cursor, "seed": self.seed, "n": self.n, "B": self.B}

    def load_state_dict(self, sd: dict) -> None:
        assert sd["n"] == self.n and sd["B"] == self.B and sd["seed"] == self.seed, "sampler mismatch"
        self.cursor = sd["cursor"]


# ----------------------------------------------------------------------------- context
@dataclass
class ExecutionContext:
    env: DistEnv
    effective_global_batch: int
    microbatch: int
    accumulation: int
    sampler: Optional[GlobalBatchSampler] = None
    extra: dict = field(default_factory=dict)

    # convenience pass-throughs
    @property
    def rank(self) -> int: return self.env.rank
    @property
    def local_rank(self) -> int: return self.env.local_rank
    @property
    def world_size(self) -> int: return self.env.world_size
    @property
    def device(self) -> torch.device: return self.env.device
    @property
    def is_main(self) -> bool: return self.env.is_main

    # ---- data split -------------------------------------------------------------------
    def shard_global_batch(self, global_ids: Sequence[int]) -> List[List[int]]:
        """Rank-major split: rank r gets ids[r*M*A:(r+1)*M*A], cut into A micro-batches of M."""
        per_rank = self.microbatch * self.accumulation
        assert len(global_ids) == per_rank * self.world_size, "global batch size mismatch"
        mine = list(global_ids[self.rank * per_rank:(self.rank + 1) * per_rank])
        return [mine[i * self.microbatch:(i + 1) * self.microbatch] for i in range(self.accumulation)]

    # ---- DDP sync control ---------------------------------------------------------------
    @staticmethod
    def sync_context(model: torch.nn.Module, is_last_micro: bool):
        """no_sync() on all but the last micro-batch (handbook §9.4)."""
        if isinstance(model, torch.nn.parallel.DistributedDataParallel) and not is_last_micro:
            return model.no_sync()
        return contextlib.nullcontext()

    # ---- reductions ---------------------------------------------------------------------
    def all_reduce_sum(self, x: torch.Tensor) -> torch.Tensor:
        if self.env.is_distributed:
            x = x.clone()
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    def all_reduce_max(self, x: torch.Tensor) -> torch.Tensor:
        if self.env.is_distributed:
            x = x.clone()
            dist.all_reduce(x, op=dist.ReduceOp.MAX)
        return x

    def global_valid_tokens(self, local_count: int | torch.Tensor) -> float:
        """T_global = number of valid next-token targets in the whole effective batch."""
        t = torch.as_tensor(local_count, dtype=torch.float64, device=self.device)
        return float(self.all_reduce_sum(t).item())

    def loss_for_backward(self, loss_sum: torch.Tensor, t_global: float) -> torch.Tensor:
        """Scale a micro-batch's SUMMED token NLL so that, after DDP averaging over N ranks
        and accumulation over A micro-batches, grad == d/dθ [ Σ_all-tokens NLL / T_global ]."""
        return loss_sum * (self.world_size / t_global)

    def reduce_eval(self, nll_sum: float, tokens: float) -> tuple[float, float]:
        """Token-weighted global validation loss (handbook T13). Returns (mean_nll, tokens)."""
        t = torch.tensor([nll_sum, tokens], dtype=torch.float64, device=self.device)
        t = self.all_reduce_sum(t)
        return float(t[0] / t[1]), float(t[1])

    def eval_shard(self, num_eval_examples: int) -> List[int]:
        """Deterministic strided split of the validation set; no drop_last, no shuffle."""
        return list(range(self.rank, num_eval_examples, self.world_size))

    def agree_step(self, did_step: bool) -> bool:
        """All ranks must agree on FP16 skip/step (use MIN: any skip => all skip)."""
        t = torch.tensor([1.0 if did_step else 0.0], device=self.device)
        if self.env.is_distributed:
            dist.all_reduce(t, op=dist.ReduceOp.MIN)
        return bool(t.item() > 0.5)

    def gather_ids(self, ids: Sequence[int]) -> List[int]:
        """Gather every rank's processed sample IDs (for test T11)."""
        if not self.env.is_distributed:
            return list(ids)
        out: list = [None] * self.world_size
        dist.all_gather_object(out, list(ids))
        return [i for chunk in out for i in chunk]


def build_execution_context(world_size: int, effective_global_batch: int, microbatch: int,
                            env: Optional[DistEnv] = None, num_train_examples: Optional[int] = None,
                            seed: int = 0) -> ExecutionContext:
    """Handbook distributed contract. Raises if B is not divisible by M x N."""
    env = env or init_distributed()
    assert env.world_size == world_size, f"WORLD_SIZE={env.world_size} but config says {world_size}"
    denom = microbatch * world_size
    if effective_global_batch % denom != 0:
        raise ValueError(f"B={effective_global_batch} not divisible by M x N = {microbatch} x {world_size}")
    accumulation = effective_global_batch // denom
    sampler = (GlobalBatchSampler(num_train_examples, effective_global_batch, seed)
               if num_train_examples is not None else None)
    return ExecutionContext(env, effective_global_batch, microbatch, accumulation, sampler)


# ----------------------------------------------------------------------------- model helpers
def wrap_ddp(model: torch.nn.Module, ctx: ExecutionContext) -> torch.nn.Module:
    """DDP over trainable params only. LoRA: frozen base params get no buckets at all."""
    if not ctx.env.is_distributed:
        return model
    return torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[ctx.local_rank] if ctx.device.type == "cuda" else None,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
        broadcast_buffers=False,
    )


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


@torch.no_grad()
def param_checksum(model: torch.nn.Module, trainable_only: bool = True) -> str:
    """SHA-1 over trainable parameters (fp64 sum + sum of squares per tensor), for cross-rank checks."""
    h = hashlib.sha1()
    for n, p in unwrap(model).named_parameters():
        if trainable_only and not p.requires_grad:
            continue
        d = p.detach().double()
        h.update(f"{n}:{d.sum().item():.12e}:{(d * d).sum().item():.12e}".encode())
    return h.hexdigest()


def ranks_agree_on_checksum(model: torch.nn.Module, ctx: ExecutionContext) -> bool:
    c = param_checksum(model)
    if not ctx.env.is_distributed:
        return True
    out: list = [None] * ctx.world_size
    dist.all_gather_object(out, c)
    return len(set(out)) == 1


# ----------------------------------------------------------------------------- timing & memory
class StepTimer:
    """Wall-clock per effective update with device synchronisation at both ends."""

    def __init__(self, device: torch.device):
        self.device = device
        self._t0 = 0.0

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def start(self):
        self._sync()
        self._t0 = time.perf_counter()

    def stop(self) -> float:
        self._sync()
        return time.perf_counter() - self._t0


def memory_stats(ctx: ExecutionContext) -> dict:
    """Peak allocated/reserved bytes, max over ranks (handbook: report both)."""
    if ctx.device.type != "cuda":
        return {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
    t = torch.tensor([torch.cuda.max_memory_allocated(ctx.device),
                      torch.cuda.max_memory_reserved(ctx.device)], dtype=torch.float64, device=ctx.device)
    t = ctx.all_reduce_max(t)
    return {"peak_allocated_bytes": int(t[0].item()), "peak_reserved_bytes": int(t[1].item())}


def summarize_times(times: Sequence[float]) -> dict:
    if not times:
        return {"n": 0}
    s = sorted(times)
    q = lambda p: s[min(len(s) - 1, max(0, int(math.floor(p * (len(s) - 1) + 0.5))))]
    return {"n": len(s), "median_s": q(0.5), "p25_s": q(0.25), "p75_s": q(0.75),
            "mean_s": sum(s) / len(s), "min_s": s[0], "max_s": s[-1]}
