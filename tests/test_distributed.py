"""Single-process unit tests for src/distributed.py (Person 3). Run by the team's normal `pytest`.
The multi-process checks (T11-T13 across 1/2/4 workers) live in tests/ddp_toy_check.py (torchrun)."""
import pytest
import torch

from distributed import (DistEnv, ExecutionContext, GlobalBatchSampler, build_execution_context,
                         summarize_times)

CPU = torch.device("cpu")


def ctx_for(rank, world, B=32, M=2):
    env = DistEnv(rank=rank, local_rank=rank, world_size=world, device=CPU, backend="gloo")
    return ExecutionContext(env, B, M, B // (M * world))


@pytest.mark.parametrize("world,acc", [(1, 16), (2, 8), (4, 4)])
def test_accumulation_keeps_global_batch(world, acc):
    assert ctx_for(0, world).accumulation == acc


def test_indivisible_batch_rejected():
    env = DistEnv(0, 0, 3, CPU, "gloo")
    with pytest.raises(ValueError):
        build_execution_context(3, 32, 2, env=env)


def test_sampler_independent_of_world_size_and_deterministic():
    a = GlobalBatchSampler(100, 32, seed=2)
    b = GlobalBatchSampler(100, 32, seed=2)
    seq_a = [a.next() for _ in range(7)]          # 7*32 > 100 -> crosses epoch boundaries
    seq_b = [b.global_ids(u) for u in range(7)]
    assert seq_a == seq_b
    assert all(len(x) == 32 for x in seq_a)
    assert GlobalBatchSampler(100, 32, seed=3).global_ids(0) != seq_a[0]


def test_sampler_resume():
    a = GlobalBatchSampler(100, 32, seed=0)
    for _ in range(3):
        a.next()
    b = GlobalBatchSampler(100, 32, seed=0)
    b.load_state_dict(a.state_dict())
    assert a.next() == b.next()


@pytest.mark.parametrize("world", [1, 2, 4])
def test_shards_partition_global_batch(world):
    ids = list(range(1000, 1032))
    got = []
    for r in range(world):
        micro = ctx_for(r, world).shard_global_batch(ids)
        assert len(micro) == 32 // (2 * world) and all(len(m) == 2 for m in micro)
        got += [i for m in micro for i in m]
    assert sorted(got) == ids


def test_loss_scaling_recovers_global_token_mean():
    # per-rank backward of loss_sum * N / T, averaged over N ranks == global sum / T
    world, T = 4, 400.0
    sums = [torch.tensor(float(v)) for v in (10, 20, 30, 40)]
    scaled = [ctx_for(r, world).loss_for_backward(s, T) for r, s in enumerate(sums)]
    assert torch.isclose(sum(scaled) / world, torch.tensor(100.0 / T))


def test_summarize_times():
    s = summarize_times([3.0, 1.0, 2.0, 4.0, 5.0])
    assert s["median_s"] == 3.0 and s["n"] == 5 and s["min_s"] == 1.0
