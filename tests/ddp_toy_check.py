"""Toy DDP correctness tests T11/T12/T13 (+ cross-rank checksum), handbook §9.5 / §19.1.

Run with the SAME command for 1, 2 and 4 workers; every run must print PASS:
    torchrun --standalone --nproc_per_node=4 tests/ddp_toy_check.py --out results/p3_toy_ddp
On a CPU-only machine it uses the Gloo backend automatically; on GH200 it uses NCCL.

What it proves
  T11  every effective update contains exactly the 32 global sample IDs of the serial run
       (no duplicates, no drops), independent of world size.
  T12  after accumulation + DDP all-reduce, each parameter gradient equals the serial
       single-process gradient of  Σ token-loss / T_global  (variable valid-token masks!).
  T13  token-weighted validation loss reduced across ranks equals the serial value
       (odd-sized validation set, so ranks hold different numbers of examples).
  +    all ranks hold identical parameters after each optimizer update, and the parameter
       trajectory equals the serial trajectory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from distributed import (build_execution_context, cleanup, init_distributed, param_checksum,  # noqa: E402
                         rank0_print, ranks_agree_on_checksum, wrap_ddp)

D, L = 16, 8                # feature dim, "sequence length"
N_TRAIN, N_VAL = 96, 37     # 37 = odd on purpose
B, M = 32, 2                # effective global batch, micro-batch per worker
UPDATES = 4
TOL = 1e-10                 # fp64 on toy model; differences come only from summation order


def make_data(n: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, L, D, generator=g, dtype=torch.float64)
    w_true = torch.randn(D, 1, generator=g, dtype=torch.float64)
    y = (x @ w_true).squeeze(-1) + 0.1 * torch.randn(n, L, generator=g, dtype=torch.float64)
    # variable valid-target counts per example, like padded/masked LM positions
    valid = torch.rand(n, L, generator=g) > 0.3
    valid[:, 0] = True
    return x, y, valid


def token_loss_sum(model, x, y, valid):
    pred = model(x).squeeze(-1)
    return (((pred - y) ** 2) * valid).sum(), valid.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/p3_toy_ddp")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    env = init_distributed(force_cpu=args.cpu)
    ctx = build_execution_context(env.world_size, B, M, env=env, num_train_examples=N_TRAIN, seed=0)
    dev = ctx.device
    x, y, valid = make_data(N_TRAIN, 1)
    xv, yv, vv = make_data(N_VAL, 2)
    x, y, valid, xv, yv, vv = (t.to(dev) for t in (x, y, valid, xv, yv, vv))

    torch.manual_seed(1234)  # identical init on every rank (DDP also broadcasts rank 0's params)
    model = torch.nn.Linear(D, 1, bias=True).double().to(dev)
    serial = torch.nn.Linear(D, 1, bias=True).double().to(dev)
    serial.load_state_dict(model.state_dict())
    ddp_model = wrap_ddp(model, ctx)
    opt = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
    opt_s = torch.optim.SGD(serial.parameters(), lr=0.01)

    report = {"world_size": ctx.world_size, "backend": env.backend, "accumulation": ctx.accumulation,
              "updates": []}
    ok = True
    for u in range(UPDATES):
        gids = ctx.sampler.next()
        micro = ctx.shard_global_batch(gids)

        # ---- T11: global sample IDs ---------------------------------------------------
        seen = ctx.gather_ids([i for mb in micro for i in mb])
        t11 = sorted(seen) == sorted(gids) and len(set(seen)) == len(gids)

        # ---- DDP path -------------------------------------------------------------------
        local_tokens = int(sum(valid[mb].sum().item() for mb in micro))
        t_global = ctx.global_valid_tokens(local_tokens)
        opt.zero_grad(set_to_none=True)
        for k, mb in enumerate(micro):
            with ctx.sync_context(ddp_model, is_last_micro=(k == len(micro) - 1)):
                ls, _ = token_loss_sum(ddp_model, x[mb], y[mb], valid[mb])
                ctx.loss_for_backward(ls, t_global).backward()
        g_ddp = [p.grad.detach().clone() for p in model.parameters()]

        # ---- serial reference (whole effective batch, one process) ----------------------
        opt_s.zero_grad(set_to_none=True)
        ls_s, tok_s = token_loss_sum(serial, x[gids], y[gids], valid[gids])
        (ls_s / tok_s).backward()
        g_ser = [p.grad.detach().clone() for p in serial.parameters()]
        max_grad_err = max(float((a - b).abs().max()) for a, b in zip(g_ddp, g_ser))
        t12 = max_grad_err < TOL and abs(t_global - float(tok_s)) < 0.5

        opt.step(); opt_s.step()
        agree = ranks_agree_on_checksum(ddp_model, ctx)
        max_param_err = max(float((a.detach() - b.detach()).abs().max()) for a, b in zip(model.parameters(), serial.parameters()))
        traj = max_param_err < TOL

        # ---- T13: validation reduction -------------------------------------------------
        with torch.no_grad():
            idx = ctx.eval_shard(N_VAL)
            nll, tok = token_loss_sum(model, xv[idx], yv[idx], vv[idx]) if idx else (torch.tensor(0.), torch.tensor(0))
            val_ddp, _ = ctx.reduce_eval(float(nll), float(tok))
            nll_s, tok_s2 = token_loss_sum(serial, xv, yv, vv)
            val_ser = float(nll_s / tok_s2)
        t13 = abs(val_ddp - val_ser) < TOL

        ok &= t11 and t12 and t13 and agree and traj
        report["updates"].append({"update": u, "T11_ids_match": t11, "T12_grad_match": t12,
                                  "max_abs_grad_err": max_grad_err, "T13_val_match": t13,
                                  "val_ddp": val_ddp, "val_serial": val_ser, "ranks_agree": agree,
                                  "trajectory_match": traj, "max_abs_param_err": max_param_err,
                                  "t_global": t_global, "first_ids": gids[:4]})
        rank0_print(f"[N={ctx.world_size} A={ctx.accumulation}] update {u}: T11={t11} T12={t12} "
                     f"(max|Δg|={max_grad_err:.1e}) T13={t13} ranks_agree={agree} traj={traj} "
                     f"val={val_ddp:.10f}")

    report["final_checksum"] = param_checksum(ddp_model)
    report["PASS"] = bool(ok)
    if ctx.is_main:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, f"toy_ddp_N{ctx.world_size}.json"), "w") as f:
            json.dump(report, f, indent=2)
    rank0_print(("PASS" if ok else "FAIL") + f" — toy DDP correctness, world_size={ctx.world_size}, "
                f"backend={env.backend}, final_checksum={report['final_checksum'][:12]}")
    cleanup()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
