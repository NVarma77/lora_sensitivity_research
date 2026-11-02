"""DipDesk Person 3 — multi-GPU LoRA / full-FT runner, scaling benchmark (X05) and FT control (X06).

Launch (one torchrun per node; never nest torchrun launchers):
    torchrun --standalone --nproc_per_node=4 src/p3_ddp_runner.py \
        --model Qwen/Qwen2.5-1.5B --precision bf16 --lora_rank 16 --max_updates 50

Fixed-work guarantees (handbook §9.4): effective global batch B (default 32 x 512 tokens) is
identical for 1/2/4 GPUs; accumulation A = B / (M x N) adjusts; update u uses the same 32 global
sample IDs; loss is token-normalised over the whole effective batch.

Modes
  default            train for --max_updates SUCCESSFUL optimizer updates, eval every --eval_every
  --bench_timed K    X05 scaling benchmark: --bench_warmup W updates, then K timed updates, no eval
  --no_sync_all      timing control only: skip gradient all-reduce to expose communication cost
  --method full      X06 full fine-tuning (all base weights trainable, LoRA disabled)
  --profile          torch.profiler trace for 5 updates after warm-up (rank 0)

Outputs (handbook §5.4 schema) in <out>/<run_id>/: config.json, run.json, train.csv, eval.csv,
bench.json (benchmark mode), trace/ (profiler), adapter/ (LoRA, optional).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distributed import (StepTimer, barrier, build_execution_context, cleanup, init_distributed,  # noqa: E402
                         memory_stats, rank0_print, ranks_agree_on_checksum, summarize_times, unwrap,
                         wrap_ddp)
from team_hooks import (ALL_MODES, TE_MODES, build_precision_policy, canonical_mode,  # noqa: E402
                        convert_for_te, default_model_revision, lora_attachment_report, pilot_data,
                        synthetic_data)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B", help="HF id, or 'tiny' (random tiny Qwen2 for tests)")
    p.add_argument("--revision", default=None, help="HF commit; default = the pin used by P2's pilots")
    p.add_argument("--data", default="pilot", choices=["pilot", "synthetic"],
                   help="pilot = same cached WikiText-103 tensors as P2's pilots (until P1's data.py lands)")
    p.add_argument("--max_train_tokens", type=int, default=8_000_000, help="pilot data: train prefix size")
    p.add_argument("--val_blocks", type=int, default=128, help="pilot data: first N validation blocks")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--global_batch", type=int, default=32)
    p.add_argument("--micro_batch", type=int, default=8, help="8 matches P2 pilots; A = B/(M*N)")
    p.add_argument("--max_updates", type=int, default=800)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--max_eval_examples", type=int, default=0, help="0 = full validation split")
    p.add_argument("--method", default="lora", choices=["lora", "full"])
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=None, help="default = rank (handbook §4.1)")
    p.add_argument("--targets", default="q_proj,v_proj")
    p.add_argument("--lr", type=float, default=None, help="default 2e-4 LoRA, 1e-5 full FT")
    p.add_argument("--warmup_frac", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--precision", default="bf16", choices=list(ALL_MODES) + ["fp8"], help="fp8 = fp8_te")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grad_ckpt", action="store_true", help="activation checkpointing (log as protocol change)")
    p.add_argument("--bench_warmup", type=int, default=20)
    p.add_argument("--bench_timed", type=int, default=0, help=">0 switches to X05 benchmark mode")
    p.add_argument("--no_sync_all", action="store_true", help="benchmark control: no gradient all-reduce")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--save_adapter", action="store_true")
    p.add_argument("--out", default="results/p3_runs")
    p.add_argument("--run_id", default=None)
    p.add_argument("--cpu", action="store_true", help="force CPU/Gloo (tests)")
    return p.parse_args(argv)


# ----------------------------------------------------------------------------- model
def build_model(args, device):
    from transformers import AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(args.seed)  # identical adapter init on every rank (DDP also broadcasts)
    if args.model == "tiny":
        cfg = Qwen2Config(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                          tie_word_embeddings=True)
        model = Qwen2ForCausalLM(cfg)
        tokenizer = None
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
        # Frozen base loaded FP32 in every mode (team open decision 1, as in P2's pilots).
        model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
                                                     dtype=torch.float32, attn_implementation="sdpa")
    model.config.use_cache = False
    model.to(device)
    te_coverage = convert_for_te(model, args.precision)   # TE modes: before LoRA, like P2's pilots
    torch.manual_seed(args.seed)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    if args.method == "lora":
        from peft import LoraConfig, get_peft_model
        alpha = args.lora_alpha if args.lora_alpha is not None else args.lora_rank
        lcfg = LoraConfig(r=args.lora_rank, lora_alpha=alpha, lora_dropout=0.0, target_modules=targets,
                          bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, lcfg)
    else:
        for p_ in model.parameters():
            p_.requires_grad_(True)
    model.to(device)

    # ---- trainable-parameter audit (T03 for LoRA, T14 for full FT) --------------------
    names = [n for n, p_ in model.named_parameters() if p_.requires_grad]
    n_train = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    n_total = sum(p_.numel() for p_ in model.parameters())
    if args.method == "lora":
        bad = [n for n in names if "lora_" not in n]
        assert not bad, f"non-adapter params are trainable: {bad[:5]}"
        b_zero = all(float(p_.abs().max()) == 0.0 for n, p_ in model.named_parameters() if "lora_B" in n)
        assert b_zero, "LoRA B must be zero at init"
    audit = {"method": args.method, "trainable_params": n_train, "total_params": n_total,
             "trainable_pct": 100.0 * n_train / n_total, "n_trainable_tensors": len(names),
             "targets": targets if args.method == "lora" else "all",
             "te_converted_layers": len(te_coverage) if te_coverage is not None else 0,
             "full_ft_state_bytes_estimate": 16 * n_train if args.method == "full" else None}
    audit["_te_coverage"] = te_coverage
    return model, tokenizer, audit


def next_token_loss_sum(model, ids: torch.Tensor, policy) -> tuple[torch.Tensor, int]:
    """Summed next-token NLL over all valid targets (fixed-length packed blocks: L-1 per row)."""
    with policy.forward_context():
        logits = model(input_ids=ids).logits
    logits = logits[:, :-1, :].float()
    tgt = ids[:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="sum")
    return loss, tgt.numel()


@torch.no_grad()
def evaluate(model, val: torch.Tensor, ctx, policy, mb: int) -> tuple[float, float]:
    model.eval()
    idx = ctx.eval_shard(val.size(0))
    nll, tok = 0.0, 0.0
    for i in range(0, len(idx), mb):
        ids = val[idx[i:i + mb]].to(ctx.device, non_blocking=True)
        s, t = next_token_loss_sum(model, ids, policy)
        nll += float(s); tok += t
    model.train()
    return ctx.reduce_eval(nll, tok)


def lr_lambda(step, total, warmup):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))


def _nccl_version():
    try:
        v = torch.cuda.nccl.version()
        return ".".join(map(str, v)) if isinstance(v, tuple) else str(v)
    except Exception:
        return None


def env_report(ctx) -> dict:
    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return None
    import transformers
    try:
        import peft; peft_v = peft.__version__
    except ImportError:
        peft_v = None
    return {"hostname": socket.gethostname(), "platform": platform.platform(), "python": sys.version.split()[0],
            "torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__,
            "peft": peft_v, "nccl": _nccl_version(),
            "gpu": torch.cuda.get_device_name(ctx.device) if ctx.device.type == "cuda" else "cpu",
            "world_size": ctx.world_size, "backend": ctx.env.backend,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "code_sha": sh("git -C %s rev-parse HEAD" % os.path.dirname(os.path.abspath(__file__))),
            "git_dirty": bool(sh("git -C %s status --porcelain" % os.path.dirname(os.path.abspath(__file__))))}


# ----------------------------------------------------------------------------- main
def main(argv=None):
    args = parse_args(argv)
    args.precision = canonical_mode(args.precision)
    if args.revision is None and args.model != "tiny":
        args.revision = default_model_revision()
    if args.method == "full" and args.precision in TE_MODES:
        raise SystemExit("full FT + TE/FP8 is a stretch goal (handbook X06); not wired here")
    if args.lr is None:
        args.lr = 2e-4 if args.method == "lora" else 1e-5
    env = init_distributed(force_cpu=args.cpu)
    policy = build_precision_policy(args.precision, env.device)
    model, tokenizer, audit = build_model(args, env.device)

    te_coverage = audit.pop("_te_coverage", None)
    if args.data == "synthetic" or args.model == "tiny":
        vocab = unwrap(model).config.vocab_size if hasattr(unwrap(model), "config") else 512
        data = synthetic_data(vocab, args.seq_len)
    else:
        data = pilot_data(tokenizer, args.seq_len, args.max_train_tokens, args.val_blocks, env.is_main, barrier)
    if args.max_eval_examples:
        data.val = data.val[: args.max_eval_examples]

    ctx = build_execution_context(env.world_size, args.global_batch, args.micro_batch, env=env,
                                  num_train_examples=data.train.size(0), seed=args.seed)
    bench = args.bench_timed > 0
    total_updates = (args.bench_warmup + args.bench_timed) if bench else args.max_updates
    run_id = args.run_id or (f"{'X05' if bench else ('X06' if args.method == 'full' else 'P3')}_"
                             f"{args.method}_r{args.lora_rank if args.method == 'lora' else 0}_{args.precision}"
                             f"_s{args.seed}_N{ctx.world_size}_{time.strftime('%Y%m%d-%H%M%S')}")
    # all ranks must agree on the run id
    if ctx.env.is_distributed:
        holder = [run_id]; torch.distributed.broadcast_object_list(holder, src=0); run_id = holder[0]
    out = os.path.join(args.out, run_id)

    ddp_model = wrap_ddp(model, ctx)
    params = [p_ for p_ in ddp_model.parameters() if p_.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    warm = max(1, int(round(args.warmup_frac * total_updates)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, total_updates, warm))

    if ctx.is_main:
        os.makedirs(out, exist_ok=True)
        cfg = {**vars(args), "run_id": run_id, "accumulation": ctx.accumulation, "world_size": ctx.world_size,
               "tokens_per_update": args.global_batch * args.seq_len, "total_updates": total_updates,
               "warmup_updates": warm, "mode": "benchmark" if bench else "train"}
        json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
        json.dump({"run_id": run_id, "status": "RUNNING", "env": env_report(ctx), "audit": audit,
                   "precision_manifest": policy.manifest(), "data": data.meta,
                   "n_train_examples": data.train.size(0), "n_val_examples": data.val.size(0)},
                  open(os.path.join(out, "run.json"), "w"), indent=2, default=str)
    rank0_print(f"[{run_id}] N={ctx.world_size} M={args.micro_batch} A={ctx.accumulation} B={args.global_batch} "
                f"| trainable={audit['trainable_params']:,} ({audit['trainable_pct']:.3f}%) | precision={args.precision} "
                f"({policy.manifest().get('source')})")
    if ctx.is_main and (te_coverage is not None or args.method == "lora"):
        json.dump({"te_conversion": te_coverage, "lora_attachment": lora_attachment_report(unwrap(model))},
                  open(os.path.join(out, "coverage.json"), "w"), indent=2, default=str)

    train_rows, eval_rows, step_times = [], [], []
    timer = StepTimer(ctx.device)
    prof = None
    if args.profile and ctx.is_main and bench:
        acts = [torch.profiler.ProfilerActivity.CPU] + ([torch.profiler.ProfilerActivity.CUDA]
                                                       if ctx.device.type == "cuda" else [])
        prof = torch.profiler.profile(activities=acts, record_shapes=True,
                                      schedule=torch.profiler.schedule(wait=args.bench_warmup, warmup=2, active=5,
                                                                       repeat=1))
        prof.start()

    def do_eval(upd, tokens_seen):
        if bench:
            return
        vl, vt = evaluate(unwrap(ddp_model), data.val, ctx, policy, args.micro_batch)
        eval_rows.append({"successful_update": upd, "tokens_seen": tokens_seen, "val_loss_numerator": vl * vt,
                          "val_loss_denominator": vt, "val_loss": vl, "perplexity": math.exp(min(vl, 50)),
                          "eval_mode": "token_weighted_all_ranks"})
        rank0_print(f"  eval @ {upd}: val_loss={vl:.6f} ppl={math.exp(min(vl, 50)):.3f} tokens={int(vt)}")

    successful, attempted, tokens_seen, status = 0, 0, 0, "COMPLETED"
    max_attempts = int(total_updates * 1.25) + 5
    do_eval(0, 0)
    ddp_model.train()
    while successful < total_updates and attempted < max_attempts:
        gids = ctx.sampler.next()
        micro = ctx.shard_global_batch(gids)
        t_global = ctx.global_valid_tokens(sum(len(mb) for mb in micro) * (args.seq_len - 1))
        if ctx.device.type == "cuda" and attempted == 0:
            torch.cuda.reset_peak_memory_stats(ctx.device)
        timer.start()
        opt.zero_grad(set_to_none=True)
        local_loss = torch.zeros((), dtype=torch.float64, device=ctx.device)
        for k, mb in enumerate(micro):
            last = k == len(micro) - 1
            sync = ddp_model.no_sync() if (args.no_sync_all and hasattr(ddp_model, "no_sync")) \
                else ctx.sync_context(ddp_model, last)
            with sync:
                ids = data.train[mb].to(ctx.device, non_blocking=True)
                loss_sum, _ = next_token_loss_sum(ddp_model, ids, policy)
                policy.backward(ctx.loss_for_backward(loss_sum, t_global))
            local_loss += loss_sum.detach().double()
        policy.unscale(opt)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, args.clip))
        did_step, scale = policy.step(opt)
        did_step = ctx.agree_step(did_step)   # all ranks must agree (they see identical all-reduced grads)
        if did_step:
            sched.step(); successful += 1
        dt = timer.stop()
        attempted += 1
        tokens_seen += args.global_batch * args.seq_len
        if prof is not None:
            prof.step()
        train_loss = float(ctx.all_reduce_sum(local_loss)) / t_global
        if (not math.isfinite(train_loss) and policy.scaler is None) or policy.consecutive_skips >= 50:
            status = "FAILED_NUMERICAL"
        if bench and attempted > args.bench_warmup:
            step_times.append(dt)
        train_rows.append({"attempted_step": attempted, "successful_update": successful,
                           "train_tokens_seen": tokens_seen, "loss_numerator": train_loss * t_global,
                           "loss_denominator": t_global, "loss": train_loss, "lr": sched.get_last_lr()[0],
                           "grad_norm": grad_norm, "scaler_value": scale, "optimizer_step_skipped": not did_step,
                           "step_seconds": dt,
                           "allocated_bytes": torch.cuda.memory_allocated(ctx.device) if ctx.device.type == "cuda" else None})
        if attempted % 10 == 0 or attempted <= 3:
            rank0_print(f"  upd {successful}/{total_updates} (attempt {attempted}) loss={train_loss:.6f} "
                        f"gnorm={grad_norm:.3f} {dt*1000:.0f} ms" + ("" if did_step else "  [SKIPPED]"))
        if status != "COMPLETED":
            break
        if did_step and args.eval_every and successful % args.eval_every == 0:
            do_eval(successful, tokens_seen)
    if successful < total_updates and status == "COMPLETED":
        status = "FAILED_NUMERICAL"  # too many skipped FP16 steps
    if not bench and (not eval_rows or eval_rows[-1]["successful_update"] != successful):
        do_eval(successful, tokens_seen)
    if prof is not None:
        prof.stop()

    agree = ranks_agree_on_checksum(ddp_model, ctx)
    mem = memory_stats(ctx)
    bench_res = None
    if bench:
        s = summarize_times(step_times)
        tok = args.global_batch * args.seq_len
        bench_res = {**s, "tokens_per_update": tok,
                     "tokens_per_s_median": tok / s["median_s"] if s.get("n") else None,
                     "samples_per_s_median": args.global_batch / s["median_s"] if s.get("n") else None,
                     "world_size": ctx.world_size, "precision": args.precision, "method": args.method,
                     "lora_rank": args.lora_rank, "accumulation": ctx.accumulation,
                     "no_sync_all": args.no_sync_all, "warmup_updates": args.bench_warmup, **mem}
    if ctx.is_main:
        def wcsv(name, rows):
            if rows:
                with open(os.path.join(out, name), "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        wcsv("train.csv", train_rows); wcsv("eval.csv", eval_rows)
        if bench_res:
            json.dump(bench_res, open(os.path.join(out, "bench.json"), "w"), indent=2)
        if prof is not None:
            os.makedirs(os.path.join(out, "trace"), exist_ok=True)
            prof.export_chrome_trace(os.path.join(out, "trace", "rank0_trace.json"))
            table = None
            for key in ("self_device_time_total", "self_cuda_time_total", "self_cpu_time_total"):
                try:
                    table = prof.key_averages().table(sort_by=key, row_limit=40); break
                except Exception:
                    continue
            with open(os.path.join(out, "trace", "rank0_top_ops.txt"), "w") as f:
                f.write(table or "profiler table unavailable")
        if args.save_adapter and args.method == "lora":
            unwrap(ddp_model).save_pretrained(os.path.join(out, "adapter"))
        rj = json.load(open(os.path.join(out, "run.json")))
        try:
            json.dump(policy.manifest(unwrap(ddp_model), opt), open(os.path.join(out, "precision.json"), "w"),
                      indent=2, default=str)
        except Exception:
            pass
        rj.update({"status": status, "successful_updates": successful, "attempted_steps": attempted,
                   "skipped_steps": attempted - successful, "ranks_agree_on_params": agree,
                   "final_train_loss": train_rows[-1]["loss"] if train_rows else None,
                   "final_val_loss": eval_rows[-1]["val_loss"] if eval_rows else None, "memory": mem,
                   "bench": bench_res})
        json.dump(rj, open(os.path.join(out, "run.json"), "w"), indent=2, default=str)
    rank0_print(f"[{run_id}] {status}: updates={successful} attempts={attempted} ranks_agree={agree} "
                f"mem={mem} " + (f"median_step={bench_res['median_s']*1000:.1f} ms "
                                 f"tok/s={bench_res['tokens_per_s_median']:.0f}" if bench_res else ""))
    cleanup()
    # ranks intentionally diverge in the --no_sync_all timing control (no all-reduce), so do not fail it
    return 0 if status == "COMPLETED" and (agree or args.no_sync_all) else 1


if __name__ == "__main__":
    sys.exit(main())
