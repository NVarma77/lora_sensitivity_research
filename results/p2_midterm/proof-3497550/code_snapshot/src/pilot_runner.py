"""P2 STAND-IN Qwen2.5-1.5B + LoRA pilot runner (midterm, Oct 31 2026).

NOT the shared DipDesk trainer. Person 1 owns data.py/model.py/lora.py/train.py
(handbook §5.5, §15.1); those were not available when P2 needed real-model
pilots, so this file exists only to exercise precision.py and fp8_backend.py on
the real workload. When P1's trainer lands, delete this file and run the same
modes through it. Do not use outputs from here as X01 matrix cells.

Follows the handbook where it is cheap to do so: §4.1 proposed protocol (Qwen2.5-1.5B
base, WikiText-103 raw v1, seq 512, LoRA q_proj/v_proj alpha=r dropout 0, AdamW lr 2e-4,
5% warmup + cosine, global batch 32, clip 1.0), §5.4 CSV fields, §12.1 token-sum loss.
Deviations, all recorded in config.json: training data is a deterministic article-ordered
prefix of the train split (max_train_tokens), validation is the first val_blocks blocks of
the validation split, microbatch 8 x accumulation 4 instead of 2 x 16.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fp8_backend  # noqa: E402
from precision import TE_MODES, build_precision_policy, record_as_row  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-1.5B"
MODEL_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
DATASET_ID = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
CACHE_DIR = Path(os.environ.get("DIPDESK_CACHE", "/work/nvme/bcrc/nvarma2/dipdesk_cache"))

TRAIN_FIELDS = [
    "attempted_step", "successful_update", "train_tokens_seen", "loss_numerator", "loss_denominator",
    "loss", "lr", "grad_norm", "scaler_value", "optimizer_step_skipped", "grad_finite",
    "scale_before", "scale_after", "step_seconds", "allocated_bytes",
]
EVAL_FIELDS = [
    "successful_update", "tokens_seen", "val_loss_numerator", "val_loss_denominator", "val_loss",
    "perplexity", "eval_mode", "wall_seconds",
]


# ----- data --------------------------------------------------------------------

_ARTICLE_HEADING = re.compile(r"^ = [^=].* = $")


def _articles(rows):
    """Group WikiText rows into articles at top-level ' = Title = ' headings."""
    current = []
    for row in rows:
        if _ARTICLE_HEADING.match(row.rstrip("\n")) and current:
            yield "".join(current)
            current = []
        current.append(row)
    if current:
        yield "".join(current)


def _pack(tokenizer, rows, seq_len, max_tokens=None):
    ids = []
    batch = []
    for article in _articles(rows):
        if not article.strip():
            continue
        batch.append(article)
        if len(batch) == 256:
            for t in tokenizer(batch, add_special_tokens=False)["input_ids"]:
                ids.extend(t + [tokenizer.eos_token_id])
            batch = []
            if max_tokens and len(ids) >= max_tokens:
                break
    if batch and not (max_tokens and len(ids) >= max_tokens):
        for t in tokenizer(batch, add_special_tokens=False)["input_ids"]:
            ids.extend(t + [tokenizer.eos_token_id])
    if max_tokens:
        ids = ids[:max_tokens]
    n_blocks = len(ids) // seq_len  # tail policy: drop the final incomplete block
    return torch.tensor(ids[: n_blocks * seq_len], dtype=torch.int32).view(n_blocks, seq_len)


def build_dataset(tokenizer, seq_len: int, max_train_tokens: int, val_blocks: int) -> dict:
    key = f"wt103-{DATASET_REVISION[:8]}-qwen-{MODEL_REVISION[:8]}-L{seq_len}-T{max_train_tokens}-V{val_blocks}"
    path = CACHE_DIR / f"{key}.pt"
    if path.exists():
        return torch.load(path)
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, DATASET_CONFIG, revision=DATASET_REVISION)
    train = _pack(tokenizer, ds["train"]["text"], seq_len, max_train_tokens)
    val = _pack(tokenizer, ds["validation"]["text"], seq_len)[:val_blocks]
    bundle = {
        "train": train,
        "val": val,
        "train_sha256": hashlib.sha256(train.numpy().tobytes()).hexdigest(),
        "val_sha256": hashlib.sha256(val.numpy().tobytes()).hexdigest(),
        "key": key,
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(bundle, tmp)
    tmp.rename(path)
    return bundle


# ----- model -------------------------------------------------------------------


def build_model(mode: str, rank: int, targets: list[str], seed: int, device):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    # Open decision 1 (docs/precision.md): frozen base loaded as FP32 in every
    # mode; checkpoint storage dtype is BF16 and is recorded, not recovered.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=torch.float32, attn_implementation="sdpa"
    ).to(device)
    model.config.use_cache = False
    coverage = None
    if mode in TE_MODES:
        coverage = fp8_backend.convert_linears_to_te(model)
    torch.manual_seed(seed)
    cfg = LoraConfig(
        task_type="CAUSAL_LM", r=rank, lora_alpha=rank, lora_dropout=0.0, bias="none",
        target_modules=targets, use_rslora=False,
    )
    model = get_peft_model(model, cfg)
    return model, coverage


def lora_attachment_report(model) -> list[dict]:
    rows = []
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "base_layer"):
            base = module.base_layer
            rows.append({
                "module": name,
                "lora_wrapper": type(module).__module__ + "." + type(module).__name__,
                "base_layer": type(base).__module__ + "." + type(base).__name__,
                "base_is_te": fp8_backend.TE_AVAILABLE and isinstance(base, fp8_backend.te.Linear),
                "lora_A_shape": list(module.lora_A["default"].weight.shape),
                "lora_B_shape": list(module.lora_B["default"].weight.shape),
                "lora_dtype": str(module.lora_A["default"].weight.dtype),
            })
    return rows


# ----- loss / eval -------------------------------------------------------------


def loss_sum(model, ids: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Token-sum next-token NLL in FP32 and its valid-target count (§12.1)."""
    logits = model(input_ids=ids).logits
    targets = ids[:, 1:]
    nll = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)).float(), targets.reshape(-1).long(), reduction="sum"
    )
    return nll, targets.numel()


@torch.no_grad()
def evaluate(model, policy, val: torch.Tensor, micro: int, device) -> tuple[float, int]:
    model.eval()
    num, den = 0.0, 0
    for i in range(0, len(val), micro):
        ids = val[i : i + micro].to(device, non_blocking=True)
        with policy.forward_context():
            s, n = loss_sum(model, ids)
        num += float(s)
        den += n
    model.train()
    return num, den


# ----- parameter audit ---------------------------------------------------------


def snapshot_params(model) -> dict:
    return {n: p.detach().to("cpu", copy=True) for n, p in model.named_parameters()}


def audit_params(model, before: dict, optimizer) -> dict:
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    changed_trainable, changed_frozen, frozen_with_grad = [], [], []
    max_frozen_diff = 0.0
    for n, p in model.named_parameters():
        same = torch.equal(p.detach().cpu(), before[n])
        if n in trainable:
            if not same:
                changed_trainable.append(n)
        else:
            if not same:
                changed_frozen.append(n)
                max_frozen_diff = max(max_frozen_diff, float((p.detach().cpu() - before[n]).abs().max()))
            if p.grad is not None:
                frozen_with_grad.append(n)
    non_lora_trainable = sorted(n for n in trainable if "lora_" not in n)
    return {
        "n_parameters_total": sum(p.numel() for p in model.parameters()),
        "n_parameters_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "n_tensors_trainable": len(trainable),
        "n_tensors_frozen": sum(1 for _ in model.parameters()) - len(trainable),
        "non_lora_trainable_tensors": non_lora_trainable,
        "optimizer_params_equal_trainable": opt_ids == {id(p) for p in model.parameters() if p.requires_grad},
        "n_trainable_tensors_changed": len(changed_trainable),
        "trainable_tensors_unchanged": sorted(trainable - set(changed_trainable)),
        "n_frozen_tensors_changed": len(changed_frozen),
        "frozen_tensors_changed": changed_frozen[:50],
        "max_abs_change_frozen": max_frozen_diff,
        "frozen_tensors_with_grad": frozen_with_grad[:50],
        "only_lora_weights_updated": (
            not changed_frozen and not non_lora_trainable and not frozen_with_grad
            and len(changed_trainable) == len(trainable)
        ),
    }


# ----- profiling ---------------------------------------------------------------


def kernel_table(prof) -> list[dict]:
    rows = []
    for evt in prof.key_averages():
        if evt.device_type != torch.autograd.DeviceType.CUDA:
            continue
        t = getattr(evt, "self_device_time_total", None) or getattr(evt, "self_cuda_time_total", 0)
        rows.append({
            "kernel": evt.key,
            "calls": evt.count,
            "device_time_us": round(float(t), 2),
            "class": fp8_backend.classify_kernel(evt.key),
        })
    rows.sort(key=lambda r: -r["device_time_us"])
    return rows


# ----- main --------------------------------------------------------------------


def lr_at(update: int, total: int, peak: float, warmup_frac: float) -> float:
    warm = max(1, round(total * warmup_frac))
    if update < warm:
        return peak * (update + 1) / warm
    progress = (update - warm) / max(1, total - warm)
    return peak * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def environment() -> dict:
    import peft
    import transformers

    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "transformer_engine": getattr(fp8_backend._te_root, "__version__", None) if fp8_backend.TE_AVAILABLE else None,
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    try:
        env["code_sha"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parents[1], text=True
        ).strip()
        env["code_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--", "src", "scripts"], cwd=Path(__file__).parents[1], text=True
        ).strip())
    except Exception:
        env["code_sha"] = None
    return env


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", required=True, choices=["fp32", "bf16", "fp16", "bf16_te", "fp8_te"])
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--targets", default="q_proj,v_proj")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--updates", type=int, default=200, help="successful optimizer updates")
    p.add_argument("--global-batch", type=int, default=32)
    p.add_argument("--micro", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--val-blocks", type=int, default=128)
    p.add_argument("--max-train-tokens", type=int, default=8_000_000)
    p.add_argument("--attempt-budget-factor", type=float, default=1.25,
                   help="stop after updates*factor attempts (FP16 skip budget, open decision 4)")
    p.add_argument("--max-consecutive-skips", type=int, default=50,
                   help="FAILED_NUMERICAL after this many consecutive skipped updates (open decision 3)")
    p.add_argument("--profile-update", type=int, default=-1,
                   help="torch.profiler trace of this successful-update index (after warmup)")
    p.add_argument("--nsys-range", action="store_true",
                   help="cudaProfilerStart/Stop around the profiled update for nsys --capture-range")
    p.add_argument("--out", required=True)
    p.add_argument("--run-id", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    device = torch.device("cuda")
    accum = args.global_batch // args.micro
    assert accum * args.micro == args.global_batch
    run_id = args.run_id or (
        f"P2PILOT-qwen15b-wt103-r{args.rank}-{args.mode}-s{args.seed}-g1-{os.environ.get('SLURM_JOB_ID', 'local')}"
    )
    out = Path(args.out) / run_id
    out.mkdir(parents=True, exist_ok=False)  # never overwrite a run (§1.3 rule 7)
    config = {
        "run_id": run_id,
        "study": "P2-pilot (stand-in runner, not X01)",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "load_dtype": "float32", "source_dtype": "bfloat16"},
        "dataset": {"id": DATASET_ID, "config": DATASET_CONFIG, "revision": DATASET_REVISION,
                    "train": f"article-ordered prefix, first {args.max_train_tokens} tokens",
                    "val": f"first {args.val_blocks} blocks of validation split",
                    "eos_between_articles": True, "tail_policy": "drop"},
        "method": {"name": "lora", "rank": args.rank, "alpha": args.rank, "dropout": 0.0,
                   "targets": args.targets.split(","), "init": "peft_default"},
        "training": {"updates": args.updates, "global_batch_sequences": args.global_batch,
                     "microbatch_sequences": args.micro, "gradient_accumulation": accum, "seq_len": args.seq_len,
                     "lr": args.lr, "scheduler": "linear_warmup_cosine", "warmup_frac": args.warmup_frac,
                     "optimizer": "AdamW(fused=False, foreach=False)", "weight_decay": args.weight_decay,
                     "max_grad_norm": args.max_grad_norm, "seed": args.seed, "eval_every": args.eval_every,
                     "attempt_budget_factor": args.attempt_budget_factor,
                     "max_consecutive_skips": args.max_consecutive_skips},
        "precision": {"mode": args.mode},
        "argv": sys.argv,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    run = {"run_id": run_id, "status": "RUNNING", "environment": environment(), "started": time.time(),
           "profile_update": args.profile_update}

    def write_run():
        (out / "run.json").write_text(json.dumps(run, indent=2, default=str) + "\n")

    write_run()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    data = build_dataset(tok, args.seq_len, args.max_train_tokens, args.val_blocks)
    run["data"] = {k: data[k] for k in ("key", "train_sha256", "val_sha256")}
    run["data"]["train_blocks"] = len(data["train"])
    run["data"]["val_blocks"] = len(data["val"])

    policy = build_precision_policy(args.mode)
    model, coverage = build_model(args.mode, args.rank, args.targets.split(","), args.seed, device)
    attach = lora_attachment_report(model)
    (out / "coverage.json").write_text(json.dumps(
        {"te_conversion": coverage, "lora_attachment": attach}, indent=2) + "\n")
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay, fused=False, foreach=False)
    before = snapshot_params(model)

    g = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(data["train"]), generator=g)
    needed = int(args.updates * args.attempt_budget_factor) + 1
    if needed * args.global_batch > len(order):
        raise SystemExit(f"need {needed * args.global_batch} blocks, have {len(order)}; raise --max-train-tokens")
    first_ids = data["train"][order[: args.global_batch]]
    run["first_global_batch_sha256"] = hashlib.sha256(first_ids.numpy().tobytes()).hexdigest()

    tf = open(out / "train.csv", "w", newline="")
    ef = open(out / "eval.csv", "w", newline="")
    tw = csv.DictWriter(tf, TRAIN_FIELDS)
    ew = csv.DictWriter(ef, EVAL_FIELDS)
    tw.writeheader()
    ew.writeheader()

    t0 = time.time()

    def do_eval(update, tokens):
        num, den = evaluate(model, policy, data["val"], args.micro, device)
        vl = num / den
        ew.writerow({"successful_update": update, "tokens_seen": tokens, "val_loss_numerator": num,
                     "val_loss_denominator": den, "val_loss": vl,
                     "perplexity": math.exp(vl) if vl < 50 else math.inf, "eval_mode": "full_val_subset",
                     "wall_seconds": time.time() - t0})
        ef.flush()
        print(f"[eval] update {update} val_loss {vl:.5f}", flush=True)
        return vl

    tokens_seen = 0
    status, failure = "COMPLETED", None
    torch.cuda.reset_peak_memory_stats()
    try:
        do_eval(0, 0)
        cursor = 0
        prof = None
        while policy.successful_updates < args.updates:
            if policy.attempted_steps >= needed - 1:
                status, failure = "FAILED_NUMERICAL", f"attempt budget {needed - 1} exhausted"
                break
            profiling = policy.successful_updates == args.profile_update and prof is None
            if profiling:
                if args.nsys_range:
                    torch.cuda.cudart().cudaProfilerStart()
                prof = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=True,
                )
                prof.__enter__()
            torch.cuda.synchronize()
            ts = time.time()
            update = policy.successful_updates
            for group in opt.param_groups:
                group["lr"] = lr_at(update, args.updates, args.lr, args.warmup_frac)
            idx = order[cursor * args.global_batch : (cursor + 1) * args.global_batch]
            cursor += 1
            batch = data["train"][idx]
            denom = args.global_batch * (args.seq_len - 1)
            opt.zero_grad(set_to_none=True)
            num = 0.0
            for m in range(accum):
                ids = batch[m * args.micro : (m + 1) * args.micro].to(device, non_blocking=True)
                with policy.forward_context():
                    s, n = loss_sum(model, ids)
                policy.backward(s / denom)
                num += float(s.detach())
            policy.unscale_once_if_needed(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            rec = policy.optimizer_step_and_update_scaler(opt)
            torch.cuda.synchronize()
            if profiling:
                prof.__exit__(None, None, None)
                if args.nsys_range:
                    torch.cuda.cudart().cudaProfilerStop()
                prof.export_chrome_trace(str(out / "profile_trace.json"))
                kt = kernel_table(prof)
                with open(out / "profile_kernels.csv", "w", newline="") as f:
                    w = csv.DictWriter(f, ["kernel", "calls", "device_time_us", "class"])
                    w.writeheader()
                    w.writerows(kt)
                (out / "profile_ops.txt").write_text(
                    prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=60)
                )
            step_s = time.time() - ts
            tokens_seen += denom if rec.did_step else 0
            row = record_as_row(rec)
            tw.writerow({
                "attempted_step": rec.attempted_step, "successful_update": rec.successful_update,
                "train_tokens_seen": tokens_seen, "loss_numerator": num, "loss_denominator": denom,
                "loss": num / denom, "lr": opt.param_groups[0]["lr"], "grad_norm": float(grad_norm),
                "scaler_value": row["scale_after"], "optimizer_step_skipped": rec.optimizer_step_skipped,
                "grad_finite": rec.grad_finite, "scale_before": row["scale_before"],
                "scale_after": row["scale_after"], "step_seconds": step_s,
                "allocated_bytes": torch.cuda.memory_allocated(),
            })
            if rec.attempted_step % 10 == 0:
                tf.flush()
                print(f"[train] attempt {rec.attempted_step} update {rec.successful_update} "
                      f"loss {num / denom:.4f} gnorm {float(grad_norm):.3g} skip {rec.optimizer_step_skipped} "
                      f"scale {row['scale_after']} {step_s:.2f}s", flush=True)
            if policy.consecutive_skips >= args.max_consecutive_skips:
                status, failure = "FAILED_NUMERICAL", f"{policy.consecutive_skips} consecutive skipped updates"
                break
            if rec.did_step and (rec.successful_update % args.eval_every == 0 or rec.successful_update == args.updates):
                vl = do_eval(rec.successful_update, tokens_seen)
                if not math.isfinite(vl):
                    status, failure = "FAILED_NUMERICAL", f"nonfinite validation loss at update {rec.successful_update}"
                    break
    except torch.cuda.OutOfMemoryError as exc:
        status, failure = "FAILED_OOM", str(exc)[:500]
    finally:
        tf.close()
        ef.close()

    audit = audit_params(model, before, opt)
    (out / "param_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    policy.write_manifest(out / "precision.json", model, opt)
    run.update({
        "status": status, "failure": failure, "finished": time.time(),
        "attempted_steps": policy.attempted_steps, "successful_updates": policy.successful_updates,
        "skipped_steps": policy.skipped_steps, "train_tokens_seen": tokens_seen,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "only_lora_weights_updated": audit["only_lora_weights_updated"],
    })
    write_run()
    print(json.dumps({k: run[k] for k in ("run_id", "status", "failure", "successful_updates", "skipped_steps",
                                          "only_lora_weights_updated")}), flush=True)
    return 0 if status == "COMPLETED" else 2


if __name__ == "__main__":
    sys.exit(main())
