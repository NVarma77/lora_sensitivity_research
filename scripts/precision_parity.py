"""3-way FP32/BF16/FP16 parity on the toy LM (P2-A deliverable).

Same seed, weights, data order and loop for every mode. Writes:
  parity_steps.csv   per-step loss/scale/skip telemetry for each mode
  parity_table.md    fixed-batch loss/grad error vs FP32 and training summary
  manifest_<mode>.json
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from precision import build_precision_policy, record_as_row  # noqa: E402
from toy_models import make_batches, make_toy, toy_loss  # noqa: E402

MODES = ("fp32", "bf16", "fp16")


def fixed_batch_probe(policy, model, batch):
    """Loss and unscaled grads at the initial state, no optimizer step."""
    model.zero_grad(set_to_none=True)
    with policy.forward_context():
        loss_sum, n = toy_loss(model, batch)
    loss = loss_sum / n
    scale = policy.current_scale() or 1.0
    (loss * scale).backward()
    grads = torch.cat([p.grad.flatten().float() for p in model.parameters()]) / scale
    model.zero_grad(set_to_none=True)
    return loss.item(), grads


def train(policy, model, opt, batches, accum):
    rows = []
    for i in range(0, len(batches), accum):
        micro = batches[i : i + accum]
        n_tokens = sum(b[:, 1:].numel() for b in micro)
        opt.zero_grad(set_to_none=True)
        loss_total = 0.0
        for ids in micro:
            with policy.forward_context():
                loss_sum, _ = toy_loss(model, ids)
            policy.backward(loss_sum / n_tokens)
            loss_total += float(loss_sum)
        policy.unscale_once_if_needed(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        rec = policy.optimizer_step_and_update_scaler(opt)
        rows.append({**record_as_row(rec), "loss": loss_total / n_tokens, "grad_norm_preclip": float(grad_norm)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batches = make_batches(args.steps * args.accum, batch=8, seq=65, seed=args.seed + 1, device=device)
    val = make_batches(4, batch=8, seq=65, seed=10_000 + args.seed, device=device)
    results, all_rows = {}, []
    for mode in MODES:
        policy = build_precision_policy(mode, device_type=device)
        model = make_toy(args.seed, device, d_model=128, d_ff=512, vocab=256)
        probe_loss, probe_grads = fixed_batch_probe(policy, model, batches[0])
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, foreach=False)
        rows = train(policy, model, opt, batches, args.accum)
        with torch.no_grad(), policy.forward_context():
            val_loss = sum(float(toy_loss(model, v)[0]) for v in val) / sum(v[:, 1:].numel() for v in val)
        policy.write_manifest(args.out / f"manifest_{mode}.json", model, opt)
        for r in rows:
            all_rows.append({"mode": mode, **r})
        results[mode] = dict(
            probe_loss=probe_loss, probe_grads=probe_grads, final_train=rows[-1]["loss"], val=val_loss,
            attempted=policy.attempted_steps, successful=policy.successful_updates,
            skipped=policy.skipped_steps, final_scale=policy.current_scale(),
        )

    with open(args.out / "parity_steps.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)

    ref = results["fp32"]
    lines = [
        f"# P2-A toy parity ({device}, {torch.cuda.get_device_name() if device == 'cuda' else 'cpu'}, "
        f"torch {torch.__version__}, seed {args.seed}, {args.steps} updates x {args.accum} microbatches)",
        "",
        "| mode | init loss | init loss Δ vs FP32 | init grad rel L2 vs FP32 | init grad cosine | "
        "final train loss | val loss | val Δ vs FP32 | attempted | successful | skipped | final scale |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for mode in MODES:
        r = results[mode]
        rel = float((r["probe_grads"] - ref["probe_grads"]).norm() / ref["probe_grads"].norm())
        cos = float(F.cosine_similarity(r["probe_grads"], ref["probe_grads"], dim=0))
        lines.append(
            f"| {mode} | {r['probe_loss']:.6f} | {r['probe_loss'] - ref['probe_loss']:+.2e} | {rel:.2e} | "
            f"{cos:.6f} | {r['final_train']:.6f} | {r['val']:.6f} | {r['val'] - ref['val']:+.2e} | "
            f"{r['attempted']} | {r['successful']} | {r['skipped']} | {r['final_scale'] if r['final_scale'] else 'n/a'} |"
        )
    lines += ["", "Toy model only: this checks the precision switch, not Qwen/LoRA behaviour."]
    (args.out / "parity_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
