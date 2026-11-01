"""First BF16 vs FP8 numerical-error check on Qwen2.5-1.5B + LoRA (X02 A/B/C, one batch).

Arms on identical model state and one fixed validation microbatch (handbook §8.8, §12.7, X02):
  A = eager BF16 autocast, B = TE BF16 (converted te.Linear, FP8 off), C = TE FP8.
B and C share one converted model; only the precision policy differs. LoRA (r=16,
q_proj/v_proj) is attached with identical weights in both models, and lora_B is set
to small fixed random values so the adapter path and lora_A gradients are nonzero.

Scopes:
  isolated layer   the same captured input (from arm A) fed to one module in each arm
  full network     logits, per-token NLL, mean loss, LoRA gradients after backward
FP8 delayed scaling has no amax history on the first pass, so C is reported at the
first-ever FP8 pass (C_cold) and after WARMUP passes on other validation batches (C).

Writes X02_layer_parity.csv and X02_layer_parity.md to --out.
"""

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import fp8_backend  # noqa: E402
import pilot_runner as pr  # noqa: E402
from precision import build_precision_policy  # noqa: E402

LAYERS = ["base_model.model.model.layers.{i}.self_attn.q_proj", "base_model.model.model.layers.{i}.mlp.down_proj"]
LAYER_IDX = (0, 14, 27)
WARMUP = 4
MICRO = 8


def errs(x, ref):
    x, ref = x.detach().float().flatten(), ref.detach().float().flatten()
    d = (x - ref).abs()
    xn, rn = float(x.norm()), float(ref.norm())
    return {
        "linf": float(d.max()),
        "median_abs": float(d.median()),
        "rel_l2": float((x - ref).norm() / max(rn, 1e-12)),
        "cosine": float(torch.dot(x, ref) / (xn * rn)) if xn > 0 and rn > 0 else float("nan"),
        "ref_l2": rn,
    }


def forward_arm(model, policy, ids, names):
    caps = {}
    hooks = [model.get_submodule(n).register_forward_hook(
        lambda m, i, o, n=n: caps.__setitem__(n, (i[0].detach(), o.detach()))) for n in names]
    model.zero_grad(set_to_none=True)
    with policy.forward_context():
        logits = model(input_ids=ids).logits
        nll = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                              ids[:, 1:].reshape(-1).long(), reduction="none")
    nll.mean().backward()
    for h in hooks:
        h.remove()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()
             if p.requires_grad and p.grad is not None}
    return {"logits": logits.detach().float(), "nll": nll.detach(), "caps": caps, "grads": grads}


@torch.no_grad()
def isolated(model, policy, name, x):
    with policy.forward_context():
        return model.get_submodule(name)(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(pr.MODEL_ID, revision=pr.MODEL_REVISION)
    data = pr.build_dataset(tok, 512, 8_000_000, 128)
    val = data["val"]
    ids = val[:MICRO].to(dev)

    eager, _ = pr.build_model("bf16", 16, ["q_proj", "v_proj"], seed=1, device=dev)
    te_model, coverage = pr.build_model("bf16_te", 16, ["q_proj", "v_proj"], seed=1, device=dev)
    g = torch.Generator(device="cpu").manual_seed(123)
    lora = {}
    for n, p in eager.named_parameters():
        if "lora_B" in n:
            lora[n] = (torch.randn(p.shape, generator=g) * 1e-3).to(dev)
        elif "lora_A" in n:
            lora[n] = p.detach().clone()
    for m in (eager, te_model):
        sd = dict(m.named_parameters())
        with torch.no_grad():
            for n, v in lora.items():
                sd[n].copy_(v)
    # Frozen weights must be bitwise identical after conversion (B vs A is then purely backend).
    sa, sb = dict(eager.named_parameters()), dict(te_model.named_parameters())
    weight_mismatch = [n for n in sa if not torch.equal(sa[n], sb[n])]

    names = [t.format(i=i) for i in LAYER_IDX for t in LAYERS]
    pol = {m: build_precision_policy(m) for m in ("bf16", "bf16_te", "fp8_te")}
    res = {"A": forward_arm(eager, pol["bf16"], ids, names), "B": forward_arm(te_model, pol["bf16_te"], ids, names)}
    res["C_cold"] = forward_arm(te_model, pol["fp8_te"], ids, names)
    for k in range(WARMUP):
        with torch.no_grad(), pol["fp8_te"].forward_context():
            te_model(input_ids=val[MICRO * (k + 1): MICRO * (k + 2)].to(dev))
    res["C"] = forward_arm(te_model, pol["fp8_te"], ids, names)

    rows = []

    def add(cmp, scope, x, ref):
        rows.append({"comparison": cmp, "scope": scope, **errs(x, ref)})

    for cmp, (a, b) in {"B_vs_A (backend)": ("B", "A"), "C_vs_B (FP8 format)": ("C", "B"),
                        "C_cold_vs_B (FP8, no amax history)": ("C_cold", "B")}.items():
        # Isolated-layer probes run after the FP8 warmup, so they cannot represent the cold state.
        for n in names if a != "C_cold" else []:
            short = n.replace("base_model.model.model.", "")
            x_in = res["A"]["caps"][n][0]  # identical input for every arm
            ma = eager if b == "A" else te_model
            ref = isolated(ma, pol["bf16"] if b == "A" else pol["bf16_te"], n, x_in)
            pa = pol["bf16_te"] if a == "B" else pol["fp8_te"]
            add(cmp, f"isolated {short}", isolated(te_model, pa, n, x_in), ref)
        for n in names:
            short = n.replace("base_model.model.model.", "")
            add(cmp, f"in-network output {short}", res[a]["caps"][n][1], res[b]["caps"][n][1])
        add(cmp, "logits", res[a]["logits"], res[b]["logits"])
        add(cmp, "per-token NLL", res[a]["nll"], res[b]["nll"])
        rows.append({"comparison": cmp, "scope": "mean loss delta",
                     "linf": float(res[a]["nll"].mean() - res[b]["nll"].mean()), "median_abs": None,
                     "rel_l2": None, "cosine": None, "ref_l2": float(res[b]["nll"].mean())})
        ga, gb = res[a]["grads"], res[b]["grads"]
        for n in [k for k in gb if "layers.14.self_attn.q_proj" in k] :
            add(cmp, "grad " + n.replace("base_model.model.model.", "").replace(".default", ""), ga[n], gb[n])
        flat_a = torch.cat([ga[n].flatten() for n in sorted(gb)])
        flat_b = torch.cat([gb[n].flatten() for n in sorted(gb)])
        add(cmp, "grad all LoRA params (concatenated)", flat_a, flat_b)

    with open(out / "X02_layer_parity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def fmt(v):
        return "" if v is None else (f"{v:.3e}" if isinstance(v, float) else str(v))

    md = ["# X02 first pass: eager BF16 (A) vs TE BF16 (B) vs TE FP8 (C), Qwen2.5-1.5B + LoRA r=16", "",
          f"One fixed validation microbatch ({MICRO} x 512 tokens, val blocks 0-7). Layers {LAYER_IDX}. "
          f"C after {WARMUP} FP8 warmup forwards on val blocks {MICRO}-{MICRO * (WARMUP + 1) - 1}; "
          "C_cold is the first FP8 pass ever (empty amax history).",
          f"Frozen weights identical across A and B/C models: {not weight_mismatch} "
          f"({len(weight_mismatch)} mismatched tensors). TE conversion: "
          f"{sum(r['converted_to_te'] for r in coverage)} of {len(coverage)} nn.Linear converted.",
          f"Mean loss: A {float(res['A']['nll'].mean()):.6f}, B {float(res['B']['nll'].mean()):.6f}, "
          f"C {float(res['C']['nll'].mean()):.6f}, C_cold {float(res['C_cold']['nll'].mean()):.6f}.", "",
          "Single checkpoint (pretrained + fixed LoRA), single batch, single seed: an instantaneous perturbation "
          "measurement (§12.7), not evidence about training outcomes.", "",
          "| comparison | scope | L_inf | median abs | rel L2 | cosine |", "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['comparison']} | {r['scope']} | {fmt(r['linf'])} | {fmt(r['median_abs'])} | "
                  f"{fmt(r['rel_l2'])} | {fmt(r['cosine'])} |")
    (out / "X02_layer_parity.md").write_text("\n".join(md) + "\n")
    (out / "X02_backend.json").write_text(json.dumps(
        pol["fp8_te"].manifest(te_model), indent=2, default=str) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
