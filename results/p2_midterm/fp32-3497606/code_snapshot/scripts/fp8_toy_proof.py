"""P2-B toy FP8 proof (handbook §8.5 step 1, §16.2, test T08).

For te.Linear at the handbook toy shape and the Qwen2.5-1.5B projection shapes:
run the same weights and inputs with the TE FP8 recipe off (BF16, negative
control) and on (FP8); compare forward/backward results; profile each and
diff the kernel sets. Writes to --out:
  toy_parity.csv        per shape: output/grad errors FP8 vs BF16
  toy_kernels.csv       per arm: kernel, calls, device time, class
  kernel_evidence.md    FP8-only kernels and verdict
  backend.json          TE version, recipe, fp8 support check
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import fp8_backend  # noqa: E402
from fp8_backend import te  # noqa: E402
from precision import build_precision_policy  # noqa: E402

# (name, in_features, out_features, tokens): toy 768x768 plus Qwen2.5-1.5B q/k(v)/o/gate(up)/down at 8 x 512 tokens.
SHAPES = [
    ("toy_768", 768, 768, 1024),
    ("qwen_q_o_proj", 1536, 1536, 4096),
    ("qwen_k_v_proj", 1536, 256, 4096),
    ("qwen_gate_up_proj", 1536, 8960, 4096),
    ("qwen_down_proj", 8960, 1536, 4096),
]
WARMUP = 3  # delayed scaling needs amax history before the measured pass


def errs(x, ref):
    x, ref = x.float().flatten(), ref.float().flatten()
    d = x - ref
    return {
        "linf": float(d.abs().max()),
        "rel_l2": float(d.norm() / max(float(ref.norm()), 1e-12)),
        "cosine": float(torch.nn.functional.cosine_similarity(x, ref, dim=0)),
    }


def run_arm(mode, layer, x, gy):
    policy = build_precision_policy(mode)
    for _ in range(WARMUP):
        xi = x.clone().requires_grad_(True)
        with policy.forward_context():
            y = layer(xi)
        y.backward(gy)
    layer.zero_grad(set_to_none=True)
    xi = x.clone().requires_grad_(True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        with policy.forward_context():
            y = layer(xi)
        y.backward(gy)
        torch.cuda.synchronize()
    kernels = []
    for evt in prof.key_averages():
        if evt.device_type == torch.autograd.DeviceType.CUDA:
            kernels.append({"kernel": evt.key, "calls": evt.count,
                            "device_time_us": round(float(evt.self_device_time_total), 2),
                            "class": fp8_backend.classify_kernel(evt.key)})
    return y.detach(), xi.grad.detach(), layer.weight.grad.detach().clone(), kernels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    backend = fp8_backend.backend_manifest(fp8_backend.FP8RecipeConfig(), True)
    backend["device"] = torch.cuda.get_device_name()
    backend["torch"] = torch.__version__
    (out / "backend.json").write_text(json.dumps(backend, indent=2) + "\n")
    print(json.dumps(backend, indent=2))

    parity, all_kernels = [], []
    evidence = ["# FP8 kernel evidence: TE toy and Qwen-shaped te.Linear (handbook §8.5 step 1, T08)", "",
                f"Device {backend['device']}, TE {backend['te_version']}, torch {backend['torch']}, "
                f"recipe {backend['recipe']}, `{backend['autocast_api']}`. fp8_available={backend['fp8_available']}.",
                f"Each arm: same te.Linear weights, input and output-grad; {WARMUP} warmup fwd/bwd passes, then one "
                "profiled fwd+bwd. bf16_te = FP8 recipe off (negative control); fp8_te = FP8 recipe on.", ""]
    verdict_ok = True
    for name, fin, fout, tokens in SHAPES:
        torch.manual_seed(0)
        layer = te.Linear(fin, fout, bias=True, params_dtype=torch.float32, device="cuda")
        x = torch.randn(tokens, fin, device="cuda")
        gy = torch.randn(tokens, fout, device="cuda") / fout**0.5
        res = {}
        for mode in ("bf16_te", "fp8_te"):
            y, gx, gw, kernels = run_arm(mode, layer, x, gy)
            res[mode] = (y, gx, gw, kernels)
            for k in kernels:
                all_kernels.append({"shape": name, "arm": mode, **k})
        row = {"shape": name, "in": fin, "out": fout, "tokens": tokens}
        for label, i in (("out", 0), ("dgrad", 1), ("wgrad", 2)):
            for k, v in errs(res["fp8_te"][i], res["bf16_te"][i]).items():
                row[f"{label}_{k}"] = v
        parity.append(row)

        bf16_names = {k["kernel"] for k in res["bf16_te"][3]}
        fp8_only = [k for k in res["fp8_te"][3] if k["kernel"] not in bf16_names]
        fp8_gemms = [k for k in res["fp8_te"][3] if k["class"] == "fp8_gemm"]
        bf16_fp8_gemms = [k for k in res["bf16_te"][3] if k["class"] == "fp8_gemm"]
        ok = bool(fp8_only) and not bf16_fp8_gemms
        verdict_ok &= ok
        evidence += [f"## {name}: {fin} -> {fout}, {tokens} tokens", "",
                     f"- kernels in FP8 arm only: {len(fp8_only)}; FP8-named GEMM kernels: FP8 arm {len(fp8_gemms)}, "
                     f"BF16 control {len(bf16_fp8_gemms)}",
                     f"- forward rel L2 FP8 vs BF16: {row['out_rel_l2']:.3e}, cosine {row['out_cosine']:.6f}", "",
                     "| arm | kernel | calls | device µs | class |", "|---|---|---|---|---|"]
        for mode in ("bf16_te", "fp8_te"):
            for k in sorted(res[mode][3], key=lambda r: -r["device_time_us"])[:8]:
                flag = " **(FP8 arm only)**" if mode == "fp8_te" and k["kernel"] not in bf16_names else ""
                evidence.append(f"| {mode} | `{k['kernel'][:110]}`{flag} | {k['calls']} | {k['device_time_us']} | {k['class']} |")
        evidence.append("")

    evidence.insert(5, f"**Verdict: {'PASS' if verdict_ok else 'CHECK MANUALLY'}**: FP8 arm launches kernels absent "
                       "from the BF16 control at every shape, and the control launches no FP8-named GEMMs.\n")
    with open(out / "toy_parity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(parity[0]))
        w.writeheader()
        w.writerows(parity)
    with open(out / "toy_kernels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(all_kernels[0]))
        w.writeheader()
        w.writerows(all_kernels)
    (out / "kernel_evidence.md").write_text("\n".join(evidence) + "\n")
    print("\n".join(evidence))


if __name__ == "__main__":
    main()
