"""FP8 profiler evidence for the Qwen+LoRA run (handbook §8.5 step 7, X02_kernel_evidence, T08).

Usage: python scripts/fp8_profile_report.py FP8_RUN_DIR BF16_TE_RUN_DIR --out DIR [--nsys CSV]
Inputs are pilot_runner runs made with --profile-update (profile_kernels.csv from torch.profiler
for exactly one effective update). Writes fp8_profiler_evidence.png and X02_kernel_evidence.md.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CLASS_COLORS = {"fp8_gemm": "#2a78d6", "fp8_cast_or_scaling": "#eb6834", "other_gemm": "#1baf7a", "other": "#b5b3ad"}
CLASS_LABELS = {"fp8_gemm": "FP8 GEMM", "fp8_cast_or_scaling": "FP8 cast / amax / scaling",
                "other_gemm": "non-FP8 GEMM", "other": "other kernels"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
TOP = 14


def read(run_dir):
    with open(Path(run_dir) / "profile_kernels.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["calls"] = int(r["calls"])
        r["device_time_us"] = float(r["device_time_us"])
    return rows


def by_class(rows):
    agg = {c: [0, 0.0] for c in CLASS_COLORS}
    for r in rows:
        agg[r["class"]][0] += r["calls"]
        agg[r["class"]][1] += r["device_time_us"]
    return agg


def short(name, n=70):
    return name if len(name) <= n else name[: n - 1] + "…"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fp8_run")
    ap.add_argument("bf16_te_run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nsys", default=None, help="nsys stats cuda_gpu_kern_sum CSV for the FP8 run")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fp8, ctl = read(args.fp8_run), read(args.bf16_te_run)
    ctl_names = {r["kernel"] for r in ctl}
    fp8_only = [r for r in fp8 if r["kernel"] not in ctl_names]
    run = json.loads((Path(args.fp8_run) / "run.json").read_text())
    prec = json.loads((Path(args.fp8_run) / "precision.json").read_text())
    te_info = prec.get("transformer_engine", {})
    cov = json.loads((Path(args.fp8_run) / "coverage.json").read_text())
    n_te = sum(r["converted_to_te"] for r in cov["te_conversion"])
    n_lin = len(cov["te_conversion"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), facecolor=SURFACE)
    for ax, rows, title in ((axes[0], fp8, "TE FP8 (fp8_te)"), (axes[1], ctl, "TE BF16 control (bf16_te), same modules")):
        top = sorted(rows, key=lambda r: -r["device_time_us"])[:TOP][::-1]
        y = range(len(top))
        ax.barh(list(y), [r["device_time_us"] / 1000 for r in top], color=[CLASS_COLORS[r["class"]] for r in top],
                height=0.7, edgecolor=SURFACE, linewidth=2)
        ax.set_yticks(list(y))
        ax.set_yticklabels([f"{short(r['kernel'])}  ×{r['calls']}" for r in top], fontsize=7, color=INK,
                           family="monospace")
        total = sum(r["device_time_us"] for r in rows) / 1000
        agg = by_class(rows)
        fp8_gemm_share = agg["fp8_gemm"][1] / max(1e-9, agg["fp8_gemm"][1] + agg["other_gemm"][1])
        ax.set_title(f"{title}\nGPU time {total:.1f} ms in one effective update (4 microbatches); "
                     f"FP8 share of GEMM time {fp8_gemm_share:.0%}", loc="left", fontsize=10, color=INK)
        ax.set_xlabel("device time (ms)", color=INK2, fontsize=9)
        ax.set_facecolor(SURFACE)
        ax.grid(True, axis="x", color=GRID, linewidth=0.6)
        ax.tick_params(colors=INK2, labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CLASS_COLORS.values()]
    fig.legend(handles, CLASS_LABELS.values(), loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle(
        f"torch.profiler (CUPTI) kernel trace, Qwen2.5-1.5B + LoRA r=16, update {run.get('profile_update', '')} "
        f"| {run['environment']['device']} | TE {te_info.get('te_version')} | recipe {te_info.get('recipe')} "
        f"| {n_te}/{n_lin} nn.Linear converted to te.Linear | job {run['environment']['slurm_job_id']}",
        x=0.01, ha="left", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(out / "fp8_profiler_evidence.png", dpi=150, facecolor=SURFACE)

    a8, ac = by_class(fp8), by_class(ctl)
    md = ["# X02 kernel evidence: FP8 kernels run in the Qwen + LoRA training step", "",
          f"FP8 run `{Path(args.fp8_run).name}`, control `{Path(args.bf16_te_run).name}`. Both profiled for one "
          "effective update (4 microbatches x 8 x 512 tokens, forward + backward + optimizer) with torch.profiler "
          "(CUDA activity via CUPTI). Kernel classes are name-based (`fp8_backend.classify_kernel`); the "
          "FP8-only set below is a direct diff against the control and does not depend on the classifier.", "",
          f"TE {te_info.get('te_version')}, `{te_info.get('autocast_api')}`, recipe {te_info.get('recipe')}, "
          f"fp8_available={te_info.get('fp8_available')}. {n_te} of {n_lin} nn.Linear converted "
          "(decoder q/k/v/o/gate/up/down x 28 layers); lm_head (tied embedding) stays eager BF16.", "",
          "| kernel class | FP8 run calls | FP8 run ms | control calls | control ms |", "|---|---|---|---|---|"]
    for c in CLASS_COLORS:
        md.append(f"| {CLASS_LABELS[c]} | {a8[c][0]} | {a8[c][1] / 1000:.2f} | {ac[c][0]} | {ac[c][1] / 1000:.2f} |")
    md += ["", f"## Kernels launched only in the FP8 run ({len(fp8_only)})", "",
           "| kernel | calls | ms | class |", "|---|---|---|---|"]
    for r in sorted(fp8_only, key=lambda r: -r["device_time_us"])[:40]:
        md.append(f"| `{r['kernel'][:150]}` | {r['calls']} | {r['device_time_us'] / 1000:.3f} | {r['class']} |")
    verdict = a8["fp8_gemm"][0] > 0 and ac["fp8_gemm"][0] == 0
    md += ["", f"**Verdict: {'PASS' if verdict else 'CHECK MANUALLY'}**: "
           f"{a8['fp8_gemm'][0]} FP8 GEMM launches in the FP8 run, {ac['fp8_gemm'][0]} in the TE BF16 control."]
    if args.nsys and Path(args.nsys).exists():
        with open(args.nsys) as f:
            ns = list(csv.DictReader(f))
        key = "Name" if ns and "Name" in ns[0] else list(ns[0])[-1]
        md += ["", "## Nsight Systems cross-check (cuda_gpu_kern_sum, same FP8 update)", "",
               "| kernel | instances | total ns |", "|---|---|---|"]
        for r in ns[:20]:
            md.append(f"| `{r[key][:150]}` | {r.get('Instances', '')} | {r.get('Total Time (ns)', '')} |")
    (out / "X02_kernel_evidence.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
