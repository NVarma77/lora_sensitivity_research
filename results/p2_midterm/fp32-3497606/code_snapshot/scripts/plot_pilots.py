"""Pilot loss curves + summary table for the P2 precision pilots.

Usage: python scripts/plot_pilots.py RUN_DIR [RUN_DIR ...] --out DIR
Each RUN_DIR is a pilot_runner output (config.json, run.json, train.csv, eval.csv).
Writes pilot_loss_curves.png, pilot_fp16_scale.png (if an fp16 run is present) and
pilot_summary.md. Curves are plotted against successful optimizer updates (§12.2);
raw per-update training loss is kept (light) with a 10-update moving average on top.
"""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Reference categorical palette (dataviz skill), fixed order by mode, never cycled.
COLORS = {"fp32": "#2a78d6", "bf16": "#eb6834", "fp16": "#1baf7a", "bf16_te": "#eda100", "fp8_te": "#e87ba4"}
LABELS = {"fp32": "FP32", "bf16": "BF16 AMP", "fp16": "FP16 AMP + GradScaler", "bf16_te": "TE BF16 (control)",
          "fp8_te": "TE FP8"}
ORDER = list(COLORS)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def read_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def load(run_dir):
    d = Path(run_dir)
    cfg = json.loads((d / "config.json").read_text())
    return {
        "dir": d, "cfg": cfg, "run": json.loads((d / "run.json").read_text()),
        "mode": cfg["precision"]["mode"], "train": read_csv(d / "train.csv"), "eval": read_csv(d / "eval.csv"),
        "audit": json.loads((d / "param_audit.json").read_text()) if (d / "param_audit.json").exists() else {},
    }


def style(ax, title, ylabel, xlabel="successful optimizer updates"):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(colors=INK2, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def moving_avg(y, k=10):
    out = []
    for i in range(len(y)):
        w = [v for v in y[max(0, i - k + 1): i + 1] if math.isfinite(v)]
        out.append(sum(w) / len(w) if w else math.nan)
    return out


def direct_labels(ax, ends):
    """Label line ends, nudging apart so labels don't collide."""
    ends = sorted(ends, key=lambda e: e[1])
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.045
    placed = []
    for x, y, text in ends:
        if placed and y - placed[-1] < gap:
            y = placed[-1] + gap
        placed.append(y)
        ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Qwen2.5-1.5B + LoRA r=16 on WikiText-103: P2 precision pilots (seed 1, 1x GH200)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = sorted((load(r) for r in args.runs), key=lambda r: ORDER.index(r["mode"]))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=SURFACE)
    ends1, ends2 = [], []
    for r in runs:
        m, c = r["mode"], COLORS[r["mode"]]
        steps = [t for t in r["train"] if t["optimizer_step_skipped"] == "False"]
        x = [int(t["successful_update"]) for t in steps]
        y = [float(t["loss"]) for t in steps]
        a1.plot(x, y, color=c, linewidth=0.8, alpha=0.25)
        ma = moving_avg(y)
        a1.plot(x, ma, color=c, linewidth=2, label=LABELS[m])
        if x:
            ends1.append((x[-1], ma[-1], LABELS[m]))
        ex = [int(e["successful_update"]) for e in r["eval"]]
        ey = [float(e["val_loss"]) for e in r["eval"]]
        a2.plot(ex, ey, color=c, linewidth=2, marker="o", markersize=4, label=LABELS[m])
        if ex:
            ends2.append((ex[-1], ey[-1], f"{LABELS[m]} {ey[-1]:.4f}"))
    style(a1, "Training loss (10-update moving average; raw faint)", "token cross-entropy (nats)")
    style(a2, "Validation loss (first 128 val blocks, 65,408 targets)", "token cross-entropy (nats)")
    for ax, ends in ((a1, ends1), (a2, ends2)):
        xmax = max(e[0] for e in ends) if ends else 1
        ax.set_xlim(0, xmax * 1.32)
        direct_labels(ax, ends)
    a1.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK)
    fig.suptitle(args.title, x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "pilot_loss_curves.png", dpi=150, facecolor=SURFACE)

    fp16 = [r for r in runs if r["mode"] == "fp16"]
    if fp16:
        r = fp16[0]
        fig, ax = plt.subplots(figsize=(7, 3.4), facecolor=SURFACE)
        x = [int(t["attempted_step"]) for t in r["train"]]
        s = [float(t["scale_after"]) for t in r["train"]]
        ax.step(x, s, where="post", color=COLORS["fp16"], linewidth=2)
        skips = [int(t["attempted_step"]) for t in r["train"] if t["optimizer_step_skipped"] == "True"]
        for k in skips:
            ax.axvline(k, color=INK2, linewidth=0.8, linestyle=":")
        ax.set_yscale("log", base=2)
        style(ax, f"FP16 GradScaler scale after each attempt ({len(skips)} skipped updates, dotted)",
              "loss scale", "attempted optimizer steps")
        fig.tight_layout()
        fig.savefig(out / "pilot_fp16_scale.png", dpi=150, facecolor=SURFACE)

    md = ["# P2 precision pilots: summary", "",
          "Stand-in runner (src/pilot_runner.py), not the shared X01 trainer. One seed, short budget: these show the "
          "precision switch works on the real model; they are not precision comparisons.", "",
          "| mode | status | attempted | successful | skipped | final train loss (mean last 10) | final val loss | "
          "median tokens/s | peak alloc GiB | only LoRA updated | run id |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        tr = [t for t in r["train"] if t["optimizer_step_skipped"] == "False"]
        last = [float(t["loss"]) for t in tr[-10:]]
        warm = [t for t in tr if int(t["successful_update"]) > 5]
        tps = statistics.median(float(t["loss_denominator"]) / float(t["step_seconds"]) for t in warm) if warm else math.nan
        run = r["run"]
        md.append(
            f"| {r['mode']} | {run['status']} | {run.get('attempted_steps')} | {run.get('successful_updates')} | "
            f"{run.get('skipped_steps')} | {sum(last) / len(last):.4f} | {float(r['eval'][-1]['val_loss']):.4f} | "
            f"{tps:,.0f} | {run.get('peak_allocated_bytes', 0) / 2**30:.1f} | {r['audit'].get('only_lora_weights_updated')} | "
            f"`{r['dir'].name}` |"
        )
    md += ["", "Tokens/s = valid target tokens per effective update / wall step time (updates > 5, includes "
           "optimizer and Python overhead, excludes eval). FP32 has TF32 disabled."]
    (out / "pilot_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
