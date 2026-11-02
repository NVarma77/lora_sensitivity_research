"""X05 — aggregate bench.json files into X05_scaling.csv and plot speed-up / efficiency.

    python scripts/p3_plot_scaling.py results/p3_x05 --out results/p3_x05/summary

Speed-up S(N) = T1 / TN and efficiency E(N) = S(N) / N use the median time of one EFFECTIVE update
(identical global work: 32 x 512 tokens) — handbook §9.6. Repeats (rep1..3) are pooled per config:
we report the median of per-job medians and the min–max across jobs.
"""
import argparse
import csv
import glob
import json
import os
import statistics as st
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="results/X05")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = []
    for f in glob.glob(os.path.join(a.root, "*", "bench.json")):
        b = json.load(open(f))
        b["run"] = os.path.basename(os.path.dirname(f))
        rows.append(b)
    if not rows:
        raise SystemExit(f"no bench.json under {a.root}")

    groups = defaultdict(list)
    for b in rows:
        if "profile" in b["run"]:
            continue  # profiler adds overhead; traces only
        groups[(b["precision"], b["world_size"], bool(b.get("no_sync_all")))].append(b)

    table = []
    for (prec, n, nosync), bs in sorted(groups.items()):
        med = st.median(x["median_s"] for x in bs)
        table.append({"precision": prec, "gpus": n, "no_sync_control": nosync, "jobs": len(bs),
                      "step_median_s": med, "step_min_job_s": min(x["median_s"] for x in bs),
                      "step_max_job_s": max(x["median_s"] for x in bs),
                      "p25_s": st.median(x["p25_s"] for x in bs), "p75_s": st.median(x["p75_s"] for x in bs),
                      "tokens_per_s": bs[0]["tokens_per_update"] / med,
                      "peak_alloc_GB": max((x.get("peak_allocated_bytes") or 0) for x in bs) / 1e9,
                      "peak_reserved_GB": max((x.get("peak_reserved_bytes") or 0) for x in bs) / 1e9})
    base = {r["precision"]: r["step_median_s"] for r in table if r["gpus"] == 1 and not r["no_sync_control"]}
    for r in table:
        if r["precision"] in base:
            r["speedup"] = base[r["precision"]] / r["step_median_s"]
            r["efficiency"] = r["speedup"] / r["gpus"]
    # exposed communication estimate: T(sync) - T(no_sync) at the same N
    sync = {(r["precision"], r["gpus"]): r["step_median_s"] for r in table if not r["no_sync_control"]}
    for r in table:
        if r["no_sync_control"] and (r["precision"], r["gpus"]) in sync:
            r["exposed_comm_s"] = sync[(r["precision"], r["gpus"])] - r["step_median_s"]

    keys = sorted({k for r in table for k in r}, key=lambda k: list(table[0]).index(k) if k in table[0] else 99)
    with open(os.path.join(a.out, "X05_scaling.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(table)
    print(open(os.path.join(a.out, "X05_scaling.csv")).read())

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for prec in sorted({r["precision"] for r in table}):
        rs = sorted([r for r in table if r["precision"] == prec and not r["no_sync_control"]], key=lambda r: r["gpus"])
        n = [r["gpus"] for r in rs]
        ax[0].plot(n, [r["tokens_per_s"] for r in rs], "o-", label=prec.upper())
        ax[1].plot(n, [r.get("speedup", float("nan")) for r in rs], "o-", label=prec.upper())
        ax[2].plot(n, [r.get("efficiency", float("nan")) for r in rs], "o-", label=prec.upper())
    ax[1].plot([1, 4], [1, 4], "k--", lw=0.8, label="ideal")
    ax[2].axhline(1.0, color="k", ls="--", lw=0.8)
    for a_, t in zip(ax, ["Throughput (tokens/s)", "Speed-up S(N)=T1/TN", "Efficiency E(N)=S(N)/N"]):
        a_.set_title(t); a_.set_xlabel("GPUs (GH200)"); a_.set_xticks([1, 2, 4]); a_.grid(alpha=0.3); a_.legend()
    fig.suptitle("X05 — LoRA r=16, Qwen2.5-1.5B, fixed global batch 32x512")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "X05_speedup_efficiency.png"), dpi=160)


if __name__ == "__main__":
    main()
