"""Compare two or more runs update-by-update (e.g. 1-GPU vs 4-GPU, same seed/config).

    python scripts/p3_compare_runs.py runs/P3_..._N1_* runs/P3_..._N4_* [--tol 2e-3]

Prints max |Δ train loss| and max |Δ val loss| against the FIRST run, plus status/agreement flags.
Pass criterion for P3-B (handbook §17.2): differences within the non-associative-arithmetic
tolerance you choose BEFORE looking (default: 2e-3 nats for BF16, use ~1e-5 for FP32).
"""
import argparse
import csv
import json
import os
import sys


def load(run):
    rows = lambda f: list(csv.DictReader(open(os.path.join(run, f)))) if os.path.exists(os.path.join(run, f)) else []
    rj = json.load(open(os.path.join(run, "run.json")))
    tr = {int(r["successful_update"]): float(r["loss"]) for r in rows("train.csv") if r["optimizer_step_skipped"] == "False"}
    ev = {int(r["successful_update"]): float(r["val_loss"]) for r in rows("eval.csv")}
    return rj, tr, ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--tol", type=float, default=2e-3)
    a = ap.parse_args()
    ref_rj, ref_tr, ref_ev = load(a.runs[0])
    ok = True
    print(f"reference: {a.runs[0]}  (N={ref_rj['env']['world_size']}, status={ref_rj['status']})")
    for run in a.runs[1:]:
        rj, tr, ev = load(run)
        common_t = sorted(set(ref_tr) & set(tr))
        common_e = sorted(set(ref_ev) & set(ev))
        dt = max((abs(ref_tr[k] - tr[k]) for k in common_t), default=float("nan"))
        de = max((abs(ref_ev[k] - ev[k]) for k in common_e), default=float("nan"))
        passed = (rj["status"] == "COMPLETED" and rj.get("ranks_agree_on_params", False)
                  and dt <= a.tol and (de <= a.tol or not common_e))
        ok &= passed
        print(f"{'PASS' if passed else 'FAIL'}  {run}  N={rj['env']['world_size']}  "
              f"max|Δtrain|={dt:.2e} over {len(common_t)} updates  max|Δval|={de:.2e} over {len(common_e)} evals  "
              f"status={rj['status']} ranks_agree={rj.get('ranks_agree_on_params')}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
