# Person 3 — multi-GPU training, scaling (X05) and full-FT control (X06)

Owner: Boubacar Issoufou Anaroua. Handbook: §5.2 distributed contract, §9 DeltaAI/DDP, §11 X05/X06, §17 P3 tasks, §19 tests T11–T14.
Builds directly on Person 2's `precision.py`, `fp8_backend.py` and the pilot data path in `pilot_runner.py`.

## Files added by Person 3 (nothing of P1/P2 is modified)

| Path | What it does |
|---|---|
| `src/distributed.py` | `build_execution_context(world_size, effective_global_batch, microbatch)`; NCCL/Gloo init (60-min timeout), world-size-independent `GlobalBatchSampler` (+ resume state), rank sharding, `no_sync` control, global-token loss scaling, token-weighted eval reduction, FP16 step agreement, cross-rank checksum, timers, peak memory (max over ranks) |
| `src/p3_ddp_runner.py` | Qwen2.5-1.5B LoRA **or** full FT on 1/2/4 GPUs with `torchrun`. Modes: train, `--bench_timed` (X05), `--no_sync_all` (comm-free timing control), `--profile`, `--method full` (X06). Writes `config.json`, `run.json`, `train.csv`, `eval.csv`, `precision.json`, `coverage.json`, `bench.json` (§5.4 fields) |
| `src/team_hooks.py` | The only place that calls teammates' code: `precision.build_precision_policy`, `fp8_backend.convert_linears_to_te` (before LoRA, like the pilots), `pilot_runner.build_dataset`. Swap to P1's `data.py`/`te_conversion.py` here when they land |
| `tests/test_distributed.py` | 11 single-process unit tests (run by the normal `pytest`) |
| `tests/ddp_toy_check.py` | torchrun check of T11/T12/T13 + cross-rank and cross-world-size parameter equality |
| `scripts/p3_ddp_job.sbatch` | One `torchrun` per node, N workers = GPUs requested; sources `scripts/env.sh`; logs topology, versions, GPU telemetry (`logs/system-<job>.csv`) |
| `scripts/p3_submit.sh` | `toy`, `smoke`, `x06smoke`, `x06`, `x05`, `x01` job sets |
| `scripts/p3_compare_runs.py` | 1-GPU vs N-GPU loss parity (P3-B pass/fail) |
| `scripts/p3_plot_scaling.py` | `X05_scaling.csv` + speed-up/efficiency figure from `bench.json` files |
| `results/p3_cpu_checks/` | CPU/Gloo evidence (see `SUMMARY.md`) |

## Fixed work across GPU counts (handbook §9.4)

Global batch **B = M × N × A = 32 × 512 tokens** in every run. Default micro-batch M = 8 (same as the P2 pilots):

| GPUs N | M | accumulation A |
|---|---|---|
| 1 | 8 | 4 |
| 2 | 8 | 2 |
| 4 | 8 | 1 |

* Update *u* uses positions `[32u, 32u+32)` of one seeded permutation → the same 32 blocks for N = 1, 2, 4 (T11).
* Each micro-batch backpropagates `Σ token NLL × N / T_global`; after DDP's mean over ranks the gradient equals the serial gradient of `Σ NLL / T_global` (T12). Same normalisation as `pilot_runner.py` (`s / denom`) when N = 1.
* `DDP.no_sync()` on all but the last micro-batch. Only `requires_grad` params get DDP buckets → LoRA all-reduces 2.18 M params (r=16, q/v).
* Validation all-reduces sums of NLL and tokens (T13).
* FP16: P2's policy detects skips with the optimizer-step hook; `agree_step` (MIN all-reduce) keeps all ranks consistent; skipped steps don't advance the LR schedule.

## Verified so far (CPU/Gloo — `results/p3_cpu_checks/SUMMARY.md`)

* Whole-repo `pytest`: 35 passed (P2's 24 + P3's 11), including P2's rule that no file outside `precision.py` calls autocast/GradScaler.
* `ddp_toy_check.py` PASS at 1/2/4 workers: max |Δgrad| vs serial 2.2e-15, identical parameter checksum across ranks and world sizes.
* Runner with a tiny random Qwen2 + PEFT LoRA through **P2's `precision.py`**: 1 vs 4 workers max |Δ train loss| 9e-8 (FP32), 0 (BF16), 4.7e-6 (FP16 + GradScaler).
* Not yet verified: CUDA/NCCL, TE modes, real Qwen2.5-1.5B, any timing. These are the first GH200 jobs.

## Fastest path: everything in one interactive session

```bash
srun --account=bcrc-dtai-gh --partition=ghx4-interactive --nodes=1 --gpus-per-node=4 \
     --cpus-per-task=64 --mem=400g --time=00:45:00 --pty bash
bash scripts/p3_tonight.sh        # tests, toy DDP 1/2/4, Qwen smoke 1 vs 4, mini X05, X06 memory smoke
```

## Run order on DeltaAI with batch jobs (repo root, after `source scripts/env.sh` works for you)

```bash
mkdir -p logs
PART=ghx4-interactive bash scripts/p3_submit.sh toy     # 1) PASS at N=1,2,4  -> grep -h "PASS\|FAIL" logs/p3-toy-N*.out
bash scripts/p3_submit.sh smoke                          # 2) Qwen LoRA r16 BF16, 50 updates, N=1 and N=4
python scripts/p3_compare_runs.py results/p3_runs/smoke/P3SMOKE-bf16-r16-s0-N1 results/p3_runs/smoke/P3SMOKE-bf16-r16-s0-N4
bash scripts/p3_submit.sh x06smoke                       # 3) full-FT memory check (expect ~25 GB static + activations)
PRECS="bf16 bf16_te fp8_te" bash scripts/p3_submit.sh x05   # 4) scaling, 3 repeats + profile + no-sync control
python scripts/p3_plot_scaling.py results/p3_x05 --out results/p3_x05/summary
bash scripts/p3_submit.sh x06                            # 5) full FT vs LoRA r16, FP32/BF16, 200 updates
bash scripts/p3_submit.sh x01                            # 6) seed-2 matrix — only after the Nov 5 protocol freeze
```

Queue note from P2 (Oct 31): `ghx4` 1-GPU batch jobs were estimated to start ~Nov 8; `ghx4-interactive` allows one
job per user (≤45 min). The toy and smoke jobs fit in interactive; X05/X06 should be queued on `ghx4` early.

## Full-FT memory estimate (X06, before measuring)

| Qwen2.5-1.5B, FP32 base loaded | weights | grads | AdamW m,v | static |
|---|---|---|---|---|
| Full FT (FP32 and BF16-AMP arms) | 6.2 GB | 6.2 GB | 12.4 GB | ≈ 24.7 GB + activations |
| LoRA r=16 q/v (2.18 M trainable) | 6.2 GB | 9 MB | 17 MB | ≈ 6.2 GB + activations |

## Open items for the team

* When P1's trainer lands it should call `distributed.py` (`build_execution_context`, `shard_global_batch`,
  `loss_for_backward`, `reduce_eval`); then X01 runs go through it with the same arguments.
* The pilot data prefix (8 M tokens ≈ 15.6 k blocks) is smaller than 800 updates × 32 blocks; the sampler
  wraps into a second seeded epoch. The final X01 data budget must be frozen by the team (Nov 5).
* Exact resume (T06) is owned jointly with P1's `checkpoints.py`.
