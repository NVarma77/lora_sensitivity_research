# Person 2 midterm status: precision policies and FP8 (Oct 31, 2026)

Scope: end of "Oct 26–Nov 1: connect the parts" (task plan; handbook gate G2). Tasks 4–5
(seed-1 X01 matrix, stability analysis) start after Nov 2.

All Qwen runs below use the **P2 stand-in runner** (`src/pilot_runner.py`, ADR-001), not P1's shared
trainer, which was not available. They are pilots that validate the precision switch, not X01 cells.
Setup: Qwen2.5-1.5B base @ `8faed761`, WikiText-103 raw v1 @ `b08601e0` (8M-token train prefix,
first 128 val blocks), LoRA r=16 alpha=16 on q_proj/v_proj, AdamW lr 2e-4, 5% warmup + cosine,
global batch 32 x 512 (micro 8 x accum 4), 200 updates, seed 1, one GH200, TF32 off.
Environment: torch 2.10.0+cu129, transformers 4.57.6, peft 0.21.2, Transformer Engine 2.20.2,
cuDNN 9.27 (see docs/precision.md, "Environment").

## 1. FP32 / BF16 / FP16 with FP16 gradient scaling + pilot curves

- Code: `src/precision.py` (`build_precision_policy`), 24 unit tests in `tests/test_precision.py`
  (pass on GH200: `results/p2_midterm/*/pytest.txt`). FP16 uses `torch.amp.GradScaler`
  (init 2^16), unscale once after all microbatches, clip after unscale, skips detected from the
  optimizer step hook.
- **Pilot curves (all five modes): `results/p2_midterm/figures/pilot_loss_curves.png`**,
  summary table `pilot_summary.md` next to it. Runs: `pilots-3497564/` (BF16/FP16/TE) and `fp32-3497606/`.
- Result (seed 1, 200 updates): all modes COMPLETED, 0 skipped updates, only LoRA weights changed.

| mode | val loss @0 | val loss @200 | tokens/s | skipped |
|---|---|---|---|---|
| BF16 AMP | 2.4003 | 2.1908 | 28.7k | 0 |
| FP16 AMP + GradScaler | 2.3997 | 2.1899 | 29.2k | 0 (scale stayed 65536) |
| TE BF16 (control) | 2.4003 | 2.1905 | 24.8k | 0 |
| TE FP8 | 2.4075 | 2.2007 | 21.0k | 0 |
| FP32 (TF32 off) | 2.3997 | 2.1899 | 6.2k | 0 |

## 2. FP8 through Transformer Engine on GH200, with profiler proof

- Code: `src/fp8_backend.py` (recipe, `te.autocast` wrapper, eligibility, TE conversion prototype,
  kernel classifier); modes `fp8_te` and `bf16_te` (backend-matched control) in `precision.py`.
- Recipe: `DelayedScaling(HYBRID: E4M3 fwd / E5M2 grads, amax_history_len=16, max)`. 196 of 197
  nn.Linear (decoder q/k/v/o/gate/up/down x 28) run as te.Linear; lm_head (tied), attention,
  norms, LoRA adapters run BF16/FP32.
- **Profiler figure: `results/p2_midterm/proof-3497550/fp8_proof/fp8_profiler_evidence.png`**
  (torch.profiler/CUPTI, one training update, FP8 vs TE BF16 control on the same axis).
  Kernel tables: `X02_kernel_evidence.md`. Nsight Systems capture of the same update:
  `nsys_fp8_update.nsys-rep` (open in the Nsight Systems GUI for a screenshot) and
  `nsys_fp8_update_cuda_gpu_kern_sum.csv`.
- Evidence: 1,556 FP8 GEMM launches per update (`nvjet_qqtst_*` forward, `nvjet_qrtst_*` backward),
  E4M3/E5M2 `cast_transpose` kernels and `amax_and_scale_update`, none of which appear in the
  TE BF16 control (0 FP8 GEMMs). Toy + Qwen-shape single-layer proof: `toy/kernel_evidence.md`
  (PASS at all 5 shapes).
- Not a speedup: FP8 cuts GEMM time per update (175 ms -> 120 ms) but end-to-end it is slower
  (21.0k vs 24.8k tokens/s for TE BF16) because of cast/scaling kernels and launch overhead on
  LoRA-sized work (handbook H2, §18.2). To be measured properly in X05 (P3).

## 3. FP8 + LoRA test run: only LoRA weights update

- Run: `results/p2_midterm/pilots-3497564/pilots/P2PILOT-qwen15b-wt103-r16-fp8_te-s1-g1-3497564/`
  (200 updates; also the 30-update `fp8_proof/smoke/smoke-fp8_te`).
- `param_audit.json`: 112 trainable tensors (all `lora_A`/`lora_B`, 2,179,072 params) all changed;
  338 frozen tensors unchanged (max abs change 0.0); no frozen tensor received a gradient;
  optimizer param set == trainable set. `coverage.json`: every LoRA wrapper is PEFT `TeLinear` on
  a `te.Linear` base.
- **Needs P1**: re-run this audit once in P1's trainer (task plan item P1-3 / P2-2 is joint).

## Nice to have: first BF16 vs FP8 error on one layer and batch

`results/p2_midterm/proof-3497550/fp8_proof/x02/X02_layer_parity.md` (+ NOTE.md). Fixed
pretrained weights + fixed LoRA, one val microbatch (8 x 512):

- TE BF16 vs eager BF16 forward: bit-identical (logits error 0); LoRA grads rel L2 7e-3. Conversion is
  exact (handbook T09).
- TE FP8 vs TE BF16 (after 4 warmup passes): single layer rel L2 1.8–3.5% (cosine >= 0.9994);
  logits rel L2 9.1%; mean loss +0.026 nats; all-LoRA gradient rel L2 23%, cosine 0.973.
- First-ever FP8 pass (empty amax history): LoRA gradient cosine 0.27 vs TE BF16. Candidate lead for
  the stability analysis: early FP8 updates before delayed scaling has history.
- Pilot curves agree: the FP8 val-loss gap is 0.0072 at update 0 (pure evaluation error) and 0.0102
  at update 200. One seed, so descriptive only.

## Open items / blockers

- P1's trainer: swap the stand-in out; re-run one FP8+LoRA audit and one short pilot per mode there.
- Batch queue: `ghx4` 1-GPU batch jobs were estimated to start ~Nov 8 on Oct 31; interactive QOS
  allows one job per user. Plan the Nov 2–15 matrix around <=45-min interactive jobs or raise with
  the team/NCSA.
- Decisions needing team sign-off before Nov 5 (docs/precision.md open decisions, ADR-001): FP32
  base load, FP16 stop rule, FP8 recipe, cuDNN 9.27 override.
- Not committed to git yet.
