# Note on X02_layer_parity.* in this run (added 2026-10-31, files left unmodified)

The `C_cold_vs_B ... isolated layers.*` rows are NOT cold-state measurements: the isolated-layer
probes ran after the FP8 warmup passes, so they duplicate the warmed `C_vs_B` isolated rows.
All other C_cold rows (in-network outputs, logits, NLL, loss, LoRA grads) come from the first FP8
pass and are valid. scripts/layer_error_check.py no longer emits isolated rows for C_cold.
