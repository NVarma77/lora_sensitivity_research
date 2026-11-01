# FP8 kernel evidence: TE toy and Qwen-shaped te.Linear (handbook §8.5 step 1, T08)

Device NVIDIA GH200 120GB, TE 2.20.2, torch 2.10.0+cu129, recipe {'name': 'delayed_scaling', 'fp8_format': 'HYBRID', 'amax_history_len': 16, 'amax_compute_algo': 'max', 'margin': 0}, `transformer_engine.pytorch.autocast`. fp8_available=True.
Each arm: same te.Linear weights, input and output-grad; 3 warmup fwd/bwd passes, then one profiled fwd+bwd. bf16_te = FP8 recipe off (negative control); fp8_te = FP8 recipe on.

**Verdict: PASS**: FP8 arm launches kernels absent from the BF16 control at every shape, and the control launches no FP8-named GEMMs.

## toy_768: 768 -> 768, 1024 tokens

- kernels in FP8 arm only: 8; FP8-named GEMM kernels: FP8 arm 3, BF16 control 0
- forward rel L2 FP8 vs BF16: 3.755e-02, cosine 0.999295

| arm | kernel | calls | device µs | class |
|---|---|---|---|---|
| bf16_te | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{la` | 3 | 8.74 | other |
| bf16_te | `void at::native::vectorized_elementwise_kernel<8, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase` | 4 | 6.5 | other |
| bf16_te | `nvjet_tst_96x64_64x10_1x2_h_bz_bias_TNN` | 1 | 5.31 | other_gemm |
| bf16_te | `nvjet_tst_64x96_64x8_4x2_h_bz_NTT` | 1 | 4.99 | other_gemm |
| bf16_te | `nvjet_tst_96x64_64x8_1x2_h_bz_NNN` | 1 | 4.77 | other_gemm |
| bf16_te | `void cublasLt::epilogue::impl::globalKernelBgradAB_v1<8, 64, __nv_bfloat16, __nv_bfloat16, float>(int, int, in` | 1 | 2.37 | other |
| fp8_te | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{la` | 3 | 8.51 | other |
| fp8_te | `transformer_engine::delayed_scaling_recipe::(anonymous namespace)::amax_and_scale_update_impl::kernel_bulk(flo` **(FP8 arm only)** | 2 | 5.57 | fp8_cast_or_scaling |
| fp8_te | `cast_transpose_optimized_kernel(float const*, float const*, __nv_fp8_e4m3*, __nv_fp8_e4m3*, float const*, floa` **(FP8 arm only)** | 2 | 5.09 | fp8_cast_or_scaling |
| fp8_te | `nvjet_qrtst_72x64_128x12_1x2_h_bz_algo2_TNN` **(FP8 arm only)** | 1 | 4.9 | fp8_gemm |
| fp8_te | `nvjet_qqtst_96x64_128x10_1x2_h_bz_bias_TNN` **(FP8 arm only)** | 1 | 4.8 | fp8_gemm |
| fp8_te | `nvjet_qrtst_96x64_128x10_1x2_h_bz_algo2_TNN` **(FP8 arm only)** | 1 | 4.45 | fp8_gemm |
| fp8_te | `cast_transpose_fusion_kernel_optimized(transformer_engine::CTDBiasDActParam<__nv_bfloat16, __nv_bfloat16, __nv` **(FP8 arm only)** | 1 | 3.01 | fp8_cast_or_scaling |
| fp8_te | `void at::native::vectorized_elementwise_kernel<8, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase` | 2 | 2.85 | other |

## qwen_q_o_proj: 1536 -> 1536, 4096 tokens

- kernels in FP8 arm only: 9; FP8-named GEMM kernels: FP8 arm 3, BF16 control 0
- forward rel L2 FP8 vs BF16: 3.760e-02, cosine 0.999293

| arm | kernel | calls | device µs | class |
|---|---|---|---|---|
| bf16_te | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{la` | 3 | 31.36 | other |
| bf16_te | `nvjet_tst_128x160_64x5_4x2_h_bz_coopA_NTT` | 1 | 28.83 | other_gemm |
| bf16_te | `nvjet_tst_192x128_64x5_1x2_h_bz_coopB_bias_TNT` | 1 | 28.38 | other_gemm |
| bf16_te | `nvjet_tst_192x128_64x5_1x2_h_bz_coopB_NNT` | 1 | 27.84 | other_gemm |
| bf16_te | `void at::native::vectorized_elementwise_kernel<8, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase` | 4 | 27.55 | other |
| bf16_te | `void cublasLt::epilogue::impl::globalKernelBgradAB_v2<8, 128, __nv_bfloat16, __nv_bfloat16, float>(int, int, i` | 1 | 6.34 | other |
| fp8_te | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{la` | 3 | 30.56 | other |
| fp8_te | `cast_transpose_optimized_kernel(float const*, float const*, __nv_fp8_e4m3*, __nv_fp8_e4m3*, float const*, floa` **(FP8 arm only)** | 2 | 23.94 | fp8_cast_or_scaling |
| fp8_te | `nvjet_qrtst_128x128_128x6_2x1_v_bz_coopA_algo2_TNT` **(FP8 arm only)** | 1 | 18.91 | fp8_gemm |
| fp8_te | `nvjet_qqtst_192x128_128x5_1x2_h_bz_coopB_bias_TNT` **(FP8 arm only)** | 1 | 18.37 | fp8_gemm |
| fp8_te | `nvjet_qrtst_144x128_128x6_1x2_h_bz_coopA_algo2_TNN` **(FP8 arm only)** | 1 | 16.51 | fp8_gemm |
| fp8_te | `void at::native::vectorized_elementwise_kernel<8, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase` | 2 | 13.95 | other |
| fp8_te | `cast_transpose_fusion_kernel_optimized(transformer_engine::CTDBiasDActParam<__nv_bfloat16, __nv_bfloat16, __nv` **(FP8 arm only)** | 1 | 6.66 | fp8_cast_or_scaling |
| fp8_te | `transformer_engine::delayed_scaling_recipe::(anonymous namespace)::amax_and_scale_update_impl::kernel_bulk(flo` **(FP8 arm only)** | 2 | 6.59 | fp8_cast_or_scaling |

## qwen_k_v_proj: 1536 -> 256, 4096 tokens

- kernels in FP8 arm only: 9; FP8-named GEMM kernels: FP8 arm 3, BF16 control 0
- forward rel L2 FP8 vs BF16: 3.760e-02, cosine 0.999293

| arm | kernel | calls | device µs | class |
|---|---|---|---|---|
| bf16_te | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{la` | 3 | 21.18 | other |
| bf16_te | `void at::native::vectorized_elementwise_kernel<8, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase` | 4 | 14.34 | other |
| bf16_te | `nvjet_tst_64x64_64x13_2x4_v_bz_NTT` | 1 | 10.08 | other_gemm |
| bf16_te | `nvjet_tst_192x128_64x5_1x2_h_bz_coopB_NNT` | 1 | 8.8 | other_gemm |
| bf16_te | `nvjet_tst_64x128_64x8_2x1_v_bz_bias_TNN` | 1 | 7.97 | other_gemm |
| bf16_te | `void cublasLt::epilogue::impl::globalKernelBgradAB_v2<8, 128, __nv_bfloat16, __nv_bfloat16, float>(int, int, i` | 1 | 4.19 | other |
| fp8_te | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{la` | 3 | 20.93 | other |
| fp8_te | `cast_transpose_optimized_kernel(float const*, float const*, __nv_fp8_e4m3*, __nv_fp8_e4m3*, float const*, floa` **(FP8 arm only)** | 2 | 16.74 | fp8_cast_or_scaling |
| fp8_te | `nvjet_qrtst_128x128_128x6_2x1_v_bz_coopA_algo2_TNT` **(FP8 arm only)** | 1 | 8.83 | fp8_gemm |
| fp8_te | `nvjet_qrtst_48x64_128x15_2x1_v_bz_algo2_TNN` **(FP8 arm only)** | 1 | 7.52 | fp8_gemm |
| fp8_te | `nvjet_qqtst_64x128_128x8_2x1_v_bz_bias_TNN` **(FP8 arm only)** | 1 | 6.56 | fp8_gemm |
| fp8_te | `transformer_engine::delayed_scaling_recipe::(anonymous namespace)::amax_and_scale_update_impl::kernel_bulk(flo` **(FP8 arm only)** | 2 | 6.11 | fp8_cast_or_scaling |
| fp8_te | `void at::native::vectorized_elementwise_kernel<8, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase` | 2 | 4.86 | other |
| fp8_te | `void transformer_engine::detail::reduce_dbias_kernel<4, float, __nv_bfloat16>(__nv_bfloat16*, float const*, in` **(FP8 arm only)** | 1 | 4.22 | other |

## qwen_gate_up_proj: 1536 -> 8960, 4096 tokens

- kernels in FP8 arm only: 8; FP8-named GEMM kernels: FP8 arm 3, BF16 control 0
- forward rel L2 FP8 vs BF16: 3.763e-02, cosine 0.999292

| arm | kernel | calls | device µs | class |
|---|---|---|---|---|
| bf16_te | `nvjet_tst_192x192_64x4_2x1_v_bz_coopB_bias_TNN` | 1 | 176.34 | other_gemm |
| bf16_te | `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN` | 1 | 160.82 | other_gemm |
| bf16_te | `nvjet_tst_192x128_64x5_1x2_h_bz_coopB_NNT` | 1 | 159.13 | other_gemm |
| bf16_te | `void at::native::vectorized_elementwise_kernel<8, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase` | 4 | 93.56 | other |
| bf16_te | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{la` | 3 | 65.76 | other |
| bf16_te | `void cublasLt::epilogue::impl::globalKernelBgradAB_v2<8, 128, __nv_bfloat16, __nv_bfloat16, float>(int, int, i` | 1 | 38.33 | other |
| bf16_te | `Memset (Device)` | 2 | 1.47 | other |
| fp8_te | `nvjet_qqtst_192x192_128x4_2x1_v_bz_coopB_bias_TNN` **(FP8 arm only)** | 1 | 101.69 | fp8_gemm |
| fp8_te | `nvjet_qrtst_144x128_128x6_1x2_h_bz_coopA_algo2_TNN` **(FP8 arm only)** | 1 | 95.61 | fp8_gemm |
| fp8_te | `nvjet_qrtst_128x128_128x6_2x1_v_bz_coopA_algo2_TNT` **(FP8 arm only)** | 1 | 84.0 | fp8_gemm |
| fp8_te | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{la` | 3 | 67.17 | other |
| fp8_te | `void at::native::vectorized_elementwise_kernel<8, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase` | 2 | 62.08 | other |
| fp8_te | `cast_transpose_optimized_kernel(float const*, float const*, __nv_fp8_e4m3*, __nv_fp8_e4m3*, float const*, floa` **(FP8 arm only)** | 2 | 46.08 | fp8_cast_or_scaling |
| fp8_te | `cast_transpose_fusion_kernel_optimized(transformer_engine::CTDBiasDActParam<__nv_bfloat16, __nv_bfloat16, __nv` **(FP8 arm only)** | 1 | 45.98 | fp8_cast_or_scaling |
| fp8_te | `transformer_engine::delayed_scaling_recipe::(anonymous namespace)::amax_and_scale_update_impl::kernel_bulk(flo` **(FP8 arm only)** | 2 | 7.01 | fp8_cast_or_scaling |

## qwen_down_proj: 8960 -> 1536, 4096 tokens

- kernels in FP8 arm only: 8; FP8-named GEMM kernels: FP8 arm 3, BF16 control 0
- forward rel L2 FP8 vs BF16: 3.775e-02, cosine 0.999287

| arm | kernel | calls | device µs | class |
|---|---|---|---|---|
| bf16_te | `nvjet_tst_320x128_64x3_1x2_h_bz_coopB_NNT` | 1 | 168.41 | other_gemm |
| bf16_te | `nvjet_tst_192x192_64x3_1x2_h_bz_coopB_NTN` | 1 | 161.37 | other_gemm |
| bf16_te | `nvjet_tst_192x128_64x5_1x2_h_bz_coopB_bias_TNT` | 1 | 161.34 | other_gemm |
| bf16_te | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{la` | 3 | 156.95 | other |
| bf16_te | `void at::native::vectorized_elementwise_kernel<8, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase` | 4 | 95.39 | other |
| bf16_te | `void cublasLt::epilogue::impl::globalKernelBgradAB_v2<8, 128, __nv_bfloat16, __nv_bfloat16, float>(int, int, i` | 1 | 7.07 | other |
| bf16_te | `Memset (Device)` | 2 | 1.47 | other |
| fp8_te | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{la` | 3 | 157.53 | other |
| fp8_te | `cast_transpose_optimized_kernel(float const*, float const*, __nv_fp8_e4m3*, __nv_fp8_e4m3*, float const*, floa` **(FP8 arm only)** | 2 | 97.56 | fp8_cast_or_scaling |
| fp8_te | `nvjet_qrtst_128x160_128x5_2x1_v_bz_coopA_algo2_TNT` **(FP8 arm only)** | 1 | 96.28 | fp8_gemm |
| fp8_te | `nvjet_qrtst_144x128_128x6_1x2_h_bz_coopA_algo2_TNN` **(FP8 arm only)** | 1 | 95.74 | fp8_gemm |
| fp8_te | `nvjet_qqtst_192x128_128x5_1x2_h_bz_coopB_bias_TNT` **(FP8 arm only)** | 1 | 80.44 | fp8_gemm |
| fp8_te | `void at::native::vectorized_elementwise_kernel<8, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase` | 2 | 11.39 | other |
| fp8_te | `transformer_engine::delayed_scaling_recipe::(anonymous namespace)::amax_and_scale_update_impl::kernel_bulk(flo` **(FP8 arm only)** | 2 | 7.46 | fp8_cast_or_scaling |
| fp8_te | `cast_transpose_fusion_kernel_optimized(transformer_engine::CTDBiasDActParam<__nv_bfloat16, __nv_bfloat16, __nv` **(FP8 arm only)** | 1 | 6.62 | fp8_cast_or_scaling |

