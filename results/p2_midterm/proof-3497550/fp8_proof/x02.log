# X02 first pass: eager BF16 (A) vs TE BF16 (B) vs TE FP8 (C), Qwen2.5-1.5B + LoRA r=16

One fixed validation microbatch (8 x 512 tokens, val blocks 0-7). Layers (0, 14, 27). C after 4 FP8 warmup forwards on val blocks 8-39; C_cold is the first FP8 pass ever (empty amax history).
Frozen weights identical across A and B/C models: True (0 mismatched tensors). TE conversion: 196 of 197 nn.Linear converted.
Mean loss: A 2.138570, B 2.138570, C 2.164734, C_cold 2.134740.

Single checkpoint (pretrained + fixed LoRA), single batch, single seed: an instantaneous perturbation measurement (§12.7), not evidence about training outcomes.

| comparison | scope | L_inf | median abs | rel L2 | cosine |
|---|---|---|---|---|---|
| B_vs_A (backend) | isolated layers.0.self_attn.q_proj | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.000e+00 |
| B_vs_A (backend) | isolated layers.0.mlp.down_proj | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.000e+00 |
| B_vs_A (backend) | isolated layers.14.self_attn.q_proj | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.000e+00 |
| B_vs_A (backend) | isolated layers.14.mlp.down_proj | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.000e+00 |
| B_vs_A (backend) | isolated layers.27.self_attn.q_proj | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.000e+00 |
| B_vs_A (backend) | isolated layers.27.mlp.down_proj | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.000e+00 |
| B_vs_A (backend) | in-network output layers.0.self_attn.q_proj | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.000e+00 |
| B_vs_A (backend) | in-network output layers.0.mlp.down_proj | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.000e+00 |
| B_vs_A (backend) | in-network output layers.14.self_attn.q_proj | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.000e+00 |
| B_vs_A (backend) | in-network output layers.14.mlp.down_proj | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.000e+00 |
| B_vs_A (backend) | in-network output layers.27.self_attn.q_proj | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.000e+00 |
| B_vs_A (backend) | in-network output layers.27.mlp.down_proj | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.000e+00 |
| B_vs_A (backend) | logits | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.000e+00 |
| B_vs_A (backend) | per-token NLL | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.000e+00 |
| B_vs_A (backend) | mean loss delta | 0.000e+00 |  |  |  |
| B_vs_A (backend) | grad layers.14.self_attn.q_proj.lora_A.weight | 4.768e-07 | 7.451e-09 | 7.471e-03 | 1.000e+00 |
| B_vs_A (backend) | grad layers.14.self_attn.q_proj.lora_B.weight | 4.292e-06 | 1.192e-07 | 6.637e-03 | 1.000e+00 |
| B_vs_A (backend) | grad all LoRA params (concatenated) | 4.447e-05 | 1.490e-08 | 7.353e-03 | 1.000e+00 |
| C_vs_B (FP8 format) | isolated layers.0.self_attn.q_proj | 3.000e+00 | 1.562e-02 | 1.458e-02 | 9.999e-01 |
| C_vs_B (FP8 format) | isolated layers.0.mlp.down_proj | 3.750e-01 | 5.859e-03 | 3.185e-02 | 9.995e-01 |
| C_vs_B (FP8 format) | isolated layers.14.self_attn.q_proj | 5.156e-01 | 1.562e-02 | 1.753e-02 | 9.998e-01 |
| C_vs_B (FP8 format) | isolated layers.14.mlp.down_proj | 3.125e-01 | 7.812e-03 | 3.461e-02 | 9.994e-01 |
| C_vs_B (FP8 format) | isolated layers.27.self_attn.q_proj | 3.750e-01 | 1.562e-02 | 2.404e-02 | 9.997e-01 |
| C_vs_B (FP8 format) | isolated layers.27.mlp.down_proj | 8.000e+00 | 4.883e-02 | 2.287e-02 | 9.998e-01 |
| C_vs_B (FP8 format) | in-network output layers.0.self_attn.q_proj | 3.000e+00 | 1.562e-02 | 1.458e-02 | 9.999e-01 |
| C_vs_B (FP8 format) | in-network output layers.0.mlp.down_proj | 5.625e-01 | 1.367e-02 | 6.871e-02 | 9.977e-01 |
| C_vs_B (FP8 format) | in-network output layers.14.self_attn.q_proj | 2.400e+00 | 4.688e-02 | 4.656e-02 | 9.989e-01 |
| C_vs_B (FP8 format) | in-network output layers.14.mlp.down_proj | 2.602e+00 | 3.516e-02 | 1.492e-01 | 9.889e-01 |
| C_vs_B (FP8 format) | in-network output layers.27.self_attn.q_proj | 6.854e+00 | 3.906e-02 | 7.069e-02 | 9.975e-01 |
| C_vs_B (FP8 format) | in-network output layers.27.mlp.down_proj | 4.090e+02 | 1.406e-01 | 1.628e-01 | 9.867e-01 |
| C_vs_B (FP8 format) | logits | 1.719e+01 | 1.719e-01 | 9.084e-02 | 9.959e-01 |
| C_vs_B (FP8 format) | per-token NLL | 1.163e+01 | 5.284e-02 | 1.188e-01 | 9.932e-01 |
| C_vs_B (FP8 format) | mean loss delta | 2.616e-02 |  |  |  |
| C_vs_B (FP8 format) | grad layers.14.self_attn.q_proj.lora_A.weight | 1.627e-05 | 3.427e-07 | 2.539e-01 | 9.680e-01 |
| C_vs_B (FP8 format) | grad layers.14.self_attn.q_proj.lora_B.weight | 1.526e-04 | 5.007e-06 | 2.528e-01 | 9.685e-01 |
| C_vs_B (FP8 format) | grad all LoRA params (concatenated) | 1.718e-03 | 7.451e-07 | 2.318e-01 | 9.728e-01 |
| C_cold_vs_B (FP8, no amax history) | isolated layers.0.self_attn.q_proj | 3.000e+00 | 1.562e-02 | 1.458e-02 | 9.999e-01 |
| C_cold_vs_B (FP8, no amax history) | isolated layers.0.mlp.down_proj | 3.750e-01 | 5.859e-03 | 3.185e-02 | 9.995e-01 |
| C_cold_vs_B (FP8, no amax history) | isolated layers.14.self_attn.q_proj | 5.156e-01 | 1.562e-02 | 1.753e-02 | 9.998e-01 |
| C_cold_vs_B (FP8, no amax history) | isolated layers.14.mlp.down_proj | 3.125e-01 | 7.812e-03 | 3.461e-02 | 9.994e-01 |
| C_cold_vs_B (FP8, no amax history) | isolated layers.27.self_attn.q_proj | 3.750e-01 | 1.562e-02 | 2.404e-02 | 9.997e-01 |
| C_cold_vs_B (FP8, no amax history) | isolated layers.27.mlp.down_proj | 8.000e+00 | 4.883e-02 | 2.287e-02 | 9.998e-01 |
| C_cold_vs_B (FP8, no amax history) | in-network output layers.0.self_attn.q_proj | 5.500e+00 | 1.758e-02 | 1.723e-02 | 9.999e-01 |
| C_cold_vs_B (FP8, no amax history) | in-network output layers.0.mlp.down_proj | 5.000e-01 | 1.562e-02 | 7.813e-02 | 9.970e-01 |
| C_cold_vs_B (FP8, no amax history) | in-network output layers.14.self_attn.q_proj | 2.188e+00 | 5.469e-02 | 5.489e-02 | 9.985e-01 |
| C_cold_vs_B (FP8, no amax history) | in-network output layers.14.mlp.down_proj | 2.688e+00 | 4.126e-02 | 1.748e-01 | 9.847e-01 |
| C_cold_vs_B (FP8, no amax history) | in-network output layers.27.self_attn.q_proj | 7.191e+00 | 4.785e-02 | 8.037e-02 | 9.968e-01 |
| C_cold_vs_B (FP8, no amax history) | in-network output layers.27.mlp.down_proj | 2.380e+02 | 1.719e-01 | 1.231e-01 | 9.926e-01 |
| C_cold_vs_B (FP8, no amax history) | logits | 1.219e+01 | 2.031e-01 | 1.028e-01 | 9.947e-01 |
| C_cold_vs_B (FP8, no amax history) | per-token NLL | 4.168e+00 | 6.163e-02 | 7.374e-02 | 9.973e-01 |
| C_cold_vs_B (FP8, no amax history) | mean loss delta | -3.830e-03 |  |  |  |
| C_cold_vs_B (FP8, no amax history) | grad layers.14.self_attn.q_proj.lora_A.weight | 1.400e-04 | 1.227e-06 | 1.035e+00 | 1.504e-01 |
| C_cold_vs_B (FP8, no amax history) | grad layers.14.self_attn.q_proj.lora_B.weight | 6.862e-04 | 1.800e-05 | 1.035e+00 | 1.555e-01 |
| C_cold_vs_B (FP8, no amax history) | grad all LoRA params (concatenated) | 1.175e-02 | 3.057e-06 | 1.101e+00 | 2.679e-01 |
