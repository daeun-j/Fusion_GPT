# Fused LayerNorm+Linear CUDA Kernel

Fuses `Linear(LayerNorm(x))` and `Linear(RMSNorm(x))` into a single operation for transformer inference. The normalization scaling is baked into pre-computed weight matrices and the matmul runs alongside a lightweight CUDA denominator kernel. Supports both LayerNorm (OPT) and RMSNorm (Llama) model families, with FP32, FP16, and BF16 precision.

Best single-op result on H100 is 2.17x (V1 kernel, OPT-6.7b attention, batch=32). End-to-end, V1/V3 reach 1.24x on OPT-6.7b at batch=16 and stay positive starting from batch=1. The V1/V3 fused-normalize variants drop stream overhead and turn V0's 0.37-0.42x slowdowns into 1.03-1.34x speedups. Transformer Engine is included as a production baseline (up to 5.72x at large dimensions).

## When does this help in practice

The fusion is worth using on OPT-6.7b scale models (h=4096 and up). V1/V3 stay positive end-to-end starting from batch=1 (1.03x) and reach 1.24x at batch=16. Single-op, they hit 2.17x on attention projections at batch=32. On smaller OPT-1.3b scale models (h=2048) the benefit only shows up at high batch: V1/V3 reach 1.17x E2E at batch=32, and are near-neutral at batch=1 with no penalty (unlike V0's 0.48x). Prefill-heavy workloads are a particularly good fit because the large `[batch * seq_len, hidden]` shapes amortize the overhead.

It's the wrong choice for large FFN matmuls. At out_dim=16384, batch=512, Transformer Engine reaches 5.72x where our V1 gets 0.95x. When the matmul fully dominates, TE's fully fused kernel wins decisively. More generally, if Transformer Engine is available and the model has large FFN dimensions, TE's `LayerNormLinear` is faster at those dimensions. Our approach is competitive on moderate-dim attention projections, where weight pre-computation eliminates the normalization step entirely.

Deployment cost is zero at runtime. The weight transform (`W_new`, `b_new`) is computed once at model load, and the fused forward path replaces the original module without any extra memory allocation.

## Mathematical background

Standard transformer pattern:
```
output = Linear(LayerNorm(x)) = (x - μ) · γ / σ · W^T + β · W^T + b
```

Define the denominator `v(x) = ||x - μ(x)||₂`, so `σ(x) = √(v²/h + ε)`.

Pre-computed weights (one-time, at patch time):
```python
M     = (W * γ).T              # element-wise multiply, then transpose: diag(γ) @ W.T
W_new = (M - M.mean(dim=0)).T  # center columns = apply (I - 11^T/h), where 1 is the all-ones vector
b_new = β @ W.T + b            # matrix multiply β with W.T, then add bias
```

Assumptions: LayerNorm has both weight (γ) and bias (β) parameters, and `normalized_shape` is 1-dimensional (normalization over the last dimension only).

Fused forward (per inference call):
```python
raw = x @ W_new.T          # cuBLAS matmul (default stream)
v   = denominator_cuda(x)  # ||x - mean(x)||₂ (separate stream, concurrent)
std = √(v²/h + ε)          # exact LN std
out = raw / std + b_new
```

The matmul and denominator are independent, so running them on separate CUDA streams hides the denominator latency behind the matmul.

## Related work

The algebraic technique of absorbing LayerNorm affine parameters into subsequent linear projections has been independently explored in several concurrent works:

1. Salmani & Soloveychik (2025), [arXiv:2502.17728](https://arxiv.org/abs/2502.17728). Applies the same algebraic decomposition (centering + scaling separation) to fuse LayerNorm into linear layers, reporting roughly 20% latency reduction on the d-Matrix Corsair accelerator. Our work uses the same mathematical formulation but targets NVIDIA H100 GPUs with a custom CUDA denominator kernel.

2. FlashNorm, [arXiv:2407.09577](https://arxiv.org/abs/2407.09577). Proposes RMSNorm weight absorption for Llama/Mistral-family models, eliminating the normalization step entirely by folding RMSNorm scaling into downstream projections. Complementary to our approach: FlashNorm targets RMSNorm (no mean centering), while we handle full LayerNorm (mean + variance).

3. CCWT (Column-Centered Weight Transformation, ICLR 2025), [OpenReview](https://openreview.net/forum?id=bVdcAZAW2h). Introduces column-centered weight transformation as a general technique for fusing normalization layers, reporting 10-20% inference gains. The column-centering step `(I - 11^T/h) @ M` in our weight transform is mathematically equivalent to the CCWT formulation.

## Hardware & Software

| Component | Version |
|-----------|---------|
| GPU | NVIDIA H100 80GB HBM3 |
| CUDA Driver | 13.1 |
| CUDA Toolkit | 12.8 (for JIT compilation) |
| PyTorch | 2.10.0+cu128 |
| Python | 3.12.3 |
| transformers | 5.1.0 |
| Transformer Engine | 2.11.0 (optional, for baseline comparison) |

## Project Structure

```
fused_ln_linear/
├── README.md                         # This file
├── REPORT.md                         # Detailed development report
├── docs/
│   └── task_description.md           # Original task specification
├── csrc/
│   ├── denominator_kernel.cu         # CUDA kernels (LN + RMSNorm, all variants, FP32/FP16/BF16)
│   └── denominator.cpp               # pybind11 bindings
├── src/
│   ├── __init__.py
│   ├── load_cuda.py                  # JIT compilation loader
│   ├── weight_transform.py           # Pre-compute W_new, b_new (LayerNorm + RMSNorm)
│   ├── fused_forward.py              # Fused forward classes (LN V0/V1/V3 + RMSNorm V1/V3)
│   ├── patch_model.py                # Monkey-patch OPT models
│   ├── patch_llama.py                # Monkey-patch Llama models (RMSNorm fusion)
│   ├── test_correctness.py           # Correctness tests (all variants + FP16/BF16 + Llama)
│   └── benchmark.py                  # Performance benchmarks (JSON output, TE baseline)
├── results/                          # JSON benchmark output (gitignored)
├── build_ext.py                      # Standalone JIT build script
├── setup.py                          # setuptools build (alternative)
└── scripts/
    ├── install_deps.sh               # Dependency installation
    └── profile_nsys.sh               # Nsight Systems profiling script
```

## Setup

```bash
cd path/to/fused_ln_linear
python3 -m venv venv
source venv/bin/activate
pip install torch transformers accelerate

# Optional: Transformer Engine for baseline comparison
pip install transformer-engine[pytorch]

# The CUDA extension builds automatically via JIT on first import.
# No separate build step needed.
export CUDA_HOME=/usr/local/cuda-12.8  # Required for JIT compilation

# Run correctness tests
python3 -m src.test_correctness

# Run benchmarks (results saved as JSON to results/)
python3 -m src.benchmark --single-op   # Single-operation benchmarks
python3 -m src.benchmark --e2e         # End-to-end OPT model benchmarks
python3 -m src.benchmark --all         # Everything

# Nsight Systems profiling (optional)
bash scripts/profile_nsys.sh single-op
bash scripts/profile_nsys.sh e2e
```

## Kernel variants

### V0: original (two-pass + CUDA streams)

The baseline implementation. The denominator kernel runs on a separate CUDA stream concurrent with the matmul.

- Kernel: two-pass reduction (mean, then squared deviations), 256 threads/block, float4 vectorized loads.
- Forward: stream concurrency with pre-allocated CUDA events.
- Overhead: roughly 0.06-0.09ms from event record/wait, stream context, and elementwise ops (sqrt, div, add).

### V1: fused denominator+normalize (no streams)

Drops all stream/event overhead by fusing the denominator, output normalization, and bias into a single kernel on the default stream.

- Kernel: two-pass reduction, then in-place normalize `raw_output[row][c] = raw_output[row][c] / std + b_new[c]`.
- Forward: just `F.linear(x, W_new)` followed by one kernel call, no streams and no events.
- Benefit: removes the ~0.06-0.09ms Python-side CUDA API overhead that dominated small configurations.

### V2: Welford single-pass denominator

Halves global memory reads by computing mean and variance in one pass with Welford's online algorithm.

- Kernel: single-pass Welford reduction with parallel merge (`n, mean, M2` triples), warp-shuffle plus shared memory.
- Forward: same stream-based approach as V0 but with a faster denominator.
- Benefit: less memory traffic for large hidden dims where L2 cache pressure matters.

### V3: combined (Welford + fused normalize + 512 threads)

Combines V1 and V2: single-pass Welford, fused normalization, wider thread blocks.

- Kernel: 512 threads/block (16 warps), Welford single-pass, fused in-place normalize + bias.
- Forward: like V1, just matmul + one kernel call.
- Benefit: combines the memory-traffic reduction with the overhead elimination, and gives better utilization for large `out_dim`.

### RMSNorm variants (V1, V3)

For models using RMSNorm (Llama, Mistral, etc.) instead of LayerNorm. RMSNorm is simpler: single-pass sum-of-squares, no mean subtraction.

- Weight transform: `W_new = W * gamma` (no column centering needed).
- RMSNorm V1: 256 threads, single-pass `sum(x^2)`, fused normalize + bias.
- RMSNorm V3: 512 threads, same algorithm with wider thread blocks.
- Precision: FP32, FP16, BF16 with FP32 internal accumulation (same as LN variants).

### Precision support

All V1/V3 kernels (both LayerNorm and RMSNorm) support three precision modes:

| Precision | Load Pattern | Internal Accumulation | Typical Error vs Reference |
|-----------|-------------|----------------------|---------------------------|
| FP32 | `float` (scalar) | FP32 | ~1e-6 (kernel) / ~6e-5 (E2E) |
| FP16 | `half2` (vectorized) | FP32 | ~1e-3 |
| BF16 | `bfloat162` (vectorized) | FP32 | ~1e-2 |

### Nsight Systems Profiling

NVTX annotations are built into all forward methods, enabled via environment variable:

```bash
FUSED_LN_NVTX=1 bash scripts/profile_nsys.sh single-op
```

This generates `.nsys-rep` files in `results/` showing kernel overlap, stream concurrency, and per-variant timing breakdowns.

## Correctness Results

All variants produce outputs matching the sequential reference within floating-point tolerance. Tested across FP32, FP16, and BF16 precisions. Both LayerNorm (OPT) and RMSNorm (Llama) models are validated.

### Denominator Kernel vs PyTorch Reference
```
fp32 (   1 x  768): max_diff=0.00e+00
fp32 (  32 x  768): max_diff=1.91e-06
fp32 ( 128 x 2048): max_diff=3.81e-06
fp32 ( 512 x 4096): max_diff=7.63e-06
```

### LayerNorm Fused LN+Linear Unit Tests (all variants)
```
h= 768, out=  768, batch= 32: V0=8.34e-07  V1=8.34e-07  V3=8.34e-07
h= 768, out= 3072, batch=128: V0=3.34e-06  V1=3.34e-06  V3=3.34e-06
h=2048, out= 2048, batch= 64: V0=2.62e-06  V1=2.62e-06  V3=2.62e-06
h=4096, out=16384, batch= 16: V0=1.72e-05  V1=1.72e-05  V3=1.72e-05
```

### FP16/BF16 Correctness (V1 and V3)
```
FP16 h= 768, out=  768, batch= 32: V1=8.79e-04  V3=8.79e-04
FP16 h=2048, out= 2048, batch= 64: V1=1.95e-03  V3=1.95e-03
FP16 h=4096, out= 4096, batch= 16: V1=3.91e-03  V3=3.91e-03
BF16 h= 768, out=  768, batch= 32: V1=5.47e-02  V3=5.47e-02
BF16 h=2048, out= 2048, batch= 64: V1=1.56e-01  V3=1.56e-01
BF16 h=4096, out= 4096, batch= 16: V1=2.19e-01  V3=2.19e-01
```

### RMSNorm Fused Unit Tests (V1 and V3)
```
h= 768, out=  768, batch= 32: V1=4.77e-07  V3=4.77e-07
h= 768, out= 3072, batch=128: V1=1.91e-06  V3=1.91e-06
h=2048, out= 2048, batch= 64: V1=1.91e-06  V3=1.91e-06
h=4096, out= 4096, batch= 16: V1=3.81e-06  V3=3.81e-06
```

### OPT-125m Integration (LayerNorm, full model logits)
```
"The quick brown fox...": max_diff=6.48e-05
"In a galaxy far...":     max_diff=6.29e-05
"Machine learning...":    max_diff=6.48e-05
```

### TinyLlama-1.1B Integration (RMSNorm, full model logits)
```
"The quick brown fox...": max_diff=2.86e-05
"In a galaxy far...":     max_diff=2.67e-05
"Machine learning...":    max_diff=2.86e-05
```

## Benchmark Results

### Single-Operation: All Variants (1000 iterations, H100)

Raw data: [`results/single_op_20260209T050038Z.json`](results/single_op_20260209T050038Z.json) (120 entries)

Methodology: each configuration runs 100 warmup iterations followed by 1000 individually-timed iterations using pre-allocated `torch.cuda.Event` pairs. Reported values are mean±stddev across all 1000 iterations. No GPU clock-locking was applied, so ambient thermal state may introduce variance across runs.

```
Config                 h    out  batch |   Original       V0-Stream        V1-Fused       V2-Welford      V3-Combined     TE-Baseline
----------------------------------------------------------------------------------------------------------------------------------------
OPT-125m attn        768    768      1 |  0.0340±0.0036  0.0818±0.0074(0.42x)  0.0253±0.0022(1.34x)  0.0798±0.0088(0.43x)  0.0253±0.0019(1.34x)  0.0725±0.0057(0.47x)
OPT-125m attn        768    768     32 |  0.0325±0.0030  0.0880±0.0103(0.37x)  0.0291±0.0016(1.12x)  0.0870±0.0067(0.37x)  0.0298±0.0022(1.09x)  0.0742±0.0063(0.44x)
OPT-125m attn        768    768    128 |  0.0343±0.0013  0.0909±0.0061(0.38x)  0.0294±0.0015(1.17x)  0.0837±0.0060(0.41x)  0.0295±0.0018(1.16x)  0.0781±0.0046(0.44x)
OPT-125m attn        768    768    512 |  0.0373±0.0019  0.0928±0.0087(0.40x)  0.0294±0.0017(1.27x)  0.0899±0.0081(0.42x)  0.0308±0.0028(1.21x)  0.0789±0.0062(0.47x)
OPT-125m FFN         768   3072      1 |  0.0339±0.0037  0.0816±0.0076(0.42x)  0.0253±0.0022(1.34x)  0.0818±0.0077(0.41x)  0.0254±0.0019(1.33x)  0.0734±0.0041(0.46x)
OPT-125m FFN         768   3072     32 |  0.0350±0.0020  0.0864±0.0063(0.41x)  0.0297±0.0024(1.18x)  0.0831±0.0064(0.42x)  0.0299±0.0028(1.17x)  0.0746±0.0054(0.47x)
OPT-125m FFN         768   3072    128 |  0.0351±0.0037  0.0907±0.0062(0.39x)  0.0301±0.0019(1.17x)  0.0859±0.0081(0.41x)  0.0300±0.0017(1.17x)  0.0768±0.0044(0.46x)
OPT-125m FFN         768   3072    512 |  0.0724±0.0017  0.0934±0.0076(0.78x)  0.0733±0.0011(0.99x)  0.0915±0.0031(0.79x)  0.0745±0.0013(0.97x)  0.0793±0.0065(0.91x)
OPT-1.3b attn       2048   2048      1 |  0.0344±0.0041  0.0823±0.0079(0.42x)  0.0261±0.0036(1.32x)  0.0774±0.0071(0.44x)  0.0266±0.0046(1.29x)  0.0728±0.0067(0.47x)
OPT-1.3b attn       2048   2048     32 |  0.0454±0.0039  0.0900±0.0087(0.50x)  0.0342±0.0028(1.33x)  0.0830±0.0079(0.55x)  0.0342±0.0023(1.33x)  0.0753±0.0070(0.60x)
OPT-1.3b attn       2048   2048    128 |  0.0597±0.0018  0.0904±0.0075(0.66x)  0.0432±0.0013(1.38x)  0.0868±0.0057(0.69x)  0.0431±0.0021(1.39x)  0.0784±0.0063(0.76x)
OPT-1.3b attn       2048   2048    512 |  0.1260±0.0022  0.1243±0.0019(1.01x)  0.1070±0.0021(1.18x)  0.1240±0.0019(1.02x)  0.1081±0.0019(1.16x)  0.0812±0.0048(1.55x)
OPT-1.3b FFN        2048   8192      1 |  0.0385±0.0043  0.0815±0.0076(0.47x)  0.0337±0.0012(1.14x)  0.0761±0.0077(0.51x)  0.0328±0.0011(1.18x)  0.0772±0.0046(0.50x)
OPT-1.3b FFN        2048   8192     32 |  0.0785±0.0016  0.0926±0.0091(0.85x)  0.0539±0.0021(1.46x)  0.0864±0.0125(0.91x)  0.0530±0.0015(1.48x)  0.0843±0.0057(0.93x)
OPT-1.3b FFN        2048   8192    128 |  0.1279±0.0018  0.1314±0.0020(0.97x)  0.1145±0.0017(1.12x)  0.1311±0.0016(0.98x)  0.1131±0.0018(1.13x)  0.0838±0.0082(1.53x)
OPT-1.3b FFN        2048   8192    512 |  0.3522±0.0060  0.3938±0.0045(0.89x)  0.3644±0.0018(0.97x)  0.3936±0.0027(0.89x)  0.3649±0.0027(0.97x)  0.0884±0.0136(3.99x)
OPT-6.7b attn       4096   4096      1 |  0.0430±0.0016  0.0818±0.0076(0.53x)  0.0342±0.0012(1.26x)  0.0768±0.0104(0.56x)  0.0345±0.0022(1.25x)  0.0771±0.0063(0.56x)
OPT-6.7b attn       4096   4096     32 |  0.1174±0.0018  0.0925±0.0069(1.27x)  0.0542±0.0021(2.17x)  0.0897±0.0092(1.31x)  0.0544±0.0014(2.16x)  0.0843±0.0056(1.39x)
OPT-6.7b attn       4096   4096    128 |  0.1817±0.0024  0.1260±0.0017(1.44x)  0.1100±0.0017(1.65x)  0.1261±0.0017(1.44x)  0.1093±0.0020(1.66x)  0.0866±0.0066(2.10x)
OPT-6.7b attn       4096   4096    512 |  0.3757±0.0028  0.3781±0.0030(0.99x)  0.3558±0.0014(1.06x)  0.3783±0.0028(0.99x)  0.3592±0.0027(1.05x)  0.0948±0.0121(3.96x)
OPT-6.7b FFN        4096  16384      1 |  0.1059±0.0017  0.1103±0.0019(0.96x)  0.1000±0.0014(1.06x)  0.1103±0.0016(0.96x)  0.0986±0.0013(1.07x)  0.0971±0.0029(1.09x)
OPT-6.7b FFN        4096  16384     32 |  0.1969±0.0019  0.2203±0.0033(0.89x)  0.2045±0.0021(0.96x)  0.2202±0.0023(0.89x)  0.2044±0.0024(0.96x)  0.1034±0.0029(1.90x)
OPT-6.7b FFN        4096  16384    128 |  0.3710±0.0025  0.3791±0.0038(0.98x)  0.3767±0.0153(0.98x)  0.3812±0.0037(0.97x)  0.3761±0.0078(0.99x)  0.1058±0.0043(3.51x)
OPT-6.7b FFN        4096  16384    512 |  1.3710±0.0204  1.4874±0.0186(0.92x)  1.4356±0.0115(0.95x)  1.4887±0.0088(0.92x)  1.4374±0.0112(0.95x)  0.2397±0.0055(5.72x)
```

A note on Transformer Engine (TE) results: TE's `LayerNormLinear` uses a fully fused CUDA kernel with optimized memory access patterns, reaching up to 5.72x at large dimensions. Our V1 kernel hits 2.17x at its best (OPT-6.7b attn, batch=32). That's competitive for a research implementation, but doesn't match TE's production-grade tuning.

### End-to-End: Multi-Batch Token Generation (128 new tokens, V0/V1/V3)

Raw data: [`results/e2e_20260209T052124Z.json`](results/e2e_20260209T052124Z.json) (36 entries)

#### OPT-1.3b (all variants, 5 runs per config)
```
Batch  Tokens |   Original    tok/s |       V0    tok/s Speedup |       V1    tok/s Speedup |       V3    tok/s Speedup
    1     128 |   812.5ms    157.5 | 1708.3ms    74.9  0.476x |  808.4ms   158.3  1.005x |  814.9ms   157.1  0.997x
    2     256 |   799.3ms    320.3 | 1761.4ms   145.3  0.454x |  830.3ms   308.3  0.963x |  853.7ms   299.9  0.936x
    4     512 |   786.6ms    650.9 | 1789.7ms   286.1  0.440x |  834.7ms   613.4  0.942x |  845.8ms   605.3  0.930x
    8    1024 |  1026.0ms    998.1 | 1775.1ms   576.9  0.578x | 1051.3ms   974.1  0.976x | 1047.7ms   977.4  0.979x
   16    2048 |  1369.4ms   1495.6 | 1970.0ms  1039.6  0.695x | 1415.0ms  1447.3  0.968x | 1407.4ms  1455.2  0.973x
   32    4096 |  1724.7ms   2375.0 | 2052.8ms  1995.3  0.840x | 1479.5ms  2768.5  1.166x | 1478.8ms  2769.8  1.166x
```

Key finding: V1/V3 reach 1.17x end-to-end at batch=32 on OPT-1.3b, a model where V0 never gets to break-even. At batch=1 V1/V3 are near-neutral (1.0x) versus V0's 0.48x penalty.

#### OPT-6.7b (all variants, 5 runs per config)
```
Batch  Tokens |   Original    tok/s |       V0    tok/s Speedup |       V1    tok/s Speedup |       V3    tok/s Speedup
    1     128 |  1606.7ms     79.7 | 2211.2ms    57.9  0.727x | 1555.8ms    82.3  1.033x | 1553.8ms    82.4  1.034x
    2     256 |  1804.4ms    141.9 | 2282.7ms   112.1  0.790x | 1779.9ms   143.8  1.014x | 1777.3ms   144.0  1.015x
    4     512 |  1947.5ms    262.9 | 2359.7ms   217.0  0.825x | 1918.5ms   266.9  1.015x | 1911.8ms   267.8  1.019x
    8    1024 |  2763.7ms    370.5 | 3162.4ms   323.8  0.874x | 2724.7ms   375.8  1.014x | 2728.3ms   375.3  1.013x
   16    2048 |  4136.6ms    495.1 | 3846.6ms   532.4  1.075x | 3333.1ms   614.4  1.241x | 3323.2ms   616.3  1.245x
   32    4096 |  4826.7ms    848.6 | 4619.1ms   886.8  1.045x | 4168.8ms   982.5  1.158x | 4163.8ms   983.7  1.159x
```

Key finding: OPT-6.7b V1/V3 are positive starting from batch=1 (1.03x) and reach 1.24x at batch=16. V0 doesn't get into positive territory until batch=16 (1.07x). The V1/V3 advantage is consistent because they drop the stream overhead that dominated V0 at small batch sizes.

## What we learned

Eliminating stream overhead is by far the biggest win. V1/V3 (fused normalize, no streams) turned V0's 0.37-0.42x slowdowns at OPT-125m into 1.09-1.34x speedups. The ~0.05-0.08ms of Python-side CUDA API overhead (event record/wait, stream context) was the dominant bottleneck, not the kernel itself. V1 is faster than V0 in every tested configuration without exception, and V2 (Welford + streams) matches V0 in speed, which confirms the overhead comes from stream management rather than the kernel algorithm.

V1 and V3 perform almost identically. The Welford single-pass optimization in V3 doesn't help measurably over the two-pass version (V1) at these dimensions because the reduction is memory-bandwidth-bound and L2 cache handles the second pass cheaply. We prefer V1 for simplicity.

Best single-op result: 2.17x (V1, OPT-6.7b attn 4096x4096, batch=32).

Transformer Engine is substantially faster at large dimensions. TE's `LayerNormLinear` reaches 3.96-5.72x at large batch/dim configs where our V1 sits at 0.95-1.06x. At V1's sweet spot (4096x4096, batch=32) TE gets 1.39x against our 2.17x; we beat TE in that corner specifically because the weight pre-computation eliminates the normalization step entirely.

Large FFN layers (out_dim=16384) remain hard. At batch=512 the matmul dominates (>1.3ms) and the LN savings (~0.01ms) are proportionally tiny, so the speedup sits at 0.95x. This is inherent: the optimization targets the LN overhead, which is negligible once the matmul is 100x larger.

A precision note: computing `std = sqrt(v^2/h + eps)` rather than approximating `std ~ v/sqrt(h)` drops the max error from ~2e-2 to ~6e-5.

The end-to-end results match the single-op findings. OPT-6.7b V1/V3: positive from batch=1 (1.03x), peaking at 1.24x (batch=16). OPT-1.3b V1/V3: neutral at batch=1 (1.0x), reaching 1.17x at batch=32. V0 is consistently worse because of stream overhead.

## Supported models

### OPT (LayerNorm)
- Tested: OPT-125m, OPT-1.3b, OPT-6.7b
- Fused pairs: `self_attn_layer_norm → q/k/v_proj`, `final_layer_norm → fc1`
- Patching: `from src.patch_model import patch_opt_model`

### Llama / TinyLlama (RMSNorm)
- Tested: TinyLlama-1.1B-Chat-v1.0
- Fused pairs: `input_layernorm → q/k/v_proj`, `post_attention_layernorm → gate/up_proj`
- Handles GQA (different k/v vs q dimensions) and linear layers without bias
- Patching: `from src.patch_llama import patch_llama_model`

## Transformer Engine Baseline

When `transformer-engine[pytorch]` is installed, the single-op benchmark includes `te.LayerNormLinear` as a production-grade baseline. TE uses highly optimized fused kernels with FP8 support.

## Benchmark JSON Output

All benchmark runs produce structured JSON files in `results/`:
- `results/single_op_<timestamp>.json`: single-operation benchmark results
- `results/e2e_<timestamp>.json`: end-to-end model benchmark results

**Current results** (H100, 2026-02-09):

| File | Entries | Contents |
|------|---------|----------|
| [`single_op_20260209T050038Z.json`](results/single_op_20260209T050038Z.json) | 120 | 24 configs (3 models x 2 layers x 4 batch sizes) x 5 variants (V0/V1/V2/V3/TE) |
| [`e2e_20260209T052124Z.json`](results/e2e_20260209T052124Z.json) | 36 | 2 models (OPT-1.3b, OPT-6.7b) x 3 variants (V0/V1/V3) x 6 batch sizes (1-32) |

Each entry includes full hardware/software metadata, workload configuration, and metrics (mean, stddev, speedup vs baseline). Example entry structure:
```json
{
  "config": {
    "benchmark_id": "fused_ln_linear_single_op_OPT-6.7b_attn_V1_fp32_32",
    "hardware": { "gpu_model": "NVIDIA H100 80GB HBM3", ... },
    "software": { "framework_version": "2.10.0+cu128", "runtime_version": "V1" },
    "model": { "name": "OPT-6.7b attn", "precision": "FP32" },
    "workload": { "batch_size": 32, "dimensions": { "h": 4096, "out_dim": 4096 } }
  },
  "metrics": {
    "single_op": { "mean_ms": 0.0542, "stddev_ms": 0.0021, "variant": "V1" },
    "speedup": { "vs_baseline": 2.17, "baseline_mean_ms": 0.1174 }
  }
}
```

Compatible with `jq` for command-line analysis:
```bash
# All speedups
jq '.[].metrics.speedup.vs_baseline' results/single_op_*.json

# Peak single-op speedup per variant
jq -r '[.[] | {v: .config.software.runtime_version, s: .metrics.speedup.vs_baseline}] | group_by(.v) | .[] | "\(.[0].v): \([.[].s] | max)"' results/single_op_*.json

# E2E peak speedups per model
jq -r '.[] | "\(.config.model.name) \(.config.software.runtime_version) batch=\(.config.workload.batch_size): \(.metrics.speedup.vs_baseline)x"' results/e2e_*.json | sort -t: -k2 -rn
```

## Limitations and scope

End-to-end benchmarks are OPT-only. RMSNorm/Llama support is implemented and passes correctness tests, but end-to-end Llama benchmarks have not been collected yet.

Inference only. No backward pass: the weight fusion is a one-way transformation suitable only for inference.

No comparison to TensorRT-LLM or vLLM. The Transformer Engine baseline gives one production reference point, but TRT-LLM and vLLM have their own fused LayerNorm kernels that may perform differently.
