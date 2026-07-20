# CUDA Task: Fused LayerNorm + Linear

## Task Description

Implement a fused LayerNorm + Linear layer operation for transformer inference optimization.

### Background

In transformer architectures the pattern `Linear(LayerNorm(x))` appears frequently. Every attention projection (Q, K, V) and the first FFN layer are preceded by LayerNorm. The standard approach executes these sequentially:

1. Compute LayerNorm: normalize `x`, scale by `gamma`, shift by `beta`
2. Compute Linear: multiply by weight matrix `W`, add bias `b`

### Goal

Fuse these two operations to reduce latency by:
- Pre-computing combined weight matrices that absorb the LayerNorm scaling
- Running the expensive matrix multiplication concurrently with a lightweight "denominator" kernel that computes the normalization factor
- Eliminating the intermediate materialization of the normalized tensor

### Mathematical Foundation

**LayerNorm:**

$$\text{LayerNorm}(x) = \frac{x - \mu(x)}{\sigma(x)} \cdot \gamma + \beta$$

where $\mu(x) = \frac{1}{h}\sum_i x_i$ and $\sigma(x) = \sqrt{\frac{1}{h}\sum_i (x_i - \mu)^2 + \epsilon}$.

**Composed operation:**

$$\text{Linear}(\text{LayerNorm}(x)) = \frac{(x - \mu) \cdot \gamma}{\ \sigma} \cdot W^T + \beta \cdot W^T + b$$

Key insight, define the denominator:

$$v(x) = \|x - \mu(x)\|_2$$

Then: $\sigma(x) = \sqrt{v(x)^2 / h + \epsilon}$

**Pre-computed weights:**

$$M = (\text{diag}(\gamma) \cdot W^T) = (W \odot \gamma)^T$$

$$W_{\text{new}} = \left[(I - \frac{\mathbf{1}\mathbf{1}^T}{h}) \cdot M\right]^T$$

$$b_{\text{new}} = \beta \cdot W^T + b$$

**Fused forward pass:**

$$\text{raw} = x \cdot W_{\text{new}}^T \quad \text{(matrix multiply, expensive)}$$

$$v = \|x - \mu(x)\|_2 \quad \text{(denominator kernel, cheap)}$$

$$\text{output} = \frac{\text{raw}}{\sqrt{v^2/h + \epsilon}} + b_{\text{new}}$$

The matmul and denominator computation are independent, so they can run concurrently on separate CUDA streams.

### Requirements

1. **CUDA kernel** for computing $v(x) = \|x - \mu(x)\|_2$ per row
2. **Weight transform** to pre-compute $W_{\text{new}}$ and $b_{\text{new}}$
3. **Fused forward pass** with CUDA stream concurrency
4. **Integration** with HuggingFace OPT models (monkey-patching)
5. **Correctness tests** comparing fused vs original outputs
6. **Performance benchmarks** measuring speedup across configurations

### Target Hardware

- NVIDIA H100 80GB HBM3
- CUDA 13.1 (driver) / cu128 (PyTorch)
- Python 3.12.3, PyTorch 2.x, transformers 5.1.0
