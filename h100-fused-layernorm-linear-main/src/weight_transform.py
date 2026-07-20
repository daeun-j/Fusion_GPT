"""
Pre-compute fused LayerNorm+Linear weights.

For LayerNorm(gamma, beta) followed by Linear(W, b):
    output = Linear(LayerNorm(x))
           = x @ (I - E/h) @ diag(gamma) @ W.T / std(x) + beta @ W.T + b

where std(x) = sqrt(v(x)^2/h + eps), v(x) = ||x - mean(x)||_2.

The forward pass computes: x @ W_new.T / std(x) + b_new
where std(x) is derived from v(x) returned by the CUDA kernel.

Notation: E = 11^T is the h x h all-ones matrix (outer product of the all-ones
vector 1 with itself). Thus (I - E/h) is the centering matrix that subtracts
the mean from each column.

Assumptions: LayerNorm has both weight (gamma) and bias (beta);
normalized_shape is 1-dimensional.

So:
    M = (W * gamma).T            # element-wise (W * gamma), then .T = diag(gamma) @ W.T, shape [h, out]
    F_new = M - M.mean(dim=0)    # = (I - E/h) @ M, center each column, shape [h, out]
    W_new = F_new.T              # shape [out, h]
    b_new = beta @ W.T + b       # matrix multiply beta with W.T + bias, shape [out]
"""

import torch
import torch.nn as nn


def compute_fused_weights(
    ln: nn.LayerNorm,
    linear: nn.Linear,
) -> tuple[torch.Tensor, torch.Tensor, int, float]:
    """
    Compute fused weights W_new, b_new for LayerNorm + Linear fusion.

    Args:
        ln: LayerNorm module with weight (gamma) and bias (beta)
        linear: Linear module with weight W [out, h] and optional bias b [out]

    Returns:
        W_new: [out, h] fused weight
        b_new: [out] fused bias
        h: hidden dimension (for std computation)
        eps: LayerNorm epsilon
    """
    gamma = ln.weight.data       # [h]
    beta = ln.bias.data          # [h]
    W = linear.weight.data       # [out, h]
    b = linear.bias.data if linear.bias is not None else torch.zeros(
        W.size(0), device=W.device, dtype=W.dtype
    )
    h = ln.normalized_shape[0]
    eps = ln.eps

    # M = diag(gamma) @ W.T = (W * gamma).T, shape [h, out]
    M = (W * gamma).T

    # (I - E/h) @ M: center each column (subtract column mean)
    F_new = M - M.mean(dim=0, keepdim=True)

    W_new = F_new.T              # [out, h]
    b_new = beta @ W.T + b      # [out]: beta @ W.T is like linear(beta)

    return W_new, b_new, h, eps


def compute_fused_weights_rmsnorm(
    rms_norm,
    linear: nn.Linear,
) -> tuple[torch.Tensor, torch.Tensor, int, float]:
    """
    Compute fused weights W_new, b_new for RMSNorm + Linear fusion.

    RMSNorm: output = x / rms(x) * gamma, where rms(x) = sqrt(mean(x^2) + eps).
    Unlike LayerNorm, there is no mean subtraction and typically no bias.

    The fused forward computes: x @ W_new.T / rms(x) + b_new
    where:
        W_new = W * gamma   (element-wise, no centering needed)
        b_new = b           (no beta term since RMSNorm has no bias)

    Args:
        rms_norm: RMSNorm module with weight (gamma), no bias
        linear: Linear module with weight W [out, h] and optional bias b [out]

    Returns:
        W_new: [out, h] fused weight
        b_new: [out] fused bias
        h: hidden dimension
        eps: RMSNorm epsilon
    """
    gamma = rms_norm.weight.data    # [h]
    W = linear.weight.data          # [out, h]
    b = linear.bias.data if linear.bias is not None else torch.zeros(
        W.size(0), device=W.device, dtype=W.dtype
    )
    h = gamma.shape[0]
    eps = rms_norm.eps if hasattr(rms_norm, 'eps') else rms_norm.variance_epsilon

    # W_new = W * gamma (element-wise broadcast), shape [out, h]
    # No centering needed since RMSNorm doesn't subtract mean
    W_new = W * gamma

    # b_new = b (no beta in RMSNorm)
    b_new = b

    return W_new, b_new, h, eps


def transform_opt_layer(decoder_layer) -> dict:
    """
    Compute fused weights for all LayerNorm+Linear pairs in an OPT decoder layer.

    Pairs:
        1. self_attn_layer_norm -> q_proj, k_proj, v_proj
        2. final_layer_norm -> fc1

    Returns:
        dict mapping projection name to (W_new, b_new)
    """
    attn = decoder_layer.self_attn
    ln1 = decoder_layer.self_attn_layer_norm
    ln2 = decoder_layer.final_layer_norm

    fused = {}

    # Attention projections share the same layer norm
    for name in ["q_proj", "k_proj", "v_proj"]:
        proj = getattr(attn, name)
        W_new, b_new, h, eps = compute_fused_weights(ln1, proj)
        fused[f"attn_{name}"] = (W_new, b_new, h, eps)

    # FFN fc1
    W_new, b_new, h, eps = compute_fused_weights(ln2, decoder_layer.fc1)
    fused["fc1"] = (W_new, b_new, h, eps)

    return fused


def transform_llama_layer(decoder_layer) -> dict:
    """
    Compute fused weights for all RMSNorm+Linear pairs in a Llama decoder layer.

    Pairs:
        1. input_layernorm -> q_proj, k_proj, v_proj (attention)
        2. post_attention_layernorm -> gate_proj, up_proj (MLP)

    Note: down_proj is NOT fused (it follows an activation, not a norm).

    Returns:
        dict mapping projection name to (W_new, b_new, h, eps)
    """
    attn = decoder_layer.self_attn
    ln1 = decoder_layer.input_layernorm
    ln2 = decoder_layer.post_attention_layernorm

    fused = {}

    # Attention projections share input_layernorm
    for name in ["q_proj", "k_proj", "v_proj"]:
        proj = getattr(attn, name)
        W_new, b_new, h, eps = compute_fused_weights_rmsnorm(ln1, proj)
        fused[f"attn_{name}"] = (W_new, b_new, h, eps)

    # MLP gate_proj and up_proj share post_attention_layernorm
    for name in ["gate_proj", "up_proj"]:
        proj = getattr(decoder_layer.mlp, name)
        W_new, b_new, h, eps = compute_fused_weights_rmsnorm(ln2, proj)
        fused[name] = (W_new, b_new, h, eps)

    return fused
