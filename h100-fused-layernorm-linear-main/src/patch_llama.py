"""
Monkey-patch a Llama model to use fused RMSNorm+Linear modules.

Replaces the forward pass of each Llama decoder layer so that:
  - input_layernorm + q/k/v_proj -> fused_q/k/v_proj
  - post_attention_layernorm + gate_proj, up_proj -> fused_gate_proj, fused_up_proj

The RMSNorm layers are skipped in the forward pass; their effect (gamma
scaling and 1/rms normalization) is baked into the fused weight matrices.

Note: down_proj is NOT fused (it follows an activation, not a norm).
"""

import torch
import torch.nn as nn
from typing import Callable
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer, LlamaAttention,
    ALL_ATTENTION_FUNCTIONS, eager_attention_forward,
    apply_rotary_pos_emb,
)

from src.weight_transform import transform_llama_layer
from src.fused_forward import FusedRMSNormLinearV1, FusedRMSNormLinearV3

_VARIANT_CLASSES = {
    "V1": FusedRMSNormLinearV1,
    "V3": FusedRMSNormLinearV3,
}


def patch_llama_model(model, device=None, variant="V1"):
    """
    Patch all decoder layers in a Llama model to use fused RMSNorm+Linear.

    Args:
        model: HuggingFace LlamaForCausalLM model
        device: target device (defaults to model's device)
        variant: kernel variant -- "V1" (256 threads) or "V3" (512 threads)

    Returns:
        The patched model (modified in-place)
    """
    if variant not in _VARIANT_CLASSES:
        raise ValueError(f"Unknown variant {variant!r}; choose from {list(_VARIANT_CLASSES)}")

    if device is None:
        device = next(model.parameters()).device

    for layer_idx, layer in enumerate(model.model.layers):
        _patch_decoder_layer(layer, device, variant)

    return model


def _patch_decoder_layer(layer: LlamaDecoderLayer, device, variant="V1"):
    """Patch a single Llama decoder layer."""
    fused_weights = transform_llama_layer(layer)
    cls = _VARIANT_CLASSES[variant]

    # Fused attention projections (q/k/v share input_layernorm)
    for proj_name in ["q_proj", "k_proj", "v_proj"]:
        W_new, b_new, h, eps = fused_weights[f"attn_{proj_name}"]
        fused_mod = cls(W_new.to(device), b_new.to(device), h, eps)
        setattr(layer.self_attn, f"fused_{proj_name}", fused_mod)

    # Fused MLP projections (gate_proj, up_proj share post_attention_layernorm)
    for proj_name in ["gate_proj", "up_proj"]:
        W_new, b_new, h, eps = fused_weights[proj_name]
        fused_mod = cls(W_new.to(device), b_new.to(device), h, eps)
        setattr(layer.mlp, f"fused_{proj_name}", fused_mod)

    # Patch forwards
    _patch_attention_forward(layer.self_attn)
    _patch_layer_forward(layer)


def _patch_attention_forward(attn: LlamaAttention):
    """Replace attention forward to use fused q/k/v projections (skip RMSNorm)."""

    def patched_forward(
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)

        # Fused projections: RMSNorm is baked in, input is raw hidden_states
        query_states = attn.fused_q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = attn.fused_k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = attn.fused_v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, attn.layer_idx, cache_kwargs
            )

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            attn.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            attn,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not attn.training else attn.attention_dropout,
            scaling=attn.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn.o_proj(attn_output)
        return attn_output, attn_weights

    attn.forward = patched_forward


def _patch_layer_forward(layer: LlamaDecoderLayer):
    """Replace decoder layer forward to skip standalone RMSNorm calls."""

    def patched_forward(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        # Skip input_layernorm: fused q/k/v projections handle RMSNorm internally
        hidden_states, _ = layer.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # MLP with fused gate_proj and up_proj (skip post_attention_layernorm)
        residual = hidden_states
        # Replicate LlamaMLP.forward but using fused projections
        gate = layer.mlp.fused_gate_proj(hidden_states)
        up = layer.mlp.fused_up_proj(hidden_states)
        hidden_states = layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)
        hidden_states = residual + hidden_states

        return hidden_states

    layer.forward = patched_forward
