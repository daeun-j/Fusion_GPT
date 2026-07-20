"""
Monkey-patch an OPT model to use fused LayerNorm+Linear modules.

Replaces the forward pass of each OPT decoder layer so that:
  - self_attn_layer_norm + q/k/v_proj -> fused_q/k/v_proj
  - final_layer_norm + fc1 -> fused_fc1

The layer norms are skipped in the forward pass; their effect is baked
into the fused weight matrices.

Compatible with transformers v5 API (Cache-based KV, attention_interface).
"""

import torch
import torch.nn as nn
from typing import Callable
from transformers.models.opt.modeling_opt import (
    OPTDecoderLayer, OPTAttention,
    ALL_ATTENTION_FUNCTIONS, eager_attention_forward,
)

from src.weight_transform import transform_opt_layer
from src.fused_forward import FusedLNLinear, FusedLNLinearV1, FusedLNLinearV3

_VARIANT_CLASSES = {
    "V0": FusedLNLinear,
    "V1": FusedLNLinearV1,
    "V3": FusedLNLinearV3,
}


def patch_opt_model(model, device=None, variant="V0"):
    """
    Patch all decoder layers in an OPT model to use fused LN+Linear.

    Args:
        model: HuggingFace OPTForCausalLM model
        device: target device (defaults to model's device)
        variant: kernel variant — "V0" (stream-based), "V1" (fused normalize),
                 or "V3" (Welford + fused normalize + 512 threads)

    Returns:
        The patched model (modified in-place)
    """
    if variant not in _VARIANT_CLASSES:
        raise ValueError(f"Unknown variant {variant!r}; choose from {list(_VARIANT_CLASSES)}")

    if device is None:
        device = next(model.parameters()).device

    # Only V0 needs a separate CUDA stream
    denom_stream = torch.cuda.Stream(device=device) if variant == "V0" else None

    for layer_idx, layer in enumerate(model.model.decoder.layers):
        _patch_decoder_layer(layer, denom_stream, device, variant)

    return model


def _patch_decoder_layer(layer: OPTDecoderLayer, denom_stream, device, variant="V0"):
    """Patch a single OPT decoder layer."""
    fused_weights = transform_opt_layer(layer)
    cls = _VARIANT_CLASSES[variant]

    # Create fused modules for attention projections
    for proj_name in ["q_proj", "k_proj", "v_proj"]:
        W_new, b_new, h, eps = fused_weights[f"attn_{proj_name}"]
        if variant == "V0":
            fused_mod = cls(W_new.to(device), b_new.to(device), denom_stream, h, eps)
        else:
            fused_mod = cls(W_new.to(device), b_new.to(device), h, eps)
        setattr(layer.self_attn, f"fused_{proj_name}", fused_mod)

    # Create fused module for fc1
    W_new, b_new, h, eps = fused_weights["fc1"]
    if variant == "V0":
        layer.fused_fc1 = cls(W_new.to(device), b_new.to(device), denom_stream, h, eps)
    else:
        layer.fused_fc1 = cls(W_new.to(device), b_new.to(device), h, eps)

    # Patch the attention forward
    _patch_attention_forward(layer.self_attn)

    # Patch the decoder layer forward
    _patch_layer_forward(layer)


def _patch_attention_forward(attn: OPTAttention):
    """Replace attention forward to use fused q/k/v projections."""

    def patched_forward(
        hidden_states: torch.Tensor,
        past_key_values=None,
        attention_mask=None,
        output_attentions: bool = False,
        cache_position=None,
        **kwargs,
    ):
        # hidden_states is RAW (no LN applied) - fused modules handle LN internally
        bsz, tgt_len, _ = hidden_states.size()

        query_states = attn.fused_q_proj(hidden_states) * attn.scaling
        query_states = query_states.view(bsz, -1, attn.num_heads, attn.head_dim).transpose(1, 2)

        key_states = attn.fused_k_proj(hidden_states)
        value_states = attn.fused_v_proj(hidden_states)
        key_states = key_states.view(bsz, -1, attn.num_heads, attn.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, -1, attn.num_heads, attn.head_dim).transpose(1, 2)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, attn.layer_idx, {"cache_position": cache_position}
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
            dropout=0.0 if not attn.training else attn.dropout,
            scaling=1.0,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
        attn_output = attn.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights

    attn.forward = patched_forward


def _patch_layer_forward(layer: OPTDecoderLayer):
    """Replace decoder layer forward to skip standalone LayerNorm calls."""

    def patched_forward(
        hidden_states: torch.Tensor,
        attention_mask=None,
        past_key_values=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_ids=None,
        cache_position=None,
        **kwargs,
    ):
        residual = hidden_states

        # Skip self_attn_layer_norm: fused q/k/v projections handle LN internally
        # (The original code applies LN before attention when do_layer_norm_before=True)
        hidden_states, self_attn_weights = layer.self_attn(
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            position_ids=position_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=layer.dropout, training=layer.training)
        hidden_states = residual + hidden_states

        # FFN
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        # Skip final_layer_norm: fused_fc1 handles LN internally
        hidden_states = layer.fused_fc1(hidden_states)
        hidden_states = layer.activation_fn(hidden_states)

        hidden_states = layer.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=layer.dropout, training=layer.training)

        hidden_states = (residual + hidden_states).view(hidden_states_shape)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs

    layer.forward = patched_forward
