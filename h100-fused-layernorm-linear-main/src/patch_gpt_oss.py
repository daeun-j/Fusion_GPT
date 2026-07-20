"""
Monkey-patch a GPT-OSS model to use fused RMSNorm+Linear on the attention input path.

Strategy (module swap — no forward rewrites):
  - attn.q/k/v_proj -> FusedRMSNormLinearV1/V3
      (gamma absorbed into W_new; the 1/rms division + bias run in the kernel
       epilogue over the matmul output)
  - layer.input_layernorm -> nn.Identity
      (raw hidden states flow into the fused projections)

The original GptOssAttention.forward is untouched: it simply calls
self.q_proj(hidden_states) etc., so swapping the modules is sufficient and
keeps compatibility with sinks / sliding-window / any attn implementation.

post_attention_layernorm is deliberately LEFT AS-IS:
  its output feeds the MoE router AND the 32 experts' batched GEMMs. Folding
  the 1/rms division into those consumers would (a) require rewriting
  GptOssExperts to separate the expert biases, and (b) divide tensors larger
  than the input (top-4 experts x 5760 dims per token vs. 2880), i.e. more
  work than the norm it removes. With a pre-fused checkpoint the gamma is
  already absorbed into router/expert weights, so this norm is a pure x/rms.

Works on both:
  - pre-fused checkpoints (norm weight == 1, gamma already inside q/k/v) and
  - original checkpoints (gamma is absorbed on the fly at patch time);
  the on-the-fly multiply is a no-op when gamma == 1.

Usage:
    import sys; sys.path.insert(0, "/path/to/h100-fused-layernorm-linear-main")
    from src.patch_gpt_oss import patch_gpt_oss_model
    model = AutoModelForCausalLM.from_pretrained(..., device_map="cuda")
    model = patch_gpt_oss_model(model, variant="V1")
"""

import torch
import torch.nn as nn

from src.fused_forward import FusedRMSNormLinearV1, FusedRMSNormLinearV3

_VARIANT_CLASSES = {
    "V1": FusedRMSNormLinearV1,
    "V3": FusedRMSNormLinearV3,
}


def _norm_eps(norm, config) -> float:
    for attr in ("variance_epsilon", "eps"):
        if hasattr(norm, attr):
            return float(getattr(norm, attr))
    return float(getattr(config, "rms_norm_eps", 1e-5))


@torch.no_grad()
def patch_gpt_oss_model(model, variant: str = "V1"):
    """
    Patch all decoder layers of a GptOssForCausalLM in-place.

    Replaces q/k/v_proj with fused RMSNorm+Linear modules and the
    input_layernorm with Identity. Returns the patched model.
    """
    if variant not in _VARIANT_CLASSES:
        raise ValueError(f"Unknown variant {variant!r}; choose from {list(_VARIANT_CLASSES)}")
    cls = _VARIANT_CLASSES[variant]

    config = model.config
    h = config.hidden_size
    n_patched = 0

    for layer in model.model.layers:
        attn = layer.self_attn
        norm = layer.input_layernorm
        if isinstance(norm, nn.Identity):
            continue  # already patched
        eps = _norm_eps(norm, config)
        gamma = norm.weight.data

        for name in ("q_proj", "k_proj", "v_proj"):
            lin = getattr(attn, name)
            if not isinstance(lin, nn.Linear):
                raise TypeError(f"{name} is {type(lin).__name__}, expected nn.Linear")
            # Absorb gamma in FP32, cast back (no-op if checkpoint is pre-fused)
            W_new = (lin.weight.data.float() * gamma.float().unsqueeze(0)) \
                .to(lin.weight.dtype).contiguous()
            if lin.bias is not None:
                b_new = lin.bias.data.clone().contiguous()
            else:
                b_new = torch.zeros(W_new.shape[0], dtype=W_new.dtype, device=W_new.device)
            setattr(attn, name, cls(W_new, b_new, h, eps))

        layer.input_layernorm = nn.Identity()
        n_patched += 1

    print(f"[patch_gpt_oss] patched {n_patched} layers "
          f"(variant={variant}, h={h}, fused: input_layernorm -> q/k/v)")
    return model
