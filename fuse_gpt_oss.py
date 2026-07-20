"""
Create an RMSNorm+Linear "fused" GPT-OSS checkpoint (FlashNorm-style gamma absorption).

What it does, per decoder layer:
  - input_layernorm.gamma          -> absorbed into self_attn.q/k/v_proj.weight
  - post_attention_layernorm.gamma -> absorbed into mlp.router.weight AND
                                      mlp.experts.gate_up_proj (per-expert, input dim)
  - both RMSNorm weights are then set to 1.0

The result is mathematically identical to the original model (RMSNorm still
normalizes, but its gamma scaling now lives inside the downstream linears),
stays a standard GPT-OSS architecture, and loads with plain
AutoModelForCausalLM / vLLM / SGLang without custom code.

Usage:
    python fuse_gpt_oss.py \
        --model-dir  ~/models/gpt-oss-20b-bf16 \
        --out-dir    ~/bench/models/fused \
        [--device cuda] [--skip-check]

Notes:
  - down_proj is NOT touched (it follows the activation, not a norm).
  - o_proj is NOT touched (its input is attention output, not a norm output).
  - The final model.norm (pre-lm_head) is NOT fused here; fusing it into
    lm_head is possible but kept out of scope to match the per-layer benchmark.
  - Transform is done in FP32 and cast back to the original dtype.
"""

import argparse
import gc
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CHECK_PROMPT = "The capital of France is"


def absorb_gamma_2d(weight: torch.nn.Parameter, gamma: torch.Tensor) -> None:
    """weight [out, in] consuming RMSNorm(x)*gamma: fold gamma into columns."""
    assert weight.ndim == 2 and weight.shape[1] == gamma.shape[0], (
        f"shape mismatch: weight {tuple(weight.shape)} vs gamma {tuple(gamma.shape)}"
    )
    w = weight.data.float() * gamma.float().unsqueeze(0)
    weight.data = w.to(weight.dtype)


def absorb_gamma_experts(weight: torch.nn.Parameter, gamma: torch.Tensor) -> None:
    """gate_up_proj [num_experts, hidden, 2*expert_dim]: fold gamma into dim 1."""
    assert weight.ndim == 3 and weight.shape[1] == gamma.shape[0], (
        f"shape mismatch: weight {tuple(weight.shape)} vs gamma {tuple(gamma.shape)}"
    )
    w = weight.data.float() * gamma.float().view(1, -1, 1)
    weight.data = w.to(weight.dtype)


@torch.no_grad()
def fuse_layer(layer) -> None:
    # --- attention side: input_layernorm -> q/k/v ---
    gamma_in = layer.input_layernorm.weight.data.clone()
    for name in ("q_proj", "k_proj", "v_proj"):
        absorb_gamma_2d(getattr(layer.self_attn, name).weight, gamma_in)
    layer.input_layernorm.weight.data.fill_(1.0)

    # --- MoE side: post_attention_layernorm -> router + experts.gate_up_proj ---
    gamma_post = layer.post_attention_layernorm.weight.data.clone()
    router = layer.mlp.router
    if hasattr(router, "weight"):
        absorb_gamma_2d(router.weight, gamma_post)
    else:
        raise RuntimeError("router has no .weight — inspect mlp.router structure")
    absorb_gamma_experts(layer.mlp.experts.gate_up_proj, gamma_post)
    layer.post_attention_layernorm.weight.data.fill_(1.0)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="local BF16 GPT-OSS checkpoint")
    ap.add_argument("--out-dir", required=True, help="where to save the fused checkpoint")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip-check", action="store_true", help="skip logits parity check")
    args = ap.parse_args()

    print(f"Loading {args.model_dir} on {args.device} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype="auto", device_map=args.device,
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    print(f"Model class: {type(model).__name__}, dtype: {model.dtype}")

    inputs = tok(CHECK_PROMPT, return_tensors="pt").to(args.device)
    logits_before = None
    if not args.skip_check:
        logits_before = model(**inputs).logits.float().cpu()

    layers = model.model.layers
    print(f"Fusing {len(layers)} layers ...")
    for i, layer in enumerate(layers):
        fuse_layer(layer)
        if (i + 1) % 6 == 0:
            print(f"  {i + 1}/{len(layers)} layers done")

    if not args.skip_check:
        logits_after = model(**inputs).logits.float().cpu()
        max_diff = (logits_before - logits_after).abs().max().item()
        cos = torch.nn.functional.cosine_similarity(
            logits_before.flatten(), logits_after.flatten(), dim=0
        ).item()
        print(f"Parity check: max|diff|={max_diff:.6f}  cosine={cos:.8f}")
        # BF16 rounding from the gamma re-quantization is expected; anything
        # beyond ~1e-1 on raw logits means a wiring bug, not rounding.
        if cos < 0.999:
            raise RuntimeError("Parity check failed — do not upload this checkpoint.")

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Saving fused checkpoint to {args.out_dir} ...")
    model.save_pretrained(args.out_dir, safe_serialization=True)
    # Copy tokenizer files verbatim instead of tok.save_pretrained():
    # re-serializing under transformers 5.x stamps a tokenizer_class
    # ("TokenizersBackend") that transformers 4.x cannot load.
    import shutil
    for fname in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                  "vocab.json", "merges.txt", "chat_template.jinja", "generation_config.json"):
        src = os.path.join(args.model_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, args.out_dir)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Done.")


if __name__ == "__main__":
    main()
