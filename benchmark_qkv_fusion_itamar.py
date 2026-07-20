"""
Checkpoint-level fused vs non-fused benchmark using Itamar's procedure,
applied to the GPT-OSS attention input path (RMSNorm -> q/k/v).

Reuses Itamar's benchmark_rmsnorm_linear_fusion.py verbatim for everything it
defines: shape sweep, warmup/measure iteration counts, DTYPE (fp16),
measure_latency / measure_peak_memory / measure_numerical_equivalence,
ShapeResult, summary table, and CSV schema. Only the model-loading step
differs, because the standard layer extraction cannot call self_attn(x) with
a bare tensor (rotary args) — so this script extracts the exact fused unit:

  non-fused checkpoint -> baseline module:  q,k,v = proj(input_layernorm(x))
  fused checkpoint     -> kernel module:    q,k,v = fused_proj(x)   (V1 or V3)

One kernel variant per run, one Itamar-format CSV per variant:

    python benchmark_qkv_fusion_itamar.py --dir ~/bench --variant V1
    python benchmark_qkv_fusion_itamar.py --dir ~/bench --variant V3

Expects <dir>/models/fused/ and <dir>/models/non-fused/ like the original.
"""

import argparse
import gc
import os
import sys

import torch
import torch.nn as nn

import benchmark_rmsnorm_linear_fusion as B  # Itamar's script, used as a library


class BaselineQKV(nn.Module):
    """Non-fused unit: input_layernorm (as shipped in the checkpoint) -> q/k/v."""

    def __init__(self, norm, q, k, v):
        super().__init__()
        self.norm, self.q, self.k, self.v = norm, q, k, v

    def forward(self, x):
        n = self.norm(x)
        return self.q(n), self.k(n), self.v(n)


class FusedQKV(nn.Module):
    """Fused unit: three FusedRMSNormLinear modules consuming raw x."""

    def __init__(self, fq, fk, fv):
        super().__init__()
        self.fq, self.fk, self.fv = fq, fk, fv

    def forward(self, x):
        return self.fq(x), self.fk(x), self.fv(x)


def extract_attn_unit(model_dir: str, label: str, layer_idx: int):
    """Load a full checkpoint (Itamar's loader), pull out one layer's norm+qkv."""
    full = B._load_hf_model(model_dir, label)
    layer = full.model.layers[layer_idx]
    norm = layer.input_layernorm
    eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-5)))
    gamma = norm.weight.data.clone()
    parts = {}
    for name in ("q_proj", "k_proj", "v_proj"):
        lin = getattr(layer.self_attn, name)
        w = lin.weight.data.clone()
        b = lin.bias.data.clone() if lin.bias is not None else torch.zeros(w.shape[0], dtype=w.dtype)
        parts[name] = (w, b)
    norm = norm.cpu()
    del full
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  {label}: extracted layer {layer_idx} input_layernorm + q/k/v "
          f"(q={tuple(parts['q_proj'][0].shape)}, eps={eps})")
    return norm, gamma, parts, eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True,
                    help="Base dir containing models/fused/ and models/non-fused/")
    ap.add_argument("--variant", default="V1", choices=["V1", "V3"],
                    help="Fused kernel variant for this run")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--fusion-repo", default="h100-fused-layernorm-linear-main")
    ap.add_argument("--out", default=None,
                    help="Output CSV (default: benchmark_results_qkv_<variant>.csv)")
    args = ap.parse_args()
    out_path = args.out or f"benchmark_results_qkv_{args.variant}.csv"

    sys.path.insert(0, os.path.abspath(args.fusion_repo))
    from src.fused_forward import FusedRMSNormLinearV1, FusedRMSNormLinearV3
    cls = {"V1": FusedRMSNormLinearV1, "V3": FusedRMSNormLinearV3}[args.variant]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}  CUDA: {torch.version.cuda}")
    print(f"Variant: {args.variant}  Layer: {args.layer}")
    print(f"Warmup iters: {B.WARMUP_ITERS}  |  Measure iters: {B.MEASURE_ITERS}")

    # --- Non-fused checkpoint -> baseline module (norm + q/k/v as shipped) ---
    nf_dir = os.path.join(args.dir, "models", "non-fused")
    norm, _, nf_parts, _ = extract_attn_unit(nf_dir, "non-fused", args.layer)
    lins = {}
    for name, (w, b) in nf_parts.items():
        lin = nn.Linear(w.shape[1], w.shape[0], bias=True)
        lin.weight.data, lin.bias.data = w, b
        lins[name] = lin
    nonfused = BaselineQKV(norm, lins["q_proj"], lins["k_proj"], lins["v_proj"])
    nonfused = nonfused.to(B.DEVICE, dtype=B.DTYPE).eval()
    hidden_dim = nf_parts["q_proj"][0].shape[1]
    out_dim = sum(w.shape[0] for (w, _) in nf_parts.values())

    # --- Fused checkpoint -> fused kernel modules (gamma already absorbed) ---
    f_dir = os.path.join(args.dir, "models", "fused")
    _, f_gamma, f_parts, f_eps = extract_attn_unit(f_dir, "fused", args.layer)
    mods = []
    for name in ("q_proj", "k_proj", "v_proj"):
        w, b = f_parts[name]
        # absorb the checkpoint's own norm weight (== 1 for a pre-fused checkpoint)
        w_new = (w.float() * f_gamma.float().unsqueeze(0)).to(B.DTYPE)
        mods.append(cls(w_new.contiguous().to(B.DEVICE),
                        b.to(B.DTYPE).contiguous().to(B.DEVICE),
                        hidden_dim, f_eps))
    fused = FusedQKV(*mods).to(B.DEVICE).eval()

    # --- Itamar's measurement loop, verbatim metrics ---
    shape_sweep = [(batch, seq_len, hidden_dim) for (batch, seq_len, _, _) in B.SHAPE_SWEEP]
    results = []
    for (batch, seq_len, hidden) in shape_sweep:
        print(f"\n{'=' * 60}")
        print(f"Shape: batch={batch}, seq={seq_len}, hidden={hidden}, out={out_dim}")
        print(f"{'=' * 60}")

        x = torch.randn(batch, seq_len, hidden, device=B.DEVICE, dtype=B.DTYPE)
        result = B.ShapeResult(batch=batch, seq_len=seq_len, hidden=hidden, out_dim=out_dim)

        print("  Measuring latency (non-fused)...")
        result.nonfused_latencies = B.measure_latency(nonfused, x)
        print("  Measuring latency (fused)...")
        result.fused_latencies = B.measure_latency(fused, x)

        print("  Measuring peak memory...")
        result.nonfused_peak_mem_mb = B.measure_peak_memory(nonfused, x)
        result.fused_peak_mem_mb = B.measure_peak_memory(fused, x)

        print("  Measuring numerical equivalence...")
        (result.max_abs_diff,
         result.cosine_sim,
         result.kl_divergence) = B.measure_numerical_equivalence(fused, nonfused, x)

        print(f"\n  Latency (median ms):  fused={result.fused_median_ms:.3f}  "
              f"non-fused={result.nonfused_median_ms:.3f}  speedup={result.speedup:.2f}x")
        print(f"  Numerical: max|diff|={result.max_abs_diff:.6f}  "
              f"cos={result.cosine_sim:.6f}  kl={result.kl_divergence:.6f}")

        del x
        gc.collect()
        torch.cuda.empty_cache()
        results.append(result)

    B.print_summary_table(results)
    B.save_csv(results, out_path)


if __name__ == "__main__":
    main()
