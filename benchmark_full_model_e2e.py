"""
Full-model E2E benchmark: fused vs non-fused GPT-OSS (whole forward pass).

Unlike benchmark_rmsnorm_linear_fusion.py (which extracts ONE sub-module),
this loads the ENTIRE model — all layers — and benchmarks a full forward
pass over the same shape sweep, reporting the same metrics:
latency (median / p99), throughput (tok/s), peak GPU memory, and numerical
equivalence (max |diff|, cosine similarity, KL divergence).

Because two 20B BF16 models do not fit on one 80GB GPU simultaneously, the
script runs in two phases: it loads the non-fused model, measures every
shape and stores the last-token logits for fixed-seed inputs, frees it,
then loads the fused model and repeats with the SAME inputs. Equivalence is
computed from the stored logits pairs.

Notes vs. the standardized single-layer procedure:
  - Inputs are token ids (fixed seed), not random hidden states.
  - Equivalence is measured on last-token logits (logits_to_keep=1), i.e.
    the distribution actually used for generation. Full-sequence logits for
    the largest shapes would need ~26GB+ just for the logits tensor.
  - Fewer iterations by default (full forward is ~100x slower than one
    layer): warmup 10, measure 50. Override with --warmup / --measure.

Usage:
    python benchmark_full_model_e2e.py --dir ~/bench [--warmup 10] [--measure 50]
        expects <dir>/models/fused/ and <dir>/models/non-fused/
"""

import argparse
import csv
import gc
import os
import statistics
import time
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

SHAPES: List[Tuple[int, int]] = [
    (1, 128), (1, 512), (1, 2048),
    (8, 128), (8, 512), (8, 2048),
    (32, 128), (32, 512), (32, 2048),
]
SEED = 1234
DEVICE = "cuda"


def percentile(data: List[float], pct: int) -> float:
    s = sorted(data)
    return s[min(int(len(s) * pct / 100), len(s) - 1)]


def make_inputs(vocab_size: int, batch: int, seq: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(SEED + batch * 100_000 + seq)
    return torch.randint(0, vocab_size, (batch, seq), generator=g).to(DEVICE)


def load_model(model_dir: str, label: str, attn: str):
    print(f"\nLoading {label} from {model_dir} (attn={attn}) ...")
    kwargs = {}
    if attn != "auto":
        kwargs["attn_implementation"] = attn
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype="auto", device_map=DEVICE, **kwargs,
    )
    model.eval()
    return model


def apply_kernel_patch(model, variant: str, fusion_repo: str):
    """Swap q/k/v_proj for fused RMSNorm+Linear kernel modules (skips input_layernorm)."""
    import sys
    repo = os.path.abspath(fusion_repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from src.patch_gpt_oss import patch_gpt_oss_model  # JIT-builds the CUDA ext on first import
    return patch_gpt_oss_model(model, variant=variant)


@torch.no_grad()
def run_phase(model_dir: str, label: str, warmup: int, measure: int, attn: str,
              patch_variant: str = "none", fusion_repo: str = "") -> Dict:
    """Load one model, measure all shapes, return per-shape stats + logits."""
    model = load_model(model_dir, label, attn)
    if patch_variant != "none":
        model = apply_kernel_patch(model, patch_variant, fusion_repo)
    vocab_size = model.config.vocab_size

    # logits_to_keep=1 computes lm_head only for the last position, keeping
    # peak memory bounded at large shapes. Fall back if the kwarg is absent.
    def make_fwd(ids):
        def fwd():
            try:
                return model(input_ids=ids, use_cache=False, logits_to_keep=1).logits
            except TypeError:
                return model(input_ids=ids, use_cache=False).logits[:, -1:, :]
        return fwd

    results = {}
    for (batch, seq) in SHAPES:
        ids = make_inputs(vocab_size, batch, seq)
        fwd = make_fwd(ids)

        try:
            for _ in range(warmup):
                fwd()
            torch.cuda.synchronize()

            latencies = []
            for _ in range(measure):
                t0 = time.perf_counter()
                fwd()
                torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000.0)

            torch.cuda.reset_peak_memory_stats(DEVICE)
            logits = fwd()
            torch.cuda.synchronize()
            peak_mb = torch.cuda.max_memory_allocated(DEVICE) / 1024 ** 2

            results[(batch, seq)] = {
                "latencies": latencies,
                "peak_mb": peak_mb,
                "logits": logits.float().cpu(),
            }
            med = statistics.median(latencies)
            print(f"  [{label}] batch={batch:>2} seq={seq:>4}: "
                  f"median={med:.2f}ms  p99={percentile(latencies, 99):.2f}ms  "
                  f"tok/s={batch * seq / (med / 1000):,.0f}  peak={peak_mb:.0f}MB")
        except torch.OutOfMemoryError:
            print(f"  [{label}] batch={batch:>2} seq={seq:>4}: OOM — skipped")
            results[(batch, seq)] = None

        del ids
        gc.collect()
        torch.cuda.empty_cache()

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


def equivalence(logits_f: torch.Tensor, logits_nf: torch.Tensor):
    """max |diff|, mean cosine sim, mean KL on last-token logits [b, 1, vocab]."""
    max_diff = (logits_f - logits_nf).abs().max().item()
    f_flat = logits_f.view(logits_f.size(0), -1)
    nf_flat = logits_nf.view(logits_nf.size(0), -1)
    cos = F.cosine_similarity(f_flat, nf_flat, dim=1).mean().item()
    p = F.softmax(logits_f, dim=-1).clamp(min=1e-10)
    q = F.softmax(logits_nf, dim=-1).clamp(min=1e-10)
    kl = (p * (p / q).log()).sum(dim=-1).mean().item()
    return max_diff, cos, kl


def main():
    ap = argparse.ArgumentParser(description="Full-model fused vs non-fused E2E benchmark")
    ap.add_argument("--dir", required=True,
                    help="Base dir containing models/fused/ and models/non-fused/")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--measure", type=int, default=50)
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2", "auto"],
                    help="Attention implementation (default sdpa; eager OOMs at large shapes)")
    ap.add_argument("--patch-fused", default="none", choices=["none", "V1", "V3"],
                    help="Apply the fused RMSNorm+Linear CUDA kernel patch to the FUSED model")
    ap.add_argument("--fusion-repo", default="h100-fused-layernorm-linear-main",
                    help="Path to the kernel-fusion repo (for --patch-fused)")
    ap.add_argument("--out", default="benchmark_full_model_results.csv")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}  CUDA: {torch.version.cuda}")
    print(f"Warmup: {args.warmup}  Measure: {args.measure}")
    print(f"Shapes: {SHAPES}")

    nonfused = run_phase(os.path.join(args.dir, "models", "non-fused"),
                         "non-fused", args.warmup, args.measure, args.attn)
    fused = run_phase(os.path.join(args.dir, "models", "fused"),
                      "fused", args.warmup, args.measure, args.attn,
                      patch_variant=args.patch_fused, fusion_repo=args.fusion_repo)

    header = (f"{'batch':>5} {'seq':>5} {'fused_med':>10} {'nf_med':>10} {'speedup':>8} "
              f"{'fused_p99':>10} {'nf_p99':>10} {'fused_mem':>10} {'nf_mem':>10} "
              f"{'max_diff':>10} {'cos_sim':>9} {'kl_div':>9}")
    print(f"\n{'=' * len(header)}\nSUMMARY (full model, all layers)\n{'=' * len(header)}")
    print(header)
    print("-" * len(header))

    rows = []
    for (batch, seq) in SHAPES:
        f, nf = fused[(batch, seq)], nonfused[(batch, seq)]
        if f is None or nf is None:
            print(f"{batch:>5} {seq:>5}  (skipped — OOM)")
            continue
        f_med, nf_med = statistics.median(f["latencies"]), statistics.median(nf["latencies"])
        max_diff, cos, kl = equivalence(f["logits"], nf["logits"])
        row = {
            "batch": batch, "seq_len": seq,
            "fused_median_ms": f_med, "nonfused_median_ms": nf_med,
            "speedup": nf_med / f_med,
            "fused_p99_ms": percentile(f["latencies"], 99),
            "nonfused_p99_ms": percentile(nf["latencies"], 99),
            "fused_throughput": batch * seq / (f_med / 1000),
            "nonfused_throughput": batch * seq / (nf_med / 1000),
            "fused_peak_mem_mb": f["peak_mb"], "nonfused_peak_mem_mb": nf["peak_mb"],
            "max_abs_diff": max_diff, "cosine_sim": cos, "kl_divergence": kl,
        }
        rows.append(row)
        print(f"{batch:>5} {seq:>5} {f_med:>10.2f} {nf_med:>10.2f} {row['speedup']:>7.2f}x "
              f"{row['fused_p99_ms']:>10.2f} {row['nonfused_p99_ms']:>10.2f} "
              f"{f['peak_mb']:>10.0f} {nf['peak_mb']:>10.0f} "
              f"{max_diff:>10.4f} {cos:>9.6f} {kl:>9.6f}")

    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to: {args.out}")


if __name__ == "__main__":
    main()
