"""
Benchmark: RMSNorm + Linear Fusion vs Non-Fused (NVFP4 Quantized)
PyTorch + CUDA

Usage:
    python benchmark_rmsnorm_linear_fusion.py --dir /path/to/dir [--layer-path model.layers.0.mlp]

    Expects the following layout under --dir:
        <dir>/models/fused/       — fused HuggingFace checkpoint
        <dir>/models/non-fused/   — non-fused HuggingFace checkpoint

    Both checkpoints are loaded via AutoModelForCausalLM.from_pretrained().
    The script then extracts a specific sub-module (the fused/non-fused layer)
    from each full model using --layer-path, and benchmarks that module in
    isolation across the shape sweep.

    --layer-path  dot-separated attribute path into the model, e.g.:
                    model.layers.0.mlp          (LLaMA-style MLP)
                    model.layers.0.self_attn    (attention block)
                  Use --print-keys to discover the right path for your model.

Requirements:
    pip install torch transformers  # CUDA build
    pip install safetensors         # if checkpoints use .safetensors format
"""

import argparse
import gc
import os
import sys
import time
import statistics
from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEVICE = "cuda"
DTYPE  = torch.float16   # activation dtype going into the layers

# Shape sweep: (batch_size, seq_len, hidden_dim, out_dim)
SHAPE_SWEEP: List[Tuple[int, int, int, int]] = [
    (1,   128,  4096, 4096),
    (1,   512,  4096, 4096),
    (1,  2048,  4096, 4096),
    (8,   128,  4096, 4096),
    (8,   512,  4096, 4096),
    (8,  2048,  4096, 4096),
    (32,  128,  4096, 4096),
    (32,  512,  4096, 4096),
    (32, 2048,  4096, 4096),
]

WARMUP_ITERS   = 50
MEASURE_ITERS  = 200
NUMERICAL_ITERS = 10   # iterations for output collection in equivalence check


# ---------------------------------------------------------------------------
# Model loading (HuggingFace full checkpoint)
# ---------------------------------------------------------------------------

def _get_nested_attr(obj, dot_path: str):
    """
    Navigate a dot-separated attribute path, supporting integer indices.
    e.g. 'model.layers.0.mlp' → obj.model.layers[0].mlp
    """
    for part in dot_path.split("."):
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def _infer_hidden_dim(module: nn.Module) -> int:
    """
    Infer hidden_dim from the first 2-D weight in the module that looks like
    an input projection (i.e. weight.shape[-1] is the hidden dim).
    """
    for name, param in module.named_parameters():
        if param.ndim == 2 and "norm" not in name:
            hidden_dim = param.shape[-1]
            print(f"  Inferred hidden_dim={hidden_dim} from '{name}' {tuple(param.shape)}")
            return hidden_dim
    raise ValueError(
        "Could not infer hidden_dim from module parameters. "
        "Check --layer-path points to the right sub-module."
    )


def _load_hf_model(model_dir: str, label: str) -> nn.Module:
    """Load a full HuggingFace model in BF16 onto CPU, then return it."""
    from transformers import AutoModelForCausalLM

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    print(f"\nLoading {label} from {model_dir} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,   # match DTYPE; cast again after extraction
        device_map="cpu",            # load to CPU first to avoid OOM with two models
        trust_remote_code=True,
    )
    model.eval()
    return model


def print_model_keys(model_dir: str) -> None:
    """
    Helper: load a model and print its top-level named modules so the user
    can identify the correct --layer-path.  Invoked via --print-keys.
    """
    model = _load_hf_model(model_dir, "key inspection")
    print("\nTop-level named modules (first 60):")
    for i, (name, _) in enumerate(model.named_modules()):
        if i >= 60:
            print("  ... (truncated, pass a more specific prefix to narrow down)")
            break
        print(f"  {name}")


def load_models(base_dir: str, layer_path: str) -> Tuple[nn.Module, nn.Module, int]:
    """
    Load fused and non-fused HuggingFace checkpoints from:
        <base_dir>/models/fused/
        <base_dir>/models/non-fused/

    Extracts the sub-module at `layer_path` from each full model.
    Returns (fused_layer, nonfused_layer, hidden_dim).

    The two full models are deleted from memory after extraction so only
    the extracted layers remain on GPU during benchmarking.
    """
    fused_dir    = os.path.join(base_dir, "models", "fused")
    nonfused_dir = os.path.join(base_dir, "models", "non-fused")

    # --- Non-fused ---
    nonfused_full  = _load_hf_model(nonfused_dir, "non-fused")
    nonfused_layer = _get_nested_attr(nonfused_full, layer_path)
    hidden_dim     = _infer_hidden_dim(nonfused_layer)
    # Detach the sub-module from the parent so the rest of the model can be freed
    nonfused_layer = nonfused_layer.cpu()
    del nonfused_full
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Non-fused layer extracted: {type(nonfused_layer).__name__}")

    # --- Fused ---
    fused_full  = _load_hf_model(fused_dir, "fused")
    fused_layer = _get_nested_attr(fused_full, layer_path)
    fused_layer = fused_layer.cpu()
    del fused_full
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Fused layer extracted:     {type(fused_layer).__name__}")

    return fused_layer, nonfused_layer, hidden_dim


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

@dataclass
class ShapeResult:
    batch:   int
    seq_len: int
    hidden:  int
    out_dim: int

    # Latency (ms)
    fused_latencies:    List[float] = field(default_factory=list)
    nonfused_latencies: List[float] = field(default_factory=list)

    # Memory (MB)
    fused_peak_mem_mb:    float = 0.0
    nonfused_peak_mem_mb: float = 0.0

    # Numerical equivalence
    max_abs_diff:   float = 0.0
    cosine_sim:     float = 0.0
    kl_divergence:  float = 0.0

    @property
    def fused_median_ms(self) -> float:
        return statistics.median(self.fused_latencies) if self.fused_latencies else float("nan")

    @property
    def nonfused_median_ms(self) -> float:
        return statistics.median(self.nonfused_latencies) if self.nonfused_latencies else float("nan")

    @property
    def fused_p99_ms(self) -> float:
        return _percentile(self.fused_latencies, 99)

    @property
    def nonfused_p99_ms(self) -> float:
        return _percentile(self.nonfused_latencies, 99)

    @property
    def speedup(self) -> float:
        if self.fused_median_ms == 0:
            return float("nan")
        return self.nonfused_median_ms / self.fused_median_ms

    @property
    def fused_throughput(self) -> float:
        """tokens / second"""
        tokens = self.batch * self.seq_len
        return tokens / (self.fused_median_ms / 1000.0)

    @property
    def nonfused_throughput(self) -> float:
        tokens = self.batch * self.seq_len
        return tokens / (self.nonfused_median_ms / 1000.0)


def _percentile(data: List[float], pct: int) -> float:
    if not data:
        return float("nan")
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * pct / 100)
    idx = min(idx, len(sorted_data) - 1)
    return sorted_data[idx]


def _sync_time(fn, *args) -> float:
    """Run fn(*args), synchronize CUDA, return wall-clock seconds."""
    start = time.perf_counter()
    fn(*args)
    torch.cuda.synchronize()
    return time.perf_counter() - start


def measure_latency(
    model: nn.Module,
    x: torch.Tensor,
    warmup: int = WARMUP_ITERS,
    measure: int = MEASURE_ITERS,
) -> List[float]:
    """Returns list of per-forward-pass latencies in milliseconds."""
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        torch.cuda.synchronize()

        latencies = []
        for _ in range(measure):
            t = _sync_time(model, x)
            latencies.append(t * 1000.0)
    return latencies


def measure_peak_memory(model: nn.Module, x: torch.Tensor) -> float:
    """Returns peak GPU memory allocated during a forward pass, in MB."""
    model.eval()
    torch.cuda.reset_peak_memory_stats(DEVICE)
    with torch.no_grad():
        _ = model(x)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(DEVICE) / 1024 ** 2


def measure_numerical_equivalence(
    fused: nn.Module,
    nonfused: nn.Module,
    x: torch.Tensor,
    n_iters: int = NUMERICAL_ITERS,
) -> Tuple[float, float, float]:
    """
    Compare fused vs non-fused outputs.

    Returns:
        max_abs_diff  — max absolute elementwise difference
        cosine_sim    — mean cosine similarity across batch
        kl_divergence — mean KL divergence of softmax distributions
    """
    fused.eval()
    nonfused.eval()

    max_diffs, cosines, kls = [], [], []

    with torch.no_grad():
        for _ in range(n_iters):
            out_f  = fused(x)
            out_nf = nonfused(x)
            # MoE MLP (e.g. GPT-OSS) returns (output, router_scores); compare the output
            if isinstance(out_f, tuple):
                out_f = out_f[0]
            if isinstance(out_nf, tuple):
                out_nf = out_nf[0]
            out_f  = out_f.float()
            out_nf = out_nf.float()

            # Max absolute difference
            max_diffs.append((out_f - out_nf).abs().max().item())

            # Cosine similarity — flatten seq dim, compute per batch item
            f_flat  = out_f.view(out_f.size(0), -1)
            nf_flat = out_nf.view(out_nf.size(0), -1)
            cos = F.cosine_similarity(f_flat, nf_flat, dim=1).mean().item()
            cosines.append(cos)

            # KL divergence of softmax distributions
            p = F.softmax(out_f,  dim=-1).clamp(min=1e-10)
            q = F.softmax(out_nf, dim=-1).clamp(min=1e-10)
            kl = (p * (p / q).log()).sum(dim=-1).mean().item()
            kls.append(kl)

    return (
        statistics.mean(max_diffs),
        statistics.mean(cosines),
        statistics.mean(kls),
    )


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark(base_dir: str, layer_path: str) -> List[ShapeResult]:
    results = []

    # Load and extract layers; discard full models immediately to free memory
    fused_model, nonfused_model, hidden_dim = load_models(base_dir, layer_path)
    fused_model    = fused_model.to(DEVICE, dtype=DTYPE).eval()
    nonfused_model = nonfused_model.to(DEVICE, dtype=DTYPE).eval()

    # Use hidden_dim from actual weights; out_dim from SHAPE_SWEEP is ignored
    # (the layer's own forward() determines output shape)
    shape_sweep = [
        (batch, seq_len, hidden_dim)
        for (batch, seq_len, _, _) in SHAPE_SWEEP
    ]

    for (batch, seq_len, hidden) in shape_sweep:
        out_dim = hidden   # placeholder for ShapeResult; real out shape is layer-defined
        print(f"\n{'='*60}")
        print(f"Shape: batch={batch}, seq={seq_len}, hidden={hidden}, out={out_dim}")
        print(f"{'='*60}")

        fused    = fused_model
        nonfused = nonfused_model

        # Input
        x = torch.randn(batch, seq_len, hidden, device=DEVICE, dtype=DTYPE)

        result = ShapeResult(batch=batch, seq_len=seq_len, hidden=hidden, out_dim=out_dim)

        # --- Latency ---
        print("  Measuring latency (non-fused)...")
        result.nonfused_latencies = measure_latency(nonfused, x)

        print("  Measuring latency (fused)...")
        result.fused_latencies = measure_latency(fused, x)

        # --- Peak memory ---
        print("  Measuring peak memory...")
        result.nonfused_peak_mem_mb = measure_peak_memory(nonfused, x)
        result.fused_peak_mem_mb    = measure_peak_memory(fused, x)

        # --- Numerical equivalence ---
        print("  Measuring numerical equivalence...")
        (result.max_abs_diff,
         result.cosine_sim,
         result.kl_divergence) = measure_numerical_equivalence(fused, nonfused, x)

        # Print summary for this shape
        print(f"\n  Latency (median ms):  fused={result.fused_median_ms:.3f}  non-fused={result.nonfused_median_ms:.3f}  speedup={result.speedup:.2f}x")
        print(f"  Latency (p99 ms):     fused={result.fused_p99_ms:.3f}  non-fused={result.nonfused_p99_ms:.3f}")
        print(f"  Throughput (tok/s):   fused={result.fused_throughput:,.0f}  non-fused={result.nonfused_throughput:,.0f}")
        print(f"  Peak mem (MB):        fused={result.fused_peak_mem_mb:.1f}  non-fused={result.nonfused_peak_mem_mb:.1f}")
        print(f"  Numerical equivalence:")
        print(f"    max |diff|  = {result.max_abs_diff:.6f}")
        print(f"    cosine sim  = {result.cosine_sim:.6f}  (1.0 = identical)")
        print(f"    KL div      = {result.kl_divergence:.6f}  (0.0 = identical distributions)")

        del x
        gc.collect()
        torch.cuda.empty_cache()

        results.append(result)

    return results


def print_summary_table(results: List[ShapeResult]):
    header = (
        f"{'batch':>5} {'seq':>5} {'hidden':>7} "
        f"{'fused_med':>10} {'nf_med':>10} {'speedup':>8} "
        f"{'fused_p99':>10} {'nf_p99':>10} "
        f"{'fused_mem':>10} {'nf_mem':>10} "
        f"{'cos_sim':>9} {'kl_div':>9}"
    )
    print(f"\n{'='*len(header)}")
    print("SUMMARY TABLE")
    print(f"{'='*len(header)}")
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.batch:>5} {r.seq_len:>5} {r.hidden:>7} "
            f"{r.fused_median_ms:>10.3f} {r.nonfused_median_ms:>10.3f} {r.speedup:>8.2f}x "
            f"{r.fused_p99_ms:>10.3f} {r.nonfused_p99_ms:>10.3f} "
            f"{r.fused_peak_mem_mb:>10.1f} {r.nonfused_peak_mem_mb:>10.1f} "
            f"{r.cosine_sim:>9.6f} {r.kl_divergence:>9.6f}"
        )
    print(f"{'='*len(header)}")


def save_csv(results: List[ShapeResult], path: str = "benchmark_results.csv"):
    import csv
    fields = [
        "batch", "seq_len", "hidden", "out_dim",
        "fused_median_ms", "nonfused_median_ms", "speedup",
        "fused_p99_ms", "nonfused_p99_ms",
        "fused_throughput", "nonfused_throughput",
        "fused_peak_mem_mb", "nonfused_peak_mem_mb",
        "max_abs_diff", "cosine_sim", "kl_divergence",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "batch": r.batch, "seq_len": r.seq_len,
                "hidden": r.hidden, "out_dim": r.out_dim,
                "fused_median_ms": r.fused_median_ms,
                "nonfused_median_ms": r.nonfused_median_ms,
                "speedup": r.speedup,
                "fused_p99_ms": r.fused_p99_ms,
                "nonfused_p99_ms": r.nonfused_p99_ms,
                "fused_throughput": r.fused_throughput,
                "nonfused_throughput": r.nonfused_throughput,
                "fused_peak_mem_mb": r.fused_peak_mem_mb,
                "nonfused_peak_mem_mb": r.nonfused_peak_mem_mb,
                "max_abs_diff": r.max_abs_diff,
                "cosine_sim": r.cosine_sim,
                "kl_divergence": r.kl_divergence,
            })
    print(f"\nResults saved to: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark RMSNorm+Linear fusion vs non-fused (HuggingFace checkpoints)."
    )
    parser.add_argument(
        "--dir", required=True,
        help="Base directory containing models/fused/ and models/non-fused/ subdirectories."
    )
    parser.add_argument(
        "--layer-path", default="model.layers.0.mlp",
        help=(
            "Dot-separated path to the sub-module to extract and benchmark. "
            "Integer parts are treated as list indices. "
            "Examples: 'model.layers.0.mlp'  'model.layers.0.self_attn'  "
            "(default: model.layers.0.mlp)"
        ),
    )
    parser.add_argument(
        "--print-keys", action="store_true",
        help=(
            "Print all named modules from the non-fused model and exit. "
            "Use this to discover the correct --layer-path for your architecture."
        ),
    )
    args = parser.parse_args()

    if args.print_keys:
        nonfused_dir = os.path.join(args.dir, "models", "non-fused")
        print_model_keys(nonfused_dir)
        sys.exit(0)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this benchmark.")

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Base dir:   {args.dir}")
    print(f"Layer path: {args.layer_path}")
    print(f"Warmup iters: {WARMUP_ITERS}  |  Measure iters: {MEASURE_ITERS}")

    results = run_benchmark(args.dir, args.layer_path)
    print_summary_table(results)
    save_csv(results)