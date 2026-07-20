"""
Performance benchmarks for fused LayerNorm+Linear.

1. Single-operation benchmark: isolated LN+Linear vs fused for various configs
2. End-to-end benchmark: token generation throughput on OPT models
"""

import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.load_cuda import denominator_cuda
from src.weight_transform import compute_fused_weights
from src.fused_forward import (
    fused_ln_linear_forward,
    fused_ln_linear_forward_v1,
    fused_ln_linear_forward_v3,
)

# Transformer Engine is an optional dependency for baseline comparison
_HAS_TE = False
try:
    import ctypes
    import glob as _glob
    import site as _site
    # TE needs cuDNN/NCCL at runtime; pre-load all .so from pip-installed nvidia packages
    _sp = _site.getsitepackages()[0] if _site.getsitepackages() else ""
    for _pkg_dir in ["nvidia/cudnn/lib", "nvidia/nccl/lib"]:
        _lib_dir = os.path.join(_sp, _pkg_dir)
        if os.path.isdir(_lib_dir):
            for _so in sorted(_glob.glob(os.path.join(_lib_dir, "*.so.*"))):
                try:
                    ctypes.CDLL(_so, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
    import transformer_engine.pytorch as te
    _HAS_TE = True
except (ImportError, OSError):
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_hardware_info():
    """Gather hardware metadata for JSON output."""
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    gpu_mem_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else 0
    try:
        import subprocess
        cpu_info = subprocess.check_output("lscpu | head -20", shell=True, text=True)
        cpu_model = [l.split(":")[1].strip() for l in cpu_info.splitlines() if "Model name" in l]
        cpu_model = cpu_model[0] if cpu_model else platform.processor()
    except Exception:
        cpu_model = platform.processor()
    try:
        import subprocess
        mem_total = subprocess.check_output("free -g | awk '/Mem:/{print $2}'", shell=True, text=True).strip()
        host_ram_gb = int(mem_total)
    except Exception:
        host_ram_gb = 0
    return {
        "gpu_vendor": "NVIDIA",
        "gpu_model": gpu_name,
        "gpu_count": torch.cuda.device_count(),
        "gpu_memory_gb": gpu_mem_gb,
        "host_cpu": cpu_model,
        "host_ram_gb": host_ram_gb,
    }


def _get_software_info(variant="N/A"):
    """Gather software metadata for JSON output."""
    return {
        "os": f"{platform.system()} {platform.release()}",
        "driver": f"CUDA {torch.version.cuda}" if torch.version.cuda else "N/A",
        "framework": "PyTorch",
        "framework_version": torch.__version__,
        "runtime": "custom CUDA kernel",
        "runtime_version": variant,
    }


def _measure_per_iter(fn, warmup_iters=100, measure_iters=1000):
    """
    Measure per-iteration GPU time using pre-allocated CUDA events.

    Returns (mean_ms, stddev_ms, times_ms_list).
    """
    # Warmup
    for _ in range(warmup_iters):
        with torch.no_grad():
            fn()
    torch.cuda.synchronize()

    # Pre-allocate events
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(measure_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(measure_iters)]

    for i in range(measure_iters):
        start_events[i].record()
        with torch.no_grad():
            fn()
        end_events[i].record()

    torch.cuda.synchronize()
    times = [start_events[i].elapsed_time(end_events[i]) for i in range(measure_iters)]

    mean_ms = sum(times) / len(times)
    variance = sum((t - mean_ms) ** 2 for t in times) / len(times)
    stddev_ms = math.sqrt(variance)
    return mean_ms, stddev_ms, times


def _save_json_results(results_data, prefix="benchmark"):
    """Write structured JSON results to results/ directory."""
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{prefix}_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)
    with open(filepath, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to {filepath}")
    return filepath


def load_results(path):
    """Load benchmark results from a JSON file."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Single-operation benchmark
# ---------------------------------------------------------------------------

def benchmark_single_op():
    """Benchmark individual LN+Linear vs all fused variants for various dimension configs."""
    print("=" * 120)
    print("SINGLE-OPERATION BENCHMARK: LayerNorm+Linear vs Fused Variants")
    print("=" * 120)

    configs = [
        # (label, h, out_dim, batch_sizes)
        ("OPT-125m attn",  768,   768,  [1, 32, 128, 512]),
        ("OPT-125m FFN",   768,  3072,  [1, 32, 128, 512]),
        ("OPT-1.3b attn", 2048,  2048,  [1, 32, 128, 512]),
        ("OPT-1.3b FFN",  2048,  8192,  [1, 32, 128, 512]),
        ("OPT-6.7b attn", 4096,  4096,  [1, 32, 128, 512]),
        ("OPT-6.7b FFN",  4096, 16384,  [1, 32, 128, 512]),
    ]

    warmup_iters = 100
    measure_iters = 1000
    denom_stream = torch.cuda.Stream()
    # Pre-allocate events for V0
    input_ready_evt = torch.cuda.Event(enable_timing=False)
    denom_done_evt = torch.cuda.Event(enable_timing=False)

    te_header = f" {'TE-Baseline':>14}" if _HAS_TE else ""
    print(f"\n{'Config':<18} {'h':>5} {'out':>6} {'batch':>6} | "
          f"{'Original':>14} {'V0-Stream':>14} {'V1-Fused':>14} {'V2-Welford':>14} {'V3-Combined':>14}{te_header}")
    print("-" * (120 + (15 if _HAS_TE else 0)))

    results = []
    json_entries = []

    for label, h, out_dim, batch_sizes in configs:
        ln = nn.LayerNorm(h).cuda()
        linear = nn.Linear(h, out_dim).cuda()
        nn.init.normal_(ln.weight, mean=1.0, std=0.1)
        nn.init.normal_(ln.bias, mean=0.0, std=0.01)

        W_new, b_new, h_dim, eps = compute_fused_weights(ln, linear)

        for batch in batch_sizes:
            x = torch.randn(batch, h, device="cuda")

            # --- Measure original ---
            orig_mean, orig_std, _ = _measure_per_iter(
                lambda: linear(ln(x)), warmup_iters, measure_iters)

            # --- Measure V0 (stream-based) ---
            v0_mean, v0_std, _ = _measure_per_iter(
                lambda: fused_ln_linear_forward(x, W_new, b_new, denom_stream, h_dim, eps, input_ready_evt, denom_done_evt),
                warmup_iters, measure_iters)

            # --- Measure V1 (fused normalize, no streams) ---
            v1_mean, v1_std, _ = _measure_per_iter(
                lambda: fused_ln_linear_forward_v1(x, W_new, b_new, h_dim, eps),
                warmup_iters, measure_iters)

            # --- Measure V2 (Welford denom, stream-based forward) ---
            def fused_v2():
                x_2d = x.reshape(-1, x.size(-1))
                default_stream = torch.cuda.current_stream()
                input_ready_evt.record(default_stream)
                denom_stream.wait_event(input_ready_evt)
                with torch.cuda.stream(denom_stream):
                    v = denominator_cuda.compute_denominator_welford(x_2d)
                raw_output = F.linear(x_2d, W_new)
                denom_done_evt.record(denom_stream)
                default_stream.wait_event(denom_done_evt)
                std = torch.sqrt(v * v / h_dim + eps)
                return raw_output / std.unsqueeze(-1) + b_new

            v2_mean, v2_std, _ = _measure_per_iter(fused_v2, warmup_iters, measure_iters)

            # --- Measure V3 (Welford + fused normalize + 512 threads) ---
            v3_mean, v3_std, _ = _measure_per_iter(
                lambda: fused_ln_linear_forward_v3(x, W_new, b_new, h_dim, eps),
                warmup_iters, measure_iters)

            # --- Measure TE (Transformer Engine LayerNormLinear baseline) ---
            te_mean, te_std, s_te = None, None, None
            if _HAS_TE:
                te_lnl = te.LayerNormLinear(h, out_dim, eps=eps).cuda()
                # Copy weights from our LN + Linear to match
                with torch.no_grad():
                    te_lnl.layer_norm_weight.copy_(ln.weight)
                    te_lnl.layer_norm_bias.copy_(ln.bias)
                    te_lnl.weight.copy_(linear.weight)
                    te_lnl.bias.copy_(linear.bias)
                # Verify correctness before timing
                with torch.no_grad():
                    te_out = te_lnl(x)
                    ref_out = linear(ln(x))
                    te_diff = (te_out - ref_out).abs().max().item()
                    if te_diff > 0.01:
                        print(f"  WARNING: TE output differs from reference by {te_diff:.2e}")
                te_mean, te_std, _ = _measure_per_iter(
                    lambda: te_lnl(x), warmup_iters, measure_iters)
                s_te = orig_mean / te_mean if te_mean > 0 else float("inf")
                del te_lnl

            s0 = orig_mean / v0_mean if v0_mean > 0 else float("inf")
            s1 = orig_mean / v1_mean if v1_mean > 0 else float("inf")
            s2 = orig_mean / v2_mean if v2_mean > 0 else float("inf")
            s3 = orig_mean / v3_mean if v3_mean > 0 else float("inf")

            result_entry = {
                "label": label, "h": h, "out_dim": out_dim, "batch": batch,
                "orig_ms": orig_mean, "orig_std": orig_std,
                "v0_ms": v0_mean, "v0_std": v0_std, "v0_speedup": s0,
                "v1_ms": v1_mean, "v1_std": v1_std, "v1_speedup": s1,
                "v2_ms": v2_mean, "v2_std": v2_std, "v2_speedup": s2,
                "v3_ms": v3_mean, "v3_std": v3_std, "v3_speedup": s3,
            }
            if _HAS_TE:
                result_entry.update({
                    "te_ms": te_mean, "te_std": te_std, "te_speedup": s_te,
                })
            results.append(result_entry)

            # JSON entry for each variant
            variant_list = [
                ("V0", v0_mean, v0_std, s0), ("V1", v1_mean, v1_std, s1),
                ("V2", v2_mean, v2_std, s2), ("V3", v3_mean, v3_std, s3),
            ]
            if _HAS_TE and te_mean is not None:
                variant_list.append(("TE", te_mean, te_std, s_te))
            for vname, vmean, vstd, vspeed in variant_list:
                json_entries.append({
                    "config": {
                        "benchmark_id": f"fused_ln_linear_single_op_{label.replace(' ', '_')}_{vname}_fp32_{batch}",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "hardware": _get_hardware_info(),
                        "software": _get_software_info(vname),
                        "model": {"name": label, "precision": "FP32"},
                        "workload": {
                            "batch_size": batch,
                            "dimensions": {"h": h, "out_dim": out_dim},
                        },
                    },
                    "metrics": {
                        "single_op": {
                            "mean_ms": round(vmean, 6),
                            "stddev_ms": round(vstd, 6),
                            "variant": vname,
                        },
                        "speedup": {
                            "vs_baseline": round(vspeed, 4),
                            "baseline_description": "PyTorch nn.LayerNorm + nn.Linear",
                            "baseline_mean_ms": round(orig_mean, 6),
                            "baseline_stddev_ms": round(orig_std, 6),
                        },
                    },
                })

            def _fmt(mean, std, speedup):
                return f"{mean:>7.4f}±{std:>5.4f}({speedup:.2f}x)"

            te_col = f" {_fmt(te_mean, te_std, s_te)}" if _HAS_TE and te_mean is not None else ""
            print(f"{label:<18} {h:>5} {out_dim:>6} {batch:>6} | "
                  f"{orig_mean:>7.4f}±{orig_std:<5.4f}ms "
                  f"{_fmt(v0_mean, v0_std, s0)} "
                  f"{_fmt(v1_mean, v1_std, s1)} "
                  f"{_fmt(v2_mean, v2_std, s2)} "
                  f"{_fmt(v3_mean, v3_std, s3)}{te_col}")

    print()

    # Save JSON
    _save_json_results(json_entries, prefix="single_op")

    return results


# ---------------------------------------------------------------------------
# End-to-end benchmark
# ---------------------------------------------------------------------------

def benchmark_end_to_end():
    """End-to-end token generation throughput on OPT models with multi-batch support."""
    print("=" * 110)
    print("END-TO-END BENCHMARK: OPT Model Token Generation")
    print("=" * 110)

    from transformers import AutoTokenizer, OPTForCausalLM
    from src.patch_model import patch_opt_model

    models_to_test = [
        ("facebook/opt-1.3b", "OPT-1.3b"),
        ("facebook/opt-6.7b", "OPT-6.7b"),
    ]

    prompts = [
        "The future of artificial intelligence is",
        "In a shocking finding, scientists discovered that",
        "The economic impact of climate change will",
        "Recent advances in quantum computing have shown",
        "The most important lesson from history is",
        "Space exploration in the next decade will focus on",
        "The relationship between technology and society has",
        "New research in neuroscience suggests that the brain",
        "The key to understanding complex systems lies in",
        "Advances in renewable energy technology have made",
        "The intersection of biology and computing creates",
        "Modern cryptography relies fundamentally on the",
        "The philosophical implications of consciousness are",
        "Deep ocean exploration has revealed unexpected",
        "The evolution of programming languages shows that",
        "Quantum entanglement challenges our understanding of",
        "The future of transportation will be shaped by",
        "Breakthroughs in materials science have enabled",
        "The role of microbiomes in human health is",
        "Artificial general intelligence remains a topic of",
        "Climate modeling has become increasingly accurate",
        "The history of mathematics reveals patterns in",
        "Neural network architectures continue to evolve",
        "The economics of space mining could transform",
        "Genetic engineering tools like CRISPR have opened",
        "The social impact of automation extends beyond",
        "Advances in battery technology are crucial for",
        "The study of ancient civilizations reveals that",
        "Protein folding prediction has been revolutionized",
        "The future of education will be transformed by",
        "Sustainable agriculture requires innovative approaches",
        "The development of fusion energy has reached",
    ]
    batch_sizes = [1, 2, 4, 8, 16, 32]
    variants = ["V0", "V1", "V3"]

    gen_kwargs = dict(
        max_new_tokens=128,
        do_sample=False,
        use_cache=True,
    )
    num_runs = 5
    json_entries = []

    for model_name, label in models_to_test:
        print(f"\n{'='*110}")
        print(f"  {label} ({model_name})")
        print(f"{'='*110}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # --- Benchmark original model ---
        print(f"\nLoading original {label}...")
        model_orig = OPTForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32
        ).cuda().eval()

        orig_results = {}
        for bs in batch_sizes:
            batch_prompts = (prompts * ((bs // len(prompts)) + 1))[:bs]
            inputs = tokenizer(
                batch_prompts, return_tensors="pt", padding=True
            ).to("cuda")

            print(f"  Warming up original model (batch_size={bs})...")
            with torch.no_grad():
                _ = model_orig.generate(**inputs, **gen_kwargs)
            torch.cuda.synchronize()

            print(f"  Benchmarking original model (batch_size={bs})...")
            times = []
            for _ in range(num_runs):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = model_orig.generate(**inputs, **gen_kwargs)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append(t1 - t0)

            num_new_tokens = out.shape[1] - inputs["input_ids"].shape[1]
            total_tokens = bs * num_new_tokens
            mean_time = sum(times) / len(times)
            stddev_time = math.sqrt(sum((t - mean_time) ** 2 for t in times) / len(times))

            # Save output text for batch_size=1 comparison
            orig_text = None
            if bs == 1:
                with torch.no_grad():
                    out_check = model_orig.generate(**inputs, **gen_kwargs)
                orig_text = tokenizer.decode(out_check[0], skip_special_tokens=True)

            orig_results[bs] = {
                "mean_time": mean_time,
                "stddev_time": stddev_time,
                "num_new_tokens": num_new_tokens,
                "total_tokens": total_tokens,
                "tps": total_tokens / mean_time,
                "text": orig_text,
            }

        # Free original model before loading fused variants
        del model_orig
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        # --- Benchmark each fused variant ---
        all_variant_results = {}
        for variant in variants:
            print(f"\nLoading fused {label} (variant={variant})...")
            model_fused = OPTForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float32
            ).cuda().eval()
            print(f"  Patching model with variant={variant}...")
            patch_opt_model(model_fused, variant=variant)

            variant_results = {}
            for bs in batch_sizes:
                batch_prompts = (prompts * ((bs // len(prompts)) + 1))[:bs]
                inputs = tokenizer(
                    batch_prompts, return_tensors="pt", padding=True
                ).to("cuda")

                print(f"  Warming up {variant} model (batch_size={bs})...")
                with torch.no_grad():
                    _ = model_fused.generate(**inputs, **gen_kwargs)
                torch.cuda.synchronize()

                print(f"  Benchmarking {variant} model (batch_size={bs})...")
                times = []
                for _ in range(num_runs):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        out = model_fused.generate(**inputs, **gen_kwargs)
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    times.append(t1 - t0)

                num_new_tokens = out.shape[1] - inputs["input_ids"].shape[1]
                total_tokens = bs * num_new_tokens
                mean_time = sum(times) / len(times)
                stddev_time = math.sqrt(sum((t - mean_time) ** 2 for t in times) / len(times))

                fused_text = None
                if bs == 1:
                    with torch.no_grad():
                        out_check = model_fused.generate(**inputs, **gen_kwargs)
                    fused_text = tokenizer.decode(out_check[0], skip_special_tokens=True)

                variant_results[bs] = {
                    "mean_time": mean_time,
                    "stddev_time": stddev_time,
                    "num_new_tokens": num_new_tokens,
                    "total_tokens": total_tokens,
                    "tps": total_tokens / mean_time,
                    "text": fused_text,
                }

                # JSON entry
                o = orig_results[bs]
                json_entries.append({
                    "config": {
                        "benchmark_id": f"fused_ln_linear_{label.replace(' ', '_')}_{variant}_fp32_{bs}",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "hardware": _get_hardware_info(),
                        "software": _get_software_info(variant),
                        "model": {
                            "name": model_name,
                            "precision": "FP32",
                            "max_context": 2048,
                        },
                        "workload": {
                            "batch_size": bs,
                            "input_tokens": inputs["input_ids"].shape[1],
                            "output_tokens": num_new_tokens,
                            "sampling": {"temperature": 0, "top_p": 1.0},
                        },
                    },
                    "metrics": {
                        "throughput": {
                            "tokens_per_second": round(variant_results[bs]["tps"], 1),
                        },
                        "latency": {
                            "mean_ms": round(mean_time * 1000, 1),
                            "stddev_ms": round(stddev_time * 1000, 1),
                        },
                        "speedup": {
                            "vs_baseline": round(variant_results[bs]["tps"] / o["tps"], 4),
                            "baseline_description": "PyTorch nn.LayerNorm + nn.Linear",
                        },
                    },
                })

            all_variant_results[variant] = variant_results
            del model_fused
            gc.collect()
            torch.cuda.empty_cache()

        # --- Report ---
        print(f"\n{'─'*120}")
        print(f"  Results for {label}")
        print(f"{'─'*120}")

        # Header
        header = f"  {'Batch':>5} {'Tokens':>7} | {'Original':>16} {'tok/s':>8}"
        for v in variants:
            header += f" | {v:>16} {'tok/s':>8} {'Speedup':>7}"
        print(header)
        print(f"  {'-'*115}")

        for bs in batch_sizes:
            o = orig_results[bs]
            line = (f"  {bs:>5} {o['total_tokens']:>7} | "
                    f"{o['mean_time']*1000:>8.1f}±{o['stddev_time']*1000:>5.1f}ms {o['tps']:>7.1f}")
            for v in variants:
                vr = all_variant_results[v][bs]
                speedup = vr["tps"] / o["tps"]
                line += (f" | {vr['mean_time']*1000:>8.1f}±{vr['stddev_time']*1000:>4.1f}ms "
                         f"{vr['tps']:>7.1f} {speedup:>6.3f}x")
            print(line)

        # Output comparison for batch_size=1
        o_text = orig_results[1]["text"]
        for v in variants:
            f_text = all_variant_results[v][1].get("text")
            if o_text and f_text:
                match = "YES" if o_text == f_text else "NO (expected - floating point differences)"
                print(f"\n  {v} output match (batch_size=1): {match}")

    # Save JSON
    _save_json_results(json_entries, prefix="e2e")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-op", action="store_true", help="Run single-operation benchmark")
    parser.add_argument("--e2e", action="store_true", help="Run end-to-end benchmark")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    args = parser.parse_args()

    if not any([args.single_op, args.e2e, args.all]):
        args.all = True

    if args.single_op or args.all:
        benchmark_single_op()

    if args.e2e or args.all:
        benchmark_end_to_end()
