"""
Micro-benchmark: GPT-OSS attention input path, fused vs baseline (HF/PyTorch).

Measures exactly the operation the kernel fusion replaces, in isolation:
    baseline (non-fused): normed = RMSNorm(x)  (FP32 internally, as in GptOssRMSNorm)
                          q,k,v = q_proj(normed), k_proj(normed), v_proj(normed)
    fused:                q,k,v = fused_q(x), fused_k(x), fused_v(x)
                          (gamma absorbed into weights; rms division + bias fused
                           into the kernel epilogue; no separate norm op)

One kernel variant per run (V1: 256 threads, V3: 512 threads — the RMSNorm
path has only these two; V0/V2 are LayerNorm-only). Output CSV uses the same
schema as Itamar's benchmark_rmsnorm_linear_fusion.py: batch, seq_len, hidden,
out_dim, fused/nonfused median & p99 latency, throughput, peak memory,
max_abs_diff, cosine_sim, kl_divergence.

Loads ONLY the layer-0 tensors it needs straight from safetensors — no full
model load, starts in seconds.

Usage:
    python benchmark_attn_fusion_micro.py --model-dir ~/bench/models/non-fused --variant V1
    python benchmark_attn_fusion_micro.py --model-dir ~/bench/models/non-fused --variant V3
"""

import argparse
import csv
import json
import os
import statistics
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

SHAPES = [  # (batch, seq)
    (1, 128), (1, 512), (1, 2048),
    (8, 128), (8, 512), (8, 2048),
    (32, 128), (32, 512), (32, 2048),
]
WARMUP = 50
MEASURE = 200
NUMERICAL_ITERS = 10
DEVICE = "cuda"


def load_tensor(model_dir: str, name: str) -> torch.Tensor:
    from safetensors import safe_open
    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        shard = json.load(open(idx_path))["weight_map"][name]
    else:
        shard = "model.safetensors"
    with safe_open(os.path.join(model_dir, shard), framework="pt") as f:
        return f.get_tensor(name)


class Baseline(nn.Module):
    """RMSNorm (computed once, FP32 internally — same as GptOssRMSNorm) -> q/k/v."""

    def __init__(self, gamma, wq, bq, wk, bk, wv, bv, eps):
        super().__init__()
        self.register_buffer("gamma", gamma)
        self.register_buffer("wq", wq); self.register_buffer("bq", bq)
        self.register_buffer("wk", wk); self.register_buffer("bk", bk)
        self.register_buffer("wv", wv); self.register_buffer("bv", bv)
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x32 = x.float()
        normed = (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps))
        normed = (self.gamma.float() * normed).to(dtype)
        return (F.linear(normed, self.wq, self.bq),
                F.linear(normed, self.wk, self.bk),
                F.linear(normed, self.wv, self.bv))


class Fused(nn.Module):
    def __init__(self, fq, fk, fv):
        super().__init__()
        self.fq, self.fk, self.fv = fq, fk, fv

    def forward(self, x):
        return self.fq(x), self.fk(x), self.fv(x)


def time_module(mod, x, warmup=WARMUP, measure=MEASURE):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    lat = []
    with torch.no_grad():
        for _ in range(warmup):
            mod(x)
        torch.cuda.synchronize()
        for _ in range(measure):
            start.record()
            mod(x)
            end.record()
            torch.cuda.synchronize()
            lat.append(start.elapsed_time(end))
    return lat


def peak_memory_mb(mod, x) -> float:
    torch.cuda.reset_peak_memory_stats(DEVICE)
    with torch.no_grad():
        mod(x)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(DEVICE) / 1024 ** 2


def percentile(data, pct):
    s = sorted(data)
    return s[min(int(len(s) * pct / 100), len(s) - 1)]


def numerical_equivalence(fused, baseline, x, n_iters=NUMERICAL_ITERS):
    """Same metrics/formulas as Itamar's script, applied per projection output."""
    max_diffs, cosines, kls = [], [], []
    with torch.no_grad():
        for _ in range(n_iters):
            outs_f = fused(x)
            outs_b = baseline(x)
            for out_f, out_nf in zip(outs_f, outs_b):
                out_f, out_nf = out_f.float(), out_nf.float()
                max_diffs.append((out_f - out_nf).abs().max().item())
                f_flat = out_f.reshape(out_f.size(0), -1)
                nf_flat = out_nf.reshape(out_nf.size(0), -1)
                cosines.append(F.cosine_similarity(f_flat, nf_flat, dim=1).mean().item())
                p = F.softmax(out_f, dim=-1).clamp(min=1e-10)
                q = F.softmax(out_nf, dim=-1).clamp(min=1e-10)
                kls.append((p * (p / q).log()).sum(dim=-1).mean().item())
    return max(max_diffs), statistics.mean(cosines), statistics.mean(kls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True,
                    help="HF checkpoint dir (use non-fused; gamma absorbed on the fly)")
    ap.add_argument("--fusion-repo", default="h100-fused-layernorm-linear-main")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--variant", default="V1", choices=["V1", "V3"],
                    help="Kernel variant for this run (RMSNorm path has only V1/V3)")
    ap.add_argument("--out", default=None,
                    help="Output CSV path (default: attn_fusion_micro_<variant>.csv)")
    args = ap.parse_args()
    out_path = args.out or f"attn_fusion_micro_{args.variant}.csv"

    sys.path.insert(0, os.path.abspath(args.fusion_repo))
    from src.fused_forward import FusedRMSNormLinearV1, FusedRMSNormLinearV3
    cls = {"V1": FusedRMSNormLinearV1, "V3": FusedRMSNormLinearV3}[args.variant]

    cfg = json.load(open(os.path.join(args.model_dir, "config.json")))
    h = cfg["hidden_size"]
    eps = cfg.get("rms_norm_eps", 1e-5)
    L = args.layer
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}  CUDA: {torch.version.cuda}")
    print(f"hidden={h}  eps={eps}  layer={L}  variant={args.variant}")
    print(f"Warmup iters: {WARMUP}  |  Measure iters: {MEASURE}")

    pre = f"model.layers.{L}"
    gamma = load_tensor(args.model_dir, f"{pre}.input_layernorm.weight").to(DEVICE)
    proj = {}
    for n in ("q_proj", "k_proj", "v_proj"):
        w = load_tensor(args.model_dir, f"{pre}.self_attn.{n}.weight").to(DEVICE)
        try:
            b = load_tensor(args.model_dir, f"{pre}.self_attn.{n}.bias").to(DEVICE)
        except KeyError:
            b = torch.zeros(w.shape[0], dtype=w.dtype, device=DEVICE)
        proj[n] = (w, b)
    dtype = proj["q_proj"][0].dtype
    out_dim = sum(proj[n][0].shape[0] for n in ("q_proj", "k_proj", "v_proj"))
    print(f"dtype={dtype}  q={tuple(proj['q_proj'][0].shape)} "
          f"k={tuple(proj['k_proj'][0].shape)} v={tuple(proj['v_proj'][0].shape)}  "
          f"out_dim(total)={out_dim}")

    baseline = Baseline(gamma,
                        *proj["q_proj"], *proj["k_proj"], *proj["v_proj"],
                        eps=eps).to(DEVICE).eval()

    mods = []
    for n in ("q_proj", "k_proj", "v_proj"):
        w, b = proj[n]
        w_new = (w.float() * gamma.float().unsqueeze(0)).to(dtype).contiguous()
        mods.append(cls(w_new, b.clone().contiguous(), h, eps))
    fused = Fused(*mods).to(DEVICE).eval()

    header = (f"{'batch':>5} {'seq':>5} {'hidden':>7} "
              f"{'fused_med':>10} {'nf_med':>10} {'speedup':>8} "
              f"{'fused_p99':>10} {'nf_p99':>10} "
              f"{'fused_mem':>10} {'nf_mem':>10} "
              f"{'cos_sim':>9} {'kl_div':>9}")
    print(f"\n{header}\n{'-' * len(header)}")

    rows = []
    for (batch, seq) in SHAPES:
        x = torch.randn(batch, seq, h, device=DEVICE, dtype=dtype)

        nf_lat = time_module(baseline, x)
        f_lat = time_module(fused, x)
        nf_mem = peak_memory_mb(baseline, x)
        f_mem = peak_memory_mb(fused, x)
        max_diff, cos, kl = numerical_equivalence(fused, baseline, x)

        f_med, nf_med = statistics.median(f_lat), statistics.median(nf_lat)
        row = {
            "batch": batch, "seq_len": seq, "hidden": h, "out_dim": out_dim,
            "fused_median_ms": f_med,
            "nonfused_median_ms": nf_med,
            "speedup": nf_med / f_med,
            "fused_p99_ms": percentile(f_lat, 99),
            "nonfused_p99_ms": percentile(nf_lat, 99),
            "fused_throughput": batch * seq / (f_med / 1000.0),
            "nonfused_throughput": batch * seq / (nf_med / 1000.0),
            "fused_peak_mem_mb": f_mem,
            "nonfused_peak_mem_mb": nf_mem,
            "max_abs_diff": max_diff,
            "cosine_sim": cos,
            "kl_divergence": kl,
        }
        rows.append(row)
        print(f"{batch:>5} {seq:>5} {h:>7} "
              f"{f_med:>10.4f} {nf_med:>10.4f} {row['speedup']:>7.2f}x "
              f"{row['fused_p99_ms']:>10.4f} {row['nonfused_p99_ms']:>10.4f} "
              f"{f_mem:>10.1f} {nf_mem:>10.1f} "
              f"{cos:>9.6f} {kl:>9.6f}")

        del x
        torch.cuda.empty_cache()

    fields = [
        "batch", "seq_len", "hidden", "out_dim",
        "fused_median_ms", "nonfused_median_ms", "speedup",
        "fused_p99_ms", "nonfused_p99_ms",
        "fused_throughput", "nonfused_throughput",
        "fused_peak_mem_mb", "nonfused_peak_mem_mb",
        "max_abs_diff", "cosine_sim", "kl_divergence",
    ]
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
