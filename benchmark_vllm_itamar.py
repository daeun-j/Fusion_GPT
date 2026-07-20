"""
vLLM serving benchmark for fused vs non-fused checkpoints, reported in
Itamar's CSV schema and shape sweep.

Because one GPU cannot host both 20B models simultaneously, this runs in two
steps:

  1) capture — against a RUNNING vllm server for ONE checkpoint, sweep
     Itamar's shapes (batch 1/8/32 x seq 128/512/2048). For each shape it
     sends `batch` fixed-seed token-id prompts of length `seq` in a single
     /v1/completions request (vLLM batches them), with echo+logprobs, and
     records wall-clock latency over N iterations plus prompt logprobs.

        # serve non-fused first
        python benchmark_vllm_itamar.py capture --url http://localhost:8000 \
            --model /root/bench/models/non-fused --out vllm_nonfused.json
        # restart server with fused, then
        python benchmark_vllm_itamar.py capture --url http://localhost:8000 \
            --model /root/bench/models/fused --out vllm_fused.json

  2) compare — merge the two captures into one Itamar-format CSV:

        python benchmark_vllm_itamar.py compare vllm_fused.json vllm_nonfused.json \
            --out vllm_itamar_results.csv

Metric notes (serving adaptation of Itamar's kernel-level metrics — label
this clearly when reporting):
  - latency/throughput: end-to-end server latency for the batched request
    (prefill-dominated; max_tokens=1).
  - max_abs_diff / cosine_sim: computed on prompt-token logprobs between the
    two models (full logits are not exposed over the API).
  - kl_divergence: NaN — full distributions are not available over the API.
  - peak memory: nvidia-smi snapshot; vLLM preallocates the KV cache, so this
    reflects --gpu-memory-utilization rather than per-shape working memory.
"""

import argparse
import json
import statistics
import subprocess
import time
import urllib.request

SHAPES = [  # (batch, seq) — Itamar's sweep
    (1, 128), (1, 512), (1, 2048),
    (8, 128), (8, 512), (8, 2048),
    (32, 128), (32, 512), (32, 2048),
]
WARMUP = 3
MEASURE = 20
SEED = 1234
TOKEN_ID_LOW, TOKEN_ID_HIGH = 2000, 150000  # plain-text id range, avoids specials


def make_prompts(batch: int, seq: int):
    """Deterministic token-id prompts (no tokenizer needed client-side)."""
    state = SEED + batch * 100_000 + seq
    prompts = []
    for b in range(batch):
        ids, s = [], state + b * 7919
        for _ in range(seq):
            s = (s * 6364136223846793005 + 1442695040888963407) % (2 ** 63)
            ids.append(TOKEN_ID_LOW + s % (TOKEN_ID_HIGH - TOKEN_ID_LOW))
        prompts.append(ids)
    return prompts


def post(url: str, body: dict, timeout=600):
    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def gpu_mem_used_mb() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        return float(out.decode().strip().splitlines()[0])
    except Exception:
        return float("nan")


def capture(args):
    results = {}
    for (batch, seq) in SHAPES:
        prompts = make_prompts(batch, seq)
        # Latency body: NO echo/logprobs — prompt-logprob computation
        # materializes [tokens x vocab] logits (~13GB at 8x2048) and both
        # skews latency and can OOM-crash the server at large shapes.
        lat_body = {
            "model": args.model,
            "prompt": prompts,
            "max_tokens": 1,
            "temperature": 0.0,
            "seed": 0,
        }
        try:
            for _ in range(args.warmup):
                post(args.url, lat_body)
            latencies = []
            for _ in range(args.measure):
                t0 = time.perf_counter()
                post(args.url, lat_body)
                latencies.append((time.perf_counter() - t0) * 1000.0)
        except Exception as e:
            print(f"  batch={batch:>2} seq={seq:>4}: FAILED ({type(e).__name__}: {e}) — "
                  f"check the server log; skipping shape")
            continue

        # Equivalence body: echo+logprobs, only at small shapes (token budget)
        logprobs = []
        if batch * seq <= args.logprob_max_tokens:
            eq_body = dict(lat_body, logprobs=1, echo=True)
            try:
                resp = post(args.url, eq_body)
                for choice in resp["choices"]:
                    lp = (choice.get("logprobs") or {}).get("token_logprobs") or []
                    logprobs.append([v for v in lp if v is not None])
            except Exception as e:
                print(f"    (logprob collection failed at batch={batch} seq={seq}: "
                      f"{type(e).__name__} — latency kept, equivalence skipped)")

        med = statistics.median(latencies)
        results[f"{batch}x{seq}"] = {
            "batch": batch, "seq_len": seq,
            "latencies_ms": latencies,
            "median_ms": med,
            "p99_ms": sorted(latencies)[min(int(len(latencies) * 0.99), len(latencies) - 1)],
            "throughput_tok_s": batch * seq / (med / 1000.0),
            "gpu_mem_mb": gpu_mem_used_mb(),
            "prompt_logprobs": logprobs,
        }
        print(f"  batch={batch:>2} seq={seq:>4}: median={med:8.1f}ms  "
              f"p99={results[f'{batch}x{seq}']['p99_ms']:8.1f}ms  "
              f"tok/s={results[f'{batch}x{seq}']['throughput_tok_s']:>10,.0f}")

    with open(args.out, "w") as f:
        json.dump({"model": args.model, "shapes": results}, f)
    print(f"\nSaved capture to {args.out}")


def compare(args):
    fused = json.load(open(args.fused_json))["shapes"]
    nonfused = json.load(open(args.nonfused_json))["shapes"]

    fields = [
        "batch", "seq_len", "hidden", "out_dim",
        "fused_median_ms", "nonfused_median_ms", "speedup",
        "fused_p99_ms", "nonfused_p99_ms",
        "fused_throughput", "nonfused_throughput",
        "fused_peak_mem_mb", "nonfused_peak_mem_mb",
        "max_abs_diff", "cosine_sim", "kl_divergence",
    ]
    rows = []
    header = (f"{'batch':>5} {'seq':>5} {'fused_med':>10} {'nf_med':>10} {'speedup':>8} "
              f"{'max_dlp':>9} {'cos_sim':>9}")
    print(f"{header}\n{'-' * len(header)}")
    for key in fused:
        f, nf = fused[key], nonfused.get(key)
        if nf is None:
            continue
        # logprob equivalence (NaN when not collected for this shape)
        max_diff, cos = float("nan"), float("nan")
        dots = norm_f = norm_nf = 0.0
        n_pairs = 0
        for lf, lnf in zip(f["prompt_logprobs"], nf["prompt_logprobs"]):
            for a, b in zip(lf, lnf):
                max_diff = max(max_diff, abs(a - b)) if n_pairs else abs(a - b)
                dots += a * b
                norm_f += a * a
                norm_nf += b * b
                n_pairs += 1
        if n_pairs and norm_f > 0 and norm_nf > 0:
            cos = dots / (norm_f ** 0.5 * norm_nf ** 0.5)

        row = {
            "batch": f["batch"], "seq_len": f["seq_len"],
            "hidden": args.hidden, "out_dim": args.hidden,
            "fused_median_ms": f["median_ms"],
            "nonfused_median_ms": nf["median_ms"],
            "speedup": nf["median_ms"] / f["median_ms"],
            "fused_p99_ms": f["p99_ms"], "nonfused_p99_ms": nf["p99_ms"],
            "fused_throughput": f["throughput_tok_s"],
            "nonfused_throughput": nf["throughput_tok_s"],
            "fused_peak_mem_mb": f["gpu_mem_mb"],
            "nonfused_peak_mem_mb": nf["gpu_mem_mb"],
            "max_abs_diff": max_diff,   # on prompt-token logprobs (see header note)
            "cosine_sim": cos,          # on prompt-token logprob vectors
            "kl_divergence": float("nan"),  # full distribution unavailable over API
        }
        rows.append(row)
        print(f"{row['batch']:>5} {row['seq_len']:>5} {row['fused_median_ms']:>10.1f} "
              f"{row['nonfused_median_ms']:>10.1f} {row['speedup']:>7.2f}x "
              f"{max_diff:>9.4f} {cos:>9.6f}")

    import csv
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nResults saved to: {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture")
    c.add_argument("--url", default="http://localhost:8000")
    c.add_argument("--model", required=True, help="served model name (the path)")
    c.add_argument("--out", required=True)
    c.add_argument("--warmup", type=int, default=WARMUP)
    c.add_argument("--measure", type=int, default=MEASURE)
    c.add_argument("--logprob-max-tokens", type=int, default=4096,
                   help="Collect equivalence logprobs only when batch*seq <= this "
                        "(prompt logprobs materialize [tokens x vocab] logits)")

    p = sub.add_parser("compare")
    p.add_argument("fused_json")
    p.add_argument("nonfused_json")
    p.add_argument("--out", default="vllm_itamar_results.csv")
    p.add_argument("--hidden", type=int, default=2880)

    args = ap.parse_args()
    capture(args) if args.cmd == "capture" else compare(args)
