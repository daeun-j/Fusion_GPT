"""
Serving equivalence check: compare two model servers (baseline vs fused)
through their OpenAI-compatible completions API. Works with both vLLM and
SGLang.

Run the server for ONE checkpoint, capture greedy outputs + logprobs:
    python serving_equiv_check.py capture --url http://localhost:8000 \
        --model <served-model-name-or-path> --out baseline.json

Then serve the OTHER checkpoint (same port or different) and capture again:
    python serving_equiv_check.py capture --url http://localhost:8000 \
        --model <path> --out fused.json

Finally compare:
    python serving_equiv_check.py compare baseline.json fused.json

Pass criteria: 100% (or near-100%) exact token match under greedy decoding,
and small mean |delta logprob|. BF16 nondeterminism (batching/kernel order)
can cause occasional divergence after many tokens, so token match is checked
position-by-position until first mismatch.
"""

import argparse
import json
import sys
import urllib.request

PROMPTS = [
    "The capital of France is",
    "In mathematics, a prime number is",
    "def fibonacci(n):",
    "The mitochondria is",
    "Once upon a time in a distant galaxy,",
    "The three laws of thermodynamics state that",
    "To make a good espresso, you need",
    "The difference between TCP and UDP is",
    "In 1969, the Apollo 11 mission",
    "Machine learning models often overfit when",
    "The Korean alphabet, Hangul, was created",
    "Photosynthesis converts",
    "A binary search tree is a data structure where",
    "The stock market crashed in 1929 because",
    "Quantum entanglement occurs when",
    "The recipe for kimchi jjigae starts with",
]
MAX_TOKENS = 64


def post_completion(url: str, model: str, prompt: str):
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "logprobs": 1,
        "seed": 0,
    }).encode()
    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def capture(args):
    results = []
    for i, prompt in enumerate(PROMPTS):
        resp = post_completion(args.url, args.model, prompt)
        choice = resp["choices"][0]
        lp = choice.get("logprobs") or {}
        results.append({
            "prompt": prompt,
            "text": choice["text"],
            "tokens": lp.get("tokens", []),
            "token_logprobs": lp.get("token_logprobs", []),
        })
        print(f"[{i + 1}/{len(PROMPTS)}] {prompt[:40]!r} -> {choice['text'][:50]!r}")
    with open(args.out, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"\nSaved {len(results)} completions to {args.out}")


def compare(args):
    a = json.load(open(args.file_a))
    b = json.load(open(args.file_b))
    assert len(a) == len(b), "prompt count mismatch"

    full_match = 0
    total_prefix = 0.0
    lp_deltas = []
    for ra, rb in zip(a, b):
        ta, tb = ra["tokens"], rb["tokens"]
        n = min(len(ta), len(tb))
        prefix = next((i for i in range(n) if ta[i] != tb[i]), n)
        is_full = prefix == n and len(ta) == len(tb)
        full_match += is_full
        total_prefix += prefix / max(n, 1)
        for la, lb in zip(ra["token_logprobs"][:prefix], rb["token_logprobs"][:prefix]):
            if la is not None and lb is not None:
                lp_deltas.append(abs(la - lb))
        status = "OK " if is_full else f"DIVERGES@{prefix}"
        print(f"  {status}  {ra['prompt'][:45]!r}")
        if not is_full:
            print(f"      A: ...{' '.join(map(str, ta[prefix:prefix + 5]))}")
            print(f"      B: ...{' '.join(map(str, tb[prefix:prefix + 5]))}")

    n = len(a)
    mean_dlp = sum(lp_deltas) / len(lp_deltas) if lp_deltas else float("nan")
    max_dlp = max(lp_deltas) if lp_deltas else float("nan")
    print(f"\nExact-match completions : {full_match}/{n}")
    print(f"Mean matched prefix     : {100 * total_prefix / n:.1f}%")
    print(f"Logprob |delta| mean/max: {mean_dlp:.5f} / {max_dlp:.5f}")
    if full_match == n:
        print("PASS: outputs identical under greedy decoding.")
    elif total_prefix / n > 0.9 and mean_dlp < 0.05:
        print("LIKELY PASS: minor late-token divergence consistent with BF16 "
              "nondeterminism; logprob deltas are small.")
    else:
        print("FAIL: outputs differ substantially — investigate before reporting.")
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture", help="query a running server, save outputs")
    c.add_argument("--url", default="http://localhost:8000")
    c.add_argument("--model", required=True, help="served model name (usually the path)")
    c.add_argument("--out", required=True)

    p = sub.add_parser("compare", help="compare two capture files")
    p.add_argument("file_a")
    p.add_argument("file_b")

    args = ap.parse_args()
    capture(args) if args.cmd == "capture" else compare(args)
