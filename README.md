# Project Plan — Fusion GPT-OSS 20B (vLLM / SGLang, A100 & H100)

## Overview
This is a high-level project overview for the RMSNorm + Linear fusion task on GPT-OSS 20B.
The goal of the project is to apply the RMSNorm + Linear fusion optimization (from the H100
kernel-fusion work) to GPT-OSS 20B, validate it on both **A100 (Ampere, sm_80)** and
**H100 (Hopper, sm_90)**, run the standardized kernel-level benchmark (Itamar's
`benchmark_rmsnorm_linear_fusion.py`) comparing fused vs. non-fused checkpoints, and finally
measure serving-level performance with **vLLM** and **SGLang** on both GPUs.

This plan covers four tracks:
1. Fusion GPT-OSS 20B — vLLM — A100/H100
2. Fusion GPT-OSS 20B — SGLang — A100/H100
3. Kernel-level fused vs. non-fused benchmark (plain PyTorch, NOT vLLM/SGLang)
4. Quantization assessment (MXFP4 native — verify whether extra quantization is needed)

## Prerequisites
Before working, please make sure you have access to the following:
- Claude Code (Runara subscription)
- GitHub (Runara repos)
- Vast.ai team subscription (A100 80GB and H100 80GB instances)
- HuggingFace account with write access (for uploading fused / non-fused checkpoints)
- (Optional) Tailscale, if any box is reached through the tailnet

## Useful Resources
- H100 RMSNorm fusion (algorithm + CUDA kernel): https://github.com/Runaraai/h100-kernel-fusion-rmsnorm-swiglu
- LayerNorm fusion: https://github.com/Runaraai/l4-layernorm-fusion
- Local copies in this folder:
  - `h100-fused-layernorm-linear-main/` — full version (LayerNorm + RMSNorm, FP32/FP16/BF16, V0–V3 kernels, TE baseline)
  - `h100-fused-layernorm-linear-opt-main/` — earlier OPT-only FP32 version (reference only)
  - `benchmark_rmsnorm_linear_fusion.py` — Itamar's standardized benchmark (do not modify; report raw CSV)
  - `RMSNorm_Linear_Fusion_Benchmark_Procedure.docx` — benchmark procedure & metric definitions

---

## Project Plan

### 1. Model: GPT-OSS 20B
- HF link (original): https://huggingface.co/openai/gpt-oss-20b
- BF16 version (upcast): https://huggingface.co/unsloth/gpt-oss-20b-BF16
- Precision:
  - **Original: MXFP4 (MoE expert weights, ~90% of params) + BF16 (attention, router, embeddings, norms)**
  - **Used version: BF16** (fusion bakes γ into weights, so we work on dequantized BF16
    weights — same decision as the DeepSeek V4 Flash plan)
- Architecture facts relevant to fusion:
  - 24 layers, hidden_size = 2880, RMSNorm, GQA (64 q-heads / 8 kv-heads, head_dim 64),
    MoE MLP (32 experts, top-4 routing, SwiGLU-with-clamp), attention sinks
  - Fusion points: `input_layernorm → q/k/v_proj` (straightforward, Llama-like) and
    `post_attention_layernorm → router + experts.gate_up_proj` (γ must be absorbed into
    BOTH the router weight and every expert's gate_up weight)
  - ⚠️ Risk: hidden=2880 is between OPT-1.3b (2048) and OPT-6.7b (4096); on H100 the fusion
    only clearly paid off from h=4096. Expect modest speedups; that is a valid finding.
- Verify model loading, feedforward, and logits sanity check (BF16, transformers)
- **Milestone: Download model (MXFP4 original + BF16) on both A100 and H100 boxes**

### 2. Environment: A100 & H100 (Vast.ai)
- A100 80GB (sm_80) — Vast.ai
- H100 80GB (sm_90) — Vast.ai
- Stack: CUDA toolkit ≥ 12.4, PyTorch ≥ 2.4 (cu12x), transformers (gpt_oss support, ≥ 4.55),
  safetensors, accelerate; vLLM ≥ 0.10 and SGLang ≥ 0.4.9 for the serving tracks
- **Milestone: Verify Access (Vast.ai A100)**
- **Milestone: Verify Access (Vast.ai H100)**
- **Milestone: CUDA extension JIT-builds and `src.test_correctness` passes on both GPUs**
  - H100: build as-is (`-arch=sm_90`)
  - A100: change `-arch=sm_90` → `-arch=sm_80` in `src/load_cuda.py`, `build_ext.py`,
    `setup.py` (or use `-gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90`
    for a single fatbinary). The kernel uses only warp shuffles, float4 and `__nv_bfloat162`
    loads — all supported on sm_80; no Hopper-only features (TMA/WGMMA/clusters) are used.

### 3. Optimization: RMSNorm + Linear fusion on GPT-OSS 20B
- Algorithm reference: https://github.com/Runaraai/h100-kernel-fusion-rmsnorm-swiglu
- Apply ONLY RMSNorm + Linear fusion (RMSNorm variant: `W_new = W * γ`, runtime kernel computes
  `rms` denominator and normalizes in-place; no column-centering needed, unlike LayerNorm)
- Work items:
  - Port `patch_llama.py` → `patch_gpt_oss.py` (GQA already handled; add MoE handling:
    absorb γ into router weight AND batched 3-D `experts.gate_up_proj`; `down_proj` is NOT
    fused — it follows the activation, not a norm; attention sinks are untouched)
  - Correctness gate: full-model logits max-diff vs. unpatched BF16 model
    (target ≲ 1e-2 in BF16, consistent with repo's BF16 numbers)
  - Save the patched model as a standard HF checkpoint (fused), and the unpatched one (non-fused)
- **Milestone: Save fused + non-fused (control) models to HuggingFace**
  - Fused: `<org>/gpt-oss-20b-rmsnorm-fused` (TBD)
  - Control: `<org>/gpt-oss-20b-baseline` (TBD)

### 4. Benchmarks for Fusion — Standardized (Itamar's)
- Script: `benchmark_rmsnorm_linear_fusion.py` (this folder)
- **Does NOT use vLLM or SGLang — plain PyTorch, kernel-level**
- Layout required on disk:
  ```
  <dir>/models/fused/        # fused HF checkpoint
  <dir>/models/non-fused/    # control HF checkpoint
  ```
- Commands:
  ```bash
  # Step 1 — discover layer paths once
  python benchmark_rmsnorm_linear_fusion.py --dir <dir> --print-keys

  # Step 2 — run the sweep (batch 1/8/32 × seq 128/512/2048)
  python benchmark_rmsnorm_linear_fusion.py --dir <dir> --layer-path model.layers.0.mlp
  ```
  - Prefer `model.layers.0.mlp`: the script calls `module(x)` with a plain tensor, which works
    for the MLP but NOT for `self_attn` (GPT-OSS attention forward needs rotary embeddings /
    masks). If attention must be benchmarked, wrap it; flag any script change to Itamar first.
  - Note: script casts activations to FP16; GPT-OSS is BF16-native. If numerics look bad,
    flag before changing `DTYPE` (procedure says report raw, unmodified results).
- Pass criteria: speedup > 1.0×, cosine ≥ 0.99, max|diff| ≤ 0.1 (else flag before reporting)
- **Deliverable: benchmark_results.csv — fused vs. control on H100**
- **Deliverable: benchmark_results.csv — fused vs. control on A100**

### 5. Serving Benchmarks: vLLM (A100 & H100)
- Serve both checkpoints with vLLM (`vllm serve <model>`), BF16
  - On H100, original MXFP4 gpt-oss also runs natively (triton/FlashInfer MXFP4 kernels);
    on A100 vLLM uses the MXFP4-Marlin / dequant-to-BF16 path — record which path is active
- Measure with `vllm bench serve` (or `benchmark_serving.py`): TTFT, TPOT/ITL, output tok/s,
  request throughput at fixed QPS levels; plus `lm_eval` quick accuracy sanity (optional)
- ⚠️ vLLM's model definition has its own RMSNorm path — the fused checkpoint must either be
  loaded with a custom model patch, or this track reduces to "control characterization"
  if patching vLLM is out of scope. Confirm scope with Itamar/Sukrit before investing here.
- **Deliverable: vLLM serving numbers, control vs. fused — H100**
- **Deliverable: vLLM serving numbers, control vs. fused — A100**

### 6. Serving Benchmarks: SGLang (A100 & H100)
- Same protocol as the vLLM track, using `python -m sglang.launch_server` +
  `sglang.bench_serving`
- Same caveat: SGLang has its own fused RMSNorm kernels; fused-checkpoint support needs a
  model-code patch. Confirm scope first.
- **Deliverable: SGLang serving numbers, control vs. fused — H100**
- **Deliverable: SGLang serving numbers, control vs. fused — A100**

### 7. Quantization
- GPT-OSS 20B is **already MXFP4-native** (expert weights). Open questions to confirm:
  - Is a separate quantization step necessary at all, given the original is MXFP4?
    (Parallel to the DeepSeek plan's "if the model is FP4 already, is this step necessary?")
  - Neither A100 (no FP8/FP4 tensor cores) nor H100 (FP8 but no FP4) computes in FP4 —
    MXFP4 on these GPUs is weight-only storage + BF16/FP16 compute via dequant kernels.
  - If the fused BF16 model must be re-quantized for memory parity with the original,
    candidate route is MXFP4 re-quantization of expert weights only (γ-absorbed); NVFP4
    is Blackwell-targeted and out of scope for A100/H100.
- **Milestone: Verify technique and necessity of quantization (decision memo)**
- (Conditional) **Milestone: Re-quantize fused + control to MXFP4; re-run Itamar's benchmark**

---

## Risks / Alternate Tasks
- **hidden=2880 may be too small for clear wins** — report neutral/negative speedups honestly;
  batch-32 / seq-2048 cells are the most favorable.
- **MoE fusion complexity** — absorbing γ into 32 experts' `gate_up_proj` (3-D tensor) and the
  router is new territory vs. the Llama patch; if blocked, descope to attention-side fusion
  only (`input_layernorm → q/k/v`) and document.
- **vLLM/SGLang fused-model loading** — both engines re-implement RMSNorm; loading a
  γ-absorbed checkpoint without patching their model code silently double-applies γ.
  Verify logits parity before trusting any serving numbers.
- **A100 perf profile differs** — ~2.0 TB/s HBM2e vs. 3.35 TB/s HBM3, 108 vs. 132 SMs;
  breakeven batch shifts upward. Keep V1 as default; V3 (512 threads) worth one re-check on A100.
- **FP16 vs BF16 in the standardized script** — GPT-OSS is BF16-trained; watch for FP16
  overflow in the equivalence metrics and flag (don't silently edit the script).
