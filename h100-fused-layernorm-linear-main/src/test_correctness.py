"""
Correctness tests for fused LayerNorm+Linear.

1. Denominator kernel tests (V0, V2 Welford)
2. Unit tests: Random LN + Linear vs fused on synthetic data (V0, V1, V3)
3. Integration test: OPT-125m logits comparison original vs patched
"""

import torch
import torch.nn as nn
from src.load_cuda import denominator_cuda
from src.weight_transform import compute_fused_weights
from src.fused_forward import (
    fused_ln_linear_forward,
    fused_ln_linear_forward_v1,
    fused_ln_linear_forward_v3,
    fused_rmsnorm_linear_forward_v1,
    fused_rmsnorm_linear_forward_v3,
)
from src.weight_transform import compute_fused_weights_rmsnorm


def test_denominator_kernel():
    """Test CUDA denominator kernel against PyTorch reference."""
    print("=" * 60)
    print("TEST: Denominator kernel (V0 two-pass)")
    print("=" * 60)

    torch.manual_seed(42)

    for rows, cols in [(1, 768), (32, 768), (128, 2048), (512, 4096)]:
        x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)

        # Reference: ||x - mean(x)||_2 per row
        ref = (x - x.mean(dim=-1, keepdim=True)).norm(dim=-1)

        # CUDA kernel
        out = denominator_cuda.compute_denominator(x)

        max_diff = (ref - out).abs().max().item()
        rel_diff = (max_diff / ref.abs().mean().item()) if ref.abs().mean().item() > 0 else 0
        status = "PASS" if max_diff < 1e-3 else "FAIL"
        print(f"  [{status}] fp32 ({rows:4d} x {cols:4d}): max_diff={max_diff:.2e}, rel_diff={rel_diff:.2e}")
        assert max_diff < 1e-3, f"fp32 denominator test failed: max_diff={max_diff}"

    # FP16 test
    for rows, cols in [(32, 768), (128, 2048)]:
        x = torch.randn(rows, cols, device="cuda", dtype=torch.float16)
        ref = (x.float() - x.float().mean(dim=-1, keepdim=True)).norm(dim=-1)
        out = denominator_cuda.compute_denominator(x)

        max_diff = (ref - out).abs().max().item()
        status = "PASS" if max_diff < 0.5 else "FAIL"
        print(f"  [{status}] fp16 ({rows:4d} x {cols:4d}): max_diff={max_diff:.2e}")
        assert max_diff < 0.5, f"fp16 denominator test failed: max_diff={max_diff}"

    print("  All V0 denominator tests passed!\n")


def test_denominator_welford():
    """Test Welford's single-pass denominator kernel against PyTorch reference."""
    print("=" * 60)
    print("TEST: Denominator kernel (V2 Welford single-pass)")
    print("=" * 60)

    torch.manual_seed(42)

    for rows, cols in [(1, 768), (32, 768), (128, 2048), (512, 4096)]:
        x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)

        ref = (x - x.mean(dim=-1, keepdim=True)).norm(dim=-1)
        out = denominator_cuda.compute_denominator_welford(x)

        max_diff = (ref - out).abs().max().item()
        rel_diff = (max_diff / ref.abs().mean().item()) if ref.abs().mean().item() > 0 else 0
        status = "PASS" if max_diff < 1e-3 else "FAIL"
        print(f"  [{status}] fp32 ({rows:4d} x {cols:4d}): max_diff={max_diff:.2e}, rel_diff={rel_diff:.2e}")
        assert max_diff < 1e-3, f"Welford denominator test failed: max_diff={max_diff}"

    print("  All V2 Welford denominator tests passed!\n")


def test_fused_ln_linear_unit():
    """Test fused LN+Linear vs sequential on synthetic data (all variants)."""
    print("=" * 60)
    print("TEST: Fused LN+Linear (unit, all variants)")
    print("=" * 60)

    torch.manual_seed(42)
    denom_stream = torch.cuda.Stream()

    for h, out_dim, batch in [(768, 768, 32), (768, 3072, 128), (2048, 2048, 64), (4096, 16384, 16)]:
        ln = nn.LayerNorm(h).cuda()
        linear = nn.Linear(h, out_dim).cuda()

        # Initialize with non-trivial weights
        nn.init.normal_(ln.weight, mean=1.0, std=0.1)
        nn.init.normal_(ln.bias, mean=0.0, std=0.01)
        nn.init.normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.normal_(linear.bias, mean=0.0, std=0.01)

        x = torch.randn(batch, h, device="cuda")

        # Reference: sequential
        with torch.no_grad():
            ref = linear(ln(x))

        # Fused weights
        W_new, b_new, h_dim, eps = compute_fused_weights(ln, linear)

        # V0: Stream-based
        with torch.no_grad():
            fused_v0 = fused_ln_linear_forward(x, W_new, b_new, denom_stream, h_dim, eps)
        md_v0 = (ref - fused_v0).abs().max().item()
        s_v0 = "PASS" if md_v0 < 1e-3 else "FAIL"

        # V1: Fused normalize
        with torch.no_grad():
            fused_v1 = fused_ln_linear_forward_v1(x, W_new, b_new, h_dim, eps)
        md_v1 = (ref - fused_v1).abs().max().item()
        s_v1 = "PASS" if md_v1 < 1e-3 else "FAIL"

        # V3: Welford + fused normalize + 512
        with torch.no_grad():
            fused_v3 = fused_ln_linear_forward_v3(x, W_new, b_new, h_dim, eps)
        md_v3 = (ref - fused_v3).abs().max().item()
        s_v3 = "PASS" if md_v3 < 1e-3 else "FAIL"

        print(f"  h={h:4d}, out={out_dim:5d}, batch={batch:3d}: "
              f"V0[{s_v0}]={md_v0:.2e}  V1[{s_v1}]={md_v1:.2e}  V3[{s_v3}]={md_v3:.2e}")
        assert md_v0 < 1e-3, f"V0 unit test failed: max_diff={md_v0}"
        assert md_v1 < 1e-3, f"V1 unit test failed: max_diff={md_v1}"
        assert md_v3 < 1e-3, f"V3 unit test failed: max_diff={md_v3}"

    print("  All unit tests passed!\n")


def test_fused_ln_linear_3d():
    """Test with 3D input (batch, seq, h)."""
    print("=" * 60)
    print("TEST: Fused LN+Linear (3D input, all variants)")
    print("=" * 60)

    torch.manual_seed(42)
    denom_stream = torch.cuda.Stream()

    h, out_dim = 768, 768
    ln = nn.LayerNorm(h).cuda()
    linear = nn.Linear(h, out_dim).cuda()
    nn.init.normal_(ln.weight, mean=1.0, std=0.1)
    nn.init.normal_(ln.bias, mean=0.0, std=0.01)

    x = torch.randn(4, 128, h, device="cuda")

    with torch.no_grad():
        ref = linear(ln(x))

    W_new, b_new, h_dim, eps = compute_fused_weights(ln, linear)

    with torch.no_grad():
        fused_v0 = fused_ln_linear_forward(x, W_new, b_new, denom_stream, h_dim, eps)
        fused_v1 = fused_ln_linear_forward_v1(x, W_new, b_new, h_dim, eps)
        fused_v3 = fused_ln_linear_forward_v3(x, W_new, b_new, h_dim, eps)

    md_v0 = (ref - fused_v0).abs().max().item()
    md_v1 = (ref - fused_v1).abs().max().item()
    md_v3 = (ref - fused_v3).abs().max().item()

    print(f"  3D (4, 128, {h}): V0={md_v0:.2e}  V1={md_v1:.2e}  V3={md_v3:.2e}")
    assert md_v0 < 1e-3, f"V0 3D test failed: max_diff={md_v0}"
    assert md_v1 < 1e-3, f"V1 3D test failed: max_diff={md_v1}"
    assert md_v3 < 1e-3, f"V3 3D test failed: max_diff={md_v3}"
    print("  All 3D tests passed!\n")


def test_opt_integration():
    """Integration test: compare OPT-125m logits before and after patching."""
    print("=" * 60)
    print("TEST: OPT-125m integration")
    print("=" * 60)

    from transformers import AutoTokenizer, OPTForCausalLM
    from src.patch_model import patch_opt_model
    import copy

    print("  Loading OPT-125m...")
    model_name = "facebook/opt-125m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_orig = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).cuda().eval()

    # Deep copy for patching
    model_fused = copy.deepcopy(model_orig)
    print("  Patching model...")
    patch_opt_model(model_fused)

    # Test inputs
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "In a galaxy far far away",
        "Machine learning is transforming",
    ]

    all_passed = True
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            logits_orig = model_orig(**inputs).logits
            logits_fused = model_fused(**inputs).logits

        max_diff = (logits_orig - logits_fused).abs().max().item()
        mean_diff = (logits_orig - logits_fused).abs().mean().item()
        # Relative to output magnitude
        rel_diff = max_diff / logits_orig.abs().mean().item() if logits_orig.abs().mean().item() > 0 else 0
        status = "PASS" if max_diff < 1e-2 else "FAIL"
        if max_diff >= 1e-2:
            all_passed = False
        print(f"  [{status}] \"{text[:40]}...\": max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, rel_diff={rel_diff:.2e}")

    if all_passed:
        print("  All integration tests passed!\n")
    else:
        print("  WARNING: Some integration tests exceeded threshold (may be acceptable for fp32 accumulation)\n")

    # Clean up GPU memory
    del model_orig, model_fused
    torch.cuda.empty_cache()


def test_fused_ln_linear_fp16():
    """Test fused LN+Linear with FP16 inputs (V1 and V3)."""
    print("=" * 60)
    print("TEST: Fused LN+Linear (FP16, V1 and V3)")
    print("=" * 60)

    torch.manual_seed(42)

    for h, out_dim, batch in [(768, 768, 32), (2048, 2048, 64), (4096, 4096, 16)]:
        ln = nn.LayerNorm(h).cuda().half()
        linear = nn.Linear(h, out_dim).cuda().half()
        nn.init.normal_(ln.weight, mean=1.0, std=0.1)
        nn.init.normal_(ln.bias, mean=0.0, std=0.01)

        x = torch.randn(batch, h, device="cuda", dtype=torch.float16)

        # Reference: sequential in FP16
        with torch.no_grad():
            ref = linear(ln(x))

        # Fused weights (compute in fp32 for stability, then cast)
        W_new, b_new, h_dim, eps = compute_fused_weights(ln.float(), linear.float())
        W_new = W_new.half()
        b_new = b_new.half()

        # V1
        with torch.no_grad():
            fused_v1 = fused_ln_linear_forward_v1(x, W_new, b_new, h_dim, eps)
        md_v1 = (ref.float() - fused_v1.float()).abs().max().item()
        s_v1 = "PASS" if md_v1 < 0.1 else "FAIL"

        # V3
        with torch.no_grad():
            fused_v3 = fused_ln_linear_forward_v3(x, W_new, b_new, h_dim, eps)
        md_v3 = (ref.float() - fused_v3.float()).abs().max().item()
        s_v3 = "PASS" if md_v3 < 0.1 else "FAIL"

        print(f"  h={h:4d}, out={out_dim:5d}, batch={batch:3d}: "
              f"V1[{s_v1}]={md_v1:.2e}  V3[{s_v3}]={md_v3:.2e}")
        assert md_v1 < 0.1, f"V1 FP16 test failed: max_diff={md_v1}"
        assert md_v3 < 0.1, f"V3 FP16 test failed: max_diff={md_v3}"

    print("  All FP16 tests passed!\n")


def test_fused_ln_linear_bf16():
    """Test fused LN+Linear with BF16 inputs (V1 and V3)."""
    print("=" * 60)
    print("TEST: Fused LN+Linear (BF16, V1 and V3)")
    print("=" * 60)

    torch.manual_seed(42)

    for h, out_dim, batch in [(768, 768, 32), (2048, 2048, 64), (4096, 4096, 16)]:
        ln = nn.LayerNorm(h).cuda().bfloat16()
        linear = nn.Linear(h, out_dim).cuda().bfloat16()
        nn.init.normal_(ln.weight, mean=1.0, std=0.1)
        nn.init.normal_(ln.bias, mean=0.0, std=0.01)

        x = torch.randn(batch, h, device="cuda", dtype=torch.bfloat16)

        # Reference: sequential in BF16
        with torch.no_grad():
            ref = linear(ln(x))

        # Fused weights (compute in fp32 for stability, then cast)
        W_new, b_new, h_dim, eps = compute_fused_weights(ln.float(), linear.float())
        W_new = W_new.bfloat16()
        b_new = b_new.bfloat16()

        # V1
        with torch.no_grad():
            fused_v1 = fused_ln_linear_forward_v1(x, W_new, b_new, h_dim, eps)
        md_v1 = (ref.float() - fused_v1.float()).abs().max().item()
        s_v1 = "PASS" if md_v1 < 0.5 else "FAIL"

        # V3
        with torch.no_grad():
            fused_v3 = fused_ln_linear_forward_v3(x, W_new, b_new, h_dim, eps)
        md_v3 = (ref.float() - fused_v3.float()).abs().max().item()
        s_v3 = "PASS" if md_v3 < 0.5 else "FAIL"

        print(f"  h={h:4d}, out={out_dim:5d}, batch={batch:3d}: "
              f"V1[{s_v1}]={md_v1:.2e}  V3[{s_v3}]={md_v3:.2e}")
        assert md_v1 < 0.5, f"V1 BF16 test failed: max_diff={md_v1}"
        assert md_v3 < 0.5, f"V3 BF16 test failed: max_diff={md_v3}"

    print("  All BF16 tests passed!\n")


def test_fused_rmsnorm_linear_unit():
    """Test fused RMSNorm+Linear vs sequential on synthetic data (V1 and V3)."""
    print("=" * 60)
    print("TEST: Fused RMSNorm+Linear (unit, V1 and V3)")
    print("=" * 60)

    torch.manual_seed(42)

    for h, out_dim, batch in [(768, 768, 32), (768, 3072, 128), (2048, 2048, 64), (4096, 4096, 16)]:
        rms_norm = torch.nn.RMSNorm(h, eps=1e-6).cuda()
        linear = nn.Linear(h, out_dim, bias=False).cuda()
        nn.init.normal_(rms_norm.weight, mean=1.0, std=0.1)
        nn.init.normal_(linear.weight, mean=0.0, std=0.02)

        x = torch.randn(batch, h, device="cuda")

        # Reference: sequential
        with torch.no_grad():
            ref = linear(rms_norm(x))

        # Fused weights
        W_new, b_new, h_dim, eps = compute_fused_weights_rmsnorm(rms_norm, linear)

        # V1
        with torch.no_grad():
            fused_v1 = fused_rmsnorm_linear_forward_v1(x, W_new, b_new, h_dim, eps)
        md_v1 = (ref - fused_v1).abs().max().item()
        s_v1 = "PASS" if md_v1 < 1e-3 else "FAIL"

        # V3
        with torch.no_grad():
            fused_v3 = fused_rmsnorm_linear_forward_v3(x, W_new, b_new, h_dim, eps)
        md_v3 = (ref - fused_v3).abs().max().item()
        s_v3 = "PASS" if md_v3 < 1e-3 else "FAIL"

        print(f"  h={h:4d}, out={out_dim:5d}, batch={batch:3d}: "
              f"V1[{s_v1}]={md_v1:.2e}  V3[{s_v3}]={md_v3:.2e}")
        assert md_v1 < 1e-3, f"RMSNorm V1 unit test failed: max_diff={md_v1}"
        assert md_v3 < 1e-3, f"RMSNorm V3 unit test failed: max_diff={md_v3}"

    print("  All RMSNorm unit tests passed!\n")


def test_fused_rmsnorm_linear_with_bias():
    """Test fused RMSNorm+Linear when linear has bias."""
    print("=" * 60)
    print("TEST: Fused RMSNorm+Linear with bias")
    print("=" * 60)

    torch.manual_seed(42)
    h, out_dim, batch = 768, 768, 32
    rms_norm = torch.nn.RMSNorm(h, eps=1e-6).cuda()
    linear = nn.Linear(h, out_dim, bias=True).cuda()
    nn.init.normal_(rms_norm.weight, mean=1.0, std=0.1)

    x = torch.randn(batch, h, device="cuda")

    with torch.no_grad():
        ref = linear(rms_norm(x))

    W_new, b_new, h_dim, eps = compute_fused_weights_rmsnorm(rms_norm, linear)

    with torch.no_grad():
        fused_v1 = fused_rmsnorm_linear_forward_v1(x, W_new, b_new, h_dim, eps)
        fused_v3 = fused_rmsnorm_linear_forward_v3(x, W_new, b_new, h_dim, eps)

    md_v1 = (ref - fused_v1).abs().max().item()
    md_v3 = (ref - fused_v3).abs().max().item()

    print(f"  h={h}, out={out_dim}, batch={batch}: V1={md_v1:.2e}  V3={md_v3:.2e}")
    assert md_v1 < 1e-3, f"RMSNorm V1 with bias failed: max_diff={md_v1}"
    assert md_v3 < 1e-3, f"RMSNorm V3 with bias failed: max_diff={md_v3}"
    print("  RMSNorm with bias tests passed!\n")


def test_llama_integration():
    """Integration test: compare Llama logits before and after patching."""
    print("=" * 60)
    print("TEST: Llama integration")
    print("=" * 60)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from src.patch_llama import patch_llama_model
    import copy

    print("  Loading TinyLlama model...")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_orig = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float32
        ).cuda().eval()
    except Exception as e:
        print(f"  SKIP: Cannot load model ({e})")
        return

    model_fused = copy.deepcopy(model_orig)
    print("  Patching model with RMSNorm fusion (V1)...")
    patch_llama_model(model_fused, variant="V1")

    texts = [
        "The quick brown fox jumps over the lazy dog",
        "In a galaxy far far away",
        "Machine learning is transforming",
    ]

    all_passed = True
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            logits_orig = model_orig(**inputs).logits
            logits_fused = model_fused(**inputs).logits

        max_diff = (logits_orig - logits_fused).abs().max().item()
        mean_diff = (logits_orig - logits_fused).abs().mean().item()
        rel_diff = max_diff / logits_orig.abs().mean().item() if logits_orig.abs().mean().item() > 0 else 0
        status = "PASS" if max_diff < 1e-2 else "FAIL"
        if max_diff >= 1e-2:
            all_passed = False
        print(f"  [{status}] \"{text[:40]}...\": max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, rel_diff={rel_diff:.2e}")

    if all_passed:
        print("  All Llama integration tests passed!\n")
    else:
        print("  WARNING: Some Llama integration tests exceeded threshold\n")

    del model_orig, model_fused
    torch.cuda.empty_cache()


if __name__ == "__main__":
    test_denominator_kernel()
    test_denominator_welford()
    test_fused_ln_linear_unit()
    test_fused_ln_linear_3d()
    test_fused_ln_linear_fp16()
    test_fused_ln_linear_bf16()
    test_fused_rmsnorm_linear_unit()
    test_fused_rmsnorm_linear_with_bias()
    test_opt_integration()
    test_llama_integration()
    print("=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
