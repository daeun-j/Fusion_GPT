"""Build the CUDA extension using JIT compilation, bypassing CUDA version check."""
import torch.utils.cpp_extension as ext

# Monkey-patch the CUDA version check to allow forward-compatible drivers
_orig_check = ext._check_cuda_version
def _noop_check(*args, **kwargs):
    pass
ext._check_cuda_version = _noop_check

module = ext.load(
    name="denominator_cuda",
    sources=[
        "csrc/denominator.cpp",
        "csrc/denominator_kernel.cu",
    ],
    extra_cuda_cflags=["-arch=sm_90", "-O3", "--use_fast_math"],
    extra_cflags=["-O3"],
    verbose=True,
)

print("Build successful!")
print(f"Module: {module}")
print(f"Functions: {dir(module)}")
