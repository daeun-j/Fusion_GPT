#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>

// 256 threads per block, one block per row
constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;

// Warp-level reduction using shuffle
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Block-level reduction via shared memory
__device__ float block_reduce_sum(float val, float* smem) {
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();

    // First warp reduces across warps
    int num_warps = BLOCK_SIZE / WARP_SIZE;  // 8 warps
    if (warp_id == 0) {
        val = (lane < num_warps) ? smem[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }
    __syncthreads();
    return val;  // Only valid in thread 0
}

// FP32 kernel: one block per row, vectorized float4 loads
__global__ void denominator_fp32_kernel(
    const float* __restrict__ x,
    float* __restrict__ output,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_ptr = x + row * cols;
    __shared__ float smem[BLOCK_SIZE / WARP_SIZE];

    // Pass 1: compute mean
    float local_sum = 0.0f;
    int vec_cols = cols / 4;  // number of float4 elements
    int remainder = cols % 4;

    // Vectorized loads
    const float4* row_ptr4 = reinterpret_cast<const float4*>(row_ptr);
    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE) {
        float4 v = row_ptr4[i];
        local_sum += v.x + v.y + v.z + v.w;
    }
    // Handle remainder
    int base = vec_cols * 4;
    for (int i = threadIdx.x; i < remainder; i += BLOCK_SIZE) {
        local_sum += row_ptr[base + i];
    }

    float total = block_reduce_sum(local_sum, smem);

    // Broadcast mean to all threads
    __shared__ float mean_shared;
    if (threadIdx.x == 0) {
        mean_shared = total / (float)cols;
    }
    __syncthreads();
    float mean = mean_shared;

    // Pass 2: compute sum of squared deviations
    float local_sq = 0.0f;
    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE) {
        float4 v = row_ptr4[i];
        float d0 = v.x - mean;
        float d1 = v.y - mean;
        float d2 = v.z - mean;
        float d3 = v.w - mean;
        local_sq += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
    }
    for (int i = threadIdx.x; i < remainder; i += BLOCK_SIZE) {
        float d = row_ptr[base + i] - mean;
        local_sq += d * d;
    }

    float total_sq = block_reduce_sum(local_sq, smem);

    if (threadIdx.x == 0) {
        output[row] = sqrtf(total_sq);
    }
}

// FP16 kernel: one block per row, accumulate in fp32
__global__ void denominator_fp16_kernel(
    const __half* __restrict__ x,
    float* __restrict__ output,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const __half* row_ptr = x + row * cols;
    __shared__ float smem[BLOCK_SIZE / WARP_SIZE];

    // Vectorized loads with half2 (pairs)
    int vec_cols = cols / 2;
    int remainder = cols % 2;
    const __half2* row_ptr2 = reinterpret_cast<const __half2*>(row_ptr);

    // Pass 1: compute mean
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE) {
        __half2 v = row_ptr2[i];
        local_sum += __half2float(v.x) + __half2float(v.y);
    }
    if (remainder && threadIdx.x == 0) {
        local_sum += __half2float(row_ptr[cols - 1]);
    }

    float total = block_reduce_sum(local_sum, smem);

    __shared__ float mean_shared;
    if (threadIdx.x == 0) {
        mean_shared = total / (float)cols;
    }
    __syncthreads();
    float mean = mean_shared;

    // Pass 2: compute sum of squared deviations
    float local_sq = 0.0f;
    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE) {
        __half2 v = row_ptr2[i];
        float d0 = __half2float(v.x) - mean;
        float d1 = __half2float(v.y) - mean;
        local_sq += d0 * d0 + d1 * d1;
    }
    if (remainder && threadIdx.x == 0) {
        float d = __half2float(row_ptr[cols - 1]) - mean;
        local_sq += d * d;
    }

    float total_sq = block_reduce_sum(local_sq, smem);

    if (threadIdx.x == 0) {
        output[row] = sqrtf(total_sq);
    }
}

// ============================================================================
// V1: Fused denominator + normalize kernel (no streams needed)
// Computes mean, sum-of-squared-deviations, then normalizes raw_output in-place:
//   raw_output[row][c] = raw_output[row][c] / std + b_new[c]
// where std = sqrt(sum_sq / h + eps)
// ============================================================================
__global__ void denominator_normalize_fp32_kernel(
    const float* __restrict__ x,         // [rows, h]
    float* __restrict__ raw_output,      // [rows, out_dim] - modified in-place
    const float* __restrict__ b_new,     // [out_dim]
    int rows,
    int h,
    int out_dim,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* x_row = x + row * h;
    float* out_row = raw_output + row * out_dim;
    __shared__ float smem[BLOCK_SIZE / WARP_SIZE];

    // Pass 1: compute mean
    float local_sum = 0.0f;
    int vec_cols = h / 4;
    int remainder = h % 4;
    const float4* x_row4 = reinterpret_cast<const float4*>(x_row);

    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE) {
        float4 v = x_row4[i];
        local_sum += v.x + v.y + v.z + v.w;
    }
    int base = vec_cols * 4;
    for (int i = threadIdx.x; i < remainder; i += BLOCK_SIZE) {
        local_sum += x_row[base + i];
    }

    float total = block_reduce_sum(local_sum, smem);

    __shared__ float mean_shared;
    if (threadIdx.x == 0) {
        mean_shared = total / (float)h;
    }
    __syncthreads();
    float mean = mean_shared;

    // Pass 2: compute sum of squared deviations
    float local_sq = 0.0f;
    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE) {
        float4 v = x_row4[i];
        float d0 = v.x - mean;
        float d1 = v.y - mean;
        float d2 = v.z - mean;
        float d3 = v.w - mean;
        local_sq += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
    }
    for (int i = threadIdx.x; i < remainder; i += BLOCK_SIZE) {
        float d = x_row[base + i] - mean;
        local_sq += d * d;
    }

    float total_sq = block_reduce_sum(local_sq, smem);

    // Broadcast std to all threads via shared memory
    __shared__ float std_shared;
    if (threadIdx.x == 0) {
        std_shared = sqrtf(total_sq / (float)h + eps);
    }
    __syncthreads();
    float std_val = std_shared;
    float inv_std = 1.0f / std_val;

    // Normalize raw_output in-place: out[c] = out[c] / std + b_new[c]
    // Use float4 vectorization for out_dim when possible
    int out_vec = out_dim / 4;
    int out_rem = out_dim % 4;
    float4* out_row4 = reinterpret_cast<float4*>(out_row);
    const float4* b_new4 = reinterpret_cast<const float4*>(b_new);

    for (int i = threadIdx.x; i < out_vec; i += BLOCK_SIZE) {
        float4 o = out_row4[i];
        float4 b = b_new4[i];
        o.x = o.x * inv_std + b.x;
        o.y = o.y * inv_std + b.y;
        o.z = o.z * inv_std + b.z;
        o.w = o.w * inv_std + b.w;
        out_row4[i] = o;
    }
    int out_base = out_vec * 4;
    for (int i = threadIdx.x; i < out_rem; i += BLOCK_SIZE) {
        int c = out_base + i;
        out_row[c] = out_row[c] * inv_std + b_new[c];
    }
}

// ============================================================================
// V1 FP16: Fused denominator + normalize, half input/output, fp32 accumulation
// ============================================================================
__global__ void denominator_normalize_fp16_kernel(
    const __half* __restrict__ x,         // [rows, h]
    __half* __restrict__ raw_output,      // [rows, out_dim] - modified in-place
    const __half* __restrict__ b_new,     // [out_dim]
    int rows,
    int h,
    int out_dim,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const __half* x_row = x + row * h;
    __half* out_row = raw_output + row * out_dim;
    __shared__ float smem[BLOCK_SIZE / WARP_SIZE];

    // Pass 1: compute mean (accumulate in fp32)
    float local_sum = 0.0f;
    int vec_cols = h / 2;
    int remainder = h % 2;
    const __half2* x_row2 = reinterpret_cast<const __half2*>(x_row);

    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE) {
        __half2 v = x_row2[i];
        local_sum += __half2float(v.x) + __half2float(v.y);
    }
    if (remainder && threadIdx.x == 0) {
        local_sum += __half2float(x_row[h - 1]);
    }

    float total = block_reduce_sum(local_sum, smem);

    __shared__ float mean_shared;
    if (threadIdx.x == 0) {
        mean_shared = total / (float)h;
    }
    __syncthreads();
    float mean = mean_shared;

    // Pass 2: compute sum of squared deviations
    float local_sq = 0.0f;
    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE) {
        __half2 v = x_row2[i];
        float d0 = __half2float(v.x) - mean;
        float d1 = __half2float(v.y) - mean;
        local_sq += d0 * d0 + d1 * d1;
    }
    if (remainder && threadIdx.x == 0) {
        float d = __half2float(x_row[h - 1]) - mean;
        local_sq += d * d;
    }

    float total_sq = block_reduce_sum(local_sq, smem);

    __shared__ float std_shared;
    if (threadIdx.x == 0) {
        std_shared = sqrtf(total_sq / (float)h + eps);
    }
    __syncthreads();
    float inv_std = 1.0f / std_shared;

    // Normalize raw_output in-place using half2 vectorization
    int out_vec = out_dim / 2;
    int out_rem = out_dim % 2;
    __half2* out_row2 = reinterpret_cast<__half2*>(out_row);
    const __half2* b_new2 = reinterpret_cast<const __half2*>(b_new);

    for (int i = threadIdx.x; i < out_vec; i += BLOCK_SIZE) {
        __half2 o = out_row2[i];
        __half2 b = b_new2[i];
        float o0 = __half2float(o.x) * inv_std + __half2float(b.x);
        float o1 = __half2float(o.y) * inv_std + __half2float(b.y);
        out_row2[i] = __halves2half2(__float2half(o0), __float2half(o1));
    }
    int out_base = out_vec * 2;
    for (int i = threadIdx.x; i < out_rem; i += BLOCK_SIZE) {
        int c = out_base + i;
        float val = __half2float(out_row[c]) * inv_std + __half2float(b_new[c]);
        out_row[c] = __float2half(val);
    }
}

// ============================================================================
// V1 BF16: Fused denominator + normalize, bfloat16 input/output, fp32 accum
// ============================================================================
__global__ void denominator_normalize_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,         // [rows, h]
    __nv_bfloat16* __restrict__ raw_output,      // [rows, out_dim] - modified in-place
    const __nv_bfloat16* __restrict__ b_new,     // [out_dim]
    int rows,
    int h,
    int out_dim,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const __nv_bfloat16* x_row = x + row * h;
    __nv_bfloat16* out_row = raw_output + row * out_dim;
    __shared__ float smem[BLOCK_SIZE / WARP_SIZE];

    // Pass 1: compute mean (accumulate in fp32)
    float local_sum = 0.0f;
    int vec_cols = h / 2;
    int remainder = h % 2;
    const __nv_bfloat162* x_row2 = reinterpret_cast<const __nv_bfloat162*>(x_row);

    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE) {
        __nv_bfloat162 v = x_row2[i];
        local_sum += __bfloat162float(v.x) + __bfloat162float(v.y);
    }
    if (remainder && threadIdx.x == 0) {
        local_sum += __bfloat162float(x_row[h - 1]);
    }

    float total = block_reduce_sum(local_sum, smem);

    __shared__ float mean_shared;
    if (threadIdx.x == 0) {
        mean_shared = total / (float)h;
    }
    __syncthreads();
    float mean = mean_shared;

    // Pass 2: compute sum of squared deviations
    float local_sq = 0.0f;
    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE) {
        __nv_bfloat162 v = x_row2[i];
        float d0 = __bfloat162float(v.x) - mean;
        float d1 = __bfloat162float(v.y) - mean;
        local_sq += d0 * d0 + d1 * d1;
    }
    if (remainder && threadIdx.x == 0) {
        float d = __bfloat162float(x_row[h - 1]) - mean;
        local_sq += d * d;
    }

    float total_sq = block_reduce_sum(local_sq, smem);

    __shared__ float std_shared;
    if (threadIdx.x == 0) {
        std_shared = sqrtf(total_sq / (float)h + eps);
    }
    __syncthreads();
    float inv_std = 1.0f / std_shared;

    // Normalize raw_output in-place using bfloat162 vectorization
    int out_vec = out_dim / 2;
    int out_rem = out_dim % 2;
    __nv_bfloat162* out_row2 = reinterpret_cast<__nv_bfloat162*>(out_row);
    const __nv_bfloat162* b_new2 = reinterpret_cast<const __nv_bfloat162*>(b_new);

    for (int i = threadIdx.x; i < out_vec; i += BLOCK_SIZE) {
        __nv_bfloat162 o = out_row2[i];
        __nv_bfloat162 b = b_new2[i];
        float o0 = __bfloat162float(o.x) * inv_std + __bfloat162float(b.x);
        float o1 = __bfloat162float(o.y) * inv_std + __bfloat162float(b.y);
        out_row2[i] = __halves2bfloat162(__float2bfloat16(o0), __float2bfloat16(o1));
    }
    int out_base = out_vec * 2;
    for (int i = threadIdx.x; i < out_rem; i += BLOCK_SIZE) {
        int c = out_base + i;
        float val = __bfloat162float(out_row[c]) * inv_std + __bfloat162float(b_new[c]);
        out_row[c] = __float2bfloat16(val);
    }
}

// ============================================================================
// V2: Welford's single-pass denominator kernel
// Computes mean and sum-of-squared-deviations in one pass, halving memory reads.
// ============================================================================

// Warp-level Welford merge using shuffle
__device__ __forceinline__ void warp_welford_reduce(float& count, float& mean, float& M2) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float o_count = __shfl_down_sync(0xFFFFFFFF, count, offset);
        float o_mean  = __shfl_down_sync(0xFFFFFFFF, mean, offset);
        float o_M2    = __shfl_down_sync(0xFFFFFFFF, M2, offset);
        // Merge: (count, mean, M2) + (o_count, o_mean, o_M2)
        float n = count + o_count;
        if (n > 0.0f) {
            float delta = o_mean - mean;
            float new_mean = mean + delta * o_count / n;
            M2 = M2 + o_M2 + delta * delta * count * o_count / n;
            mean = new_mean;
            count = n;
        }
    }
}

__global__ void denominator_welford_fp32_kernel(
    const float* __restrict__ x,
    float* __restrict__ output,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_ptr = x + row * cols;

    // Each thread maintains a Welford triple
    float count = 0.0f;
    float mean = 0.0f;
    float M2 = 0.0f;

    // Single pass with float4 vectorized loads
    int vec_cols = cols / 4;
    int remainder = cols % 4;
    const float4* row_ptr4 = reinterpret_cast<const float4*>(row_ptr);

    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE) {
        float4 v = row_ptr4[i];
        // Process each element through Welford update
        float vals[4] = {v.x, v.y, v.z, v.w};
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            count += 1.0f;
            float delta = vals[j] - mean;
            mean += delta / count;
            float delta2 = vals[j] - mean;
            M2 += delta * delta2;
        }
    }
    int base = vec_cols * 4;
    for (int i = threadIdx.x; i < remainder; i += BLOCK_SIZE) {
        float val = row_ptr[base + i];
        count += 1.0f;
        float delta = val - mean;
        mean += delta / count;
        float delta2 = val - mean;
        M2 += delta * delta2;
    }

    // Warp-level reduction
    warp_welford_reduce(count, mean, M2);

    // Inter-warp reduction via shared memory
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = BLOCK_SIZE / WARP_SIZE;

    __shared__ float s_count[BLOCK_SIZE / WARP_SIZE];
    __shared__ float s_mean[BLOCK_SIZE / WARP_SIZE];
    __shared__ float s_M2[BLOCK_SIZE / WARP_SIZE];

    if (lane == 0) {
        s_count[warp_id] = count;
        s_mean[warp_id] = mean;
        s_M2[warp_id] = M2;
    }
    __syncthreads();

    // First warp merges all warp results
    if (warp_id == 0) {
        count = (lane < num_warps) ? s_count[lane] : 0.0f;
        mean  = (lane < num_warps) ? s_mean[lane]  : 0.0f;
        M2    = (lane < num_warps) ? s_M2[lane]    : 0.0f;
        warp_welford_reduce(count, mean, M2);
    }

    if (threadIdx.x == 0) {
        // M2 = sum of squared deviations, v = sqrt(M2)
        output[row] = sqrtf(M2);
    }
}

// ============================================================================
// V3: Combined Welford + Fused Normalize + 512 Threads
// Best-of-all-worlds: single-pass, fused normalize, wider blocks.
// ============================================================================
constexpr int BLOCK_SIZE_512 = 512;

// Warp Welford for 512-thread blocks (same logic, separate function for clarity)
__device__ __forceinline__ void warp_welford_reduce_512(float& count, float& mean, float& M2) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float o_count = __shfl_down_sync(0xFFFFFFFF, count, offset);
        float o_mean  = __shfl_down_sync(0xFFFFFFFF, mean, offset);
        float o_M2    = __shfl_down_sync(0xFFFFFFFF, M2, offset);
        float n = count + o_count;
        if (n > 0.0f) {
            float delta = o_mean - mean;
            float new_mean = mean + delta * o_count / n;
            M2 = M2 + o_M2 + delta * delta * count * o_count / n;
            mean = new_mean;
            count = n;
        }
    }
}

__global__ void denominator_normalize_welford_512_fp32_kernel(
    const float* __restrict__ x,         // [rows, h]
    float* __restrict__ raw_output,      // [rows, out_dim] - modified in-place
    const float* __restrict__ b_new,     // [out_dim]
    int rows,
    int h,
    int out_dim,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* x_row = x + row * h;
    float* out_row = raw_output + row * out_dim;

    // Welford single-pass reduction
    float count = 0.0f;
    float mean = 0.0f;
    float M2 = 0.0f;

    int vec_cols = h / 4;
    int remainder = h % 4;
    const float4* x_row4 = reinterpret_cast<const float4*>(x_row);

    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE_512) {
        float4 v = x_row4[i];
        float vals[4] = {v.x, v.y, v.z, v.w};
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            count += 1.0f;
            float delta = vals[j] - mean;
            mean += delta / count;
            float delta2 = vals[j] - mean;
            M2 += delta * delta2;
        }
    }
    int base = vec_cols * 4;
    for (int i = threadIdx.x; i < remainder; i += BLOCK_SIZE_512) {
        float val = x_row[base + i];
        count += 1.0f;
        float delta = val - mean;
        mean += delta / count;
        float delta2 = val - mean;
        M2 += delta * delta2;
    }

    // Warp-level Welford reduction
    warp_welford_reduce_512(count, mean, M2);

    // Inter-warp reduction via shared memory (16 warps for 512 threads)
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    constexpr int NUM_WARPS_512 = BLOCK_SIZE_512 / WARP_SIZE;  // 16

    __shared__ float s_count[NUM_WARPS_512];
    __shared__ float s_mean[NUM_WARPS_512];
    __shared__ float s_M2[NUM_WARPS_512];

    if (lane == 0) {
        s_count[warp_id] = count;
        s_mean[warp_id] = mean;
        s_M2[warp_id] = M2;
    }
    __syncthreads();

    // First warp merges all 16 warp results
    if (warp_id == 0) {
        count = (lane < NUM_WARPS_512) ? s_count[lane] : 0.0f;
        mean  = (lane < NUM_WARPS_512) ? s_mean[lane]  : 0.0f;
        M2    = (lane < NUM_WARPS_512) ? s_M2[lane]    : 0.0f;
        warp_welford_reduce_512(count, mean, M2);
    }

    // Broadcast std to all threads
    __shared__ float std_shared;
    if (threadIdx.x == 0) {
        std_shared = sqrtf(M2 / (float)h + eps);
    }
    __syncthreads();
    float inv_std = 1.0f / std_shared;

    // Normalize raw_output in-place with float4 vectorization
    int out_vec = out_dim / 4;
    int out_rem = out_dim % 4;
    float4* out_row4 = reinterpret_cast<float4*>(out_row);
    const float4* b_new4 = reinterpret_cast<const float4*>(b_new);

    for (int i = threadIdx.x; i < out_vec; i += BLOCK_SIZE_512) {
        float4 o = out_row4[i];
        float4 b = b_new4[i];
        o.x = o.x * inv_std + b.x;
        o.y = o.y * inv_std + b.y;
        o.z = o.z * inv_std + b.z;
        o.w = o.w * inv_std + b.w;
        out_row4[i] = o;
    }
    int out_base = out_vec * 4;
    for (int i = threadIdx.x; i < out_rem; i += BLOCK_SIZE_512) {
        int c = out_base + i;
        out_row[c] = out_row[c] * inv_std + b_new[c];
    }
}

// ============================================================================
// V3 FP16: Welford + fused normalize + 512 threads, half input/output
// ============================================================================
__global__ void denominator_normalize_welford_512_fp16_kernel(
    const __half* __restrict__ x,         // [rows, h]
    __half* __restrict__ raw_output,      // [rows, out_dim]
    const __half* __restrict__ b_new,     // [out_dim]
    int rows,
    int h,
    int out_dim,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const __half* x_row = x + row * h;
    __half* out_row = raw_output + row * out_dim;

    // Welford single-pass (fp32 accumulation)
    float count = 0.0f;
    float mean = 0.0f;
    float M2 = 0.0f;

    int vec_cols = h / 2;
    int remainder = h % 2;
    const __half2* x_row2 = reinterpret_cast<const __half2*>(x_row);

    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE_512) {
        __half2 v = x_row2[i];
        float vals[2] = {__half2float(v.x), __half2float(v.y)};
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            count += 1.0f;
            float delta = vals[j] - mean;
            mean += delta / count;
            float delta2 = vals[j] - mean;
            M2 += delta * delta2;
        }
    }
    if (remainder && threadIdx.x == 0) {
        float val = __half2float(x_row[h - 1]);
        count += 1.0f;
        float delta = val - mean;
        mean += delta / count;
        float delta2 = val - mean;
        M2 += delta * delta2;
    }

    warp_welford_reduce_512(count, mean, M2);

    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    constexpr int NUM_WARPS_512 = BLOCK_SIZE_512 / WARP_SIZE;

    __shared__ float s_count[NUM_WARPS_512];
    __shared__ float s_mean[NUM_WARPS_512];
    __shared__ float s_M2[NUM_WARPS_512];

    if (lane == 0) {
        s_count[warp_id] = count;
        s_mean[warp_id] = mean;
        s_M2[warp_id] = M2;
    }
    __syncthreads();

    if (warp_id == 0) {
        count = (lane < NUM_WARPS_512) ? s_count[lane] : 0.0f;
        mean  = (lane < NUM_WARPS_512) ? s_mean[lane]  : 0.0f;
        M2    = (lane < NUM_WARPS_512) ? s_M2[lane]    : 0.0f;
        warp_welford_reduce_512(count, mean, M2);
    }

    __shared__ float std_shared;
    if (threadIdx.x == 0) {
        std_shared = sqrtf(M2 / (float)h + eps);
    }
    __syncthreads();
    float inv_std = 1.0f / std_shared;

    // Normalize with half2 vectorization
    int out_vec = out_dim / 2;
    int out_rem = out_dim % 2;
    __half2* out_row2 = reinterpret_cast<__half2*>(out_row);
    const __half2* b_new2 = reinterpret_cast<const __half2*>(b_new);

    for (int i = threadIdx.x; i < out_vec; i += BLOCK_SIZE_512) {
        __half2 o = out_row2[i];
        __half2 b = b_new2[i];
        float o0 = __half2float(o.x) * inv_std + __half2float(b.x);
        float o1 = __half2float(o.y) * inv_std + __half2float(b.y);
        out_row2[i] = __halves2half2(__float2half(o0), __float2half(o1));
    }
    int out_base = out_vec * 2;
    for (int i = threadIdx.x; i < out_rem; i += BLOCK_SIZE_512) {
        int c = out_base + i;
        float val = __half2float(out_row[c]) * inv_std + __half2float(b_new[c]);
        out_row[c] = __float2half(val);
    }
}

// ============================================================================
// V3 BF16: Welford + fused normalize + 512 threads, bfloat16 input/output
// ============================================================================
__global__ void denominator_normalize_welford_512_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,         // [rows, h]
    __nv_bfloat16* __restrict__ raw_output,      // [rows, out_dim]
    const __nv_bfloat16* __restrict__ b_new,     // [out_dim]
    int rows,
    int h,
    int out_dim,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const __nv_bfloat16* x_row = x + row * h;
    __nv_bfloat16* out_row = raw_output + row * out_dim;

    // Welford single-pass (fp32 accumulation)
    float count = 0.0f;
    float mean = 0.0f;
    float M2 = 0.0f;

    int vec_cols = h / 2;
    int remainder = h % 2;
    const __nv_bfloat162* x_row2 = reinterpret_cast<const __nv_bfloat162*>(x_row);

    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE_512) {
        __nv_bfloat162 v = x_row2[i];
        float vals[2] = {__bfloat162float(v.x), __bfloat162float(v.y)};
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            count += 1.0f;
            float delta = vals[j] - mean;
            mean += delta / count;
            float delta2 = vals[j] - mean;
            M2 += delta * delta2;
        }
    }
    if (remainder && threadIdx.x == 0) {
        float val = __bfloat162float(x_row[h - 1]);
        count += 1.0f;
        float delta = val - mean;
        mean += delta / count;
        float delta2 = val - mean;
        M2 += delta * delta2;
    }

    warp_welford_reduce_512(count, mean, M2);

    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    constexpr int NUM_WARPS_512 = BLOCK_SIZE_512 / WARP_SIZE;

    __shared__ float s_count[NUM_WARPS_512];
    __shared__ float s_mean[NUM_WARPS_512];
    __shared__ float s_M2[NUM_WARPS_512];

    if (lane == 0) {
        s_count[warp_id] = count;
        s_mean[warp_id] = mean;
        s_M2[warp_id] = M2;
    }
    __syncthreads();

    if (warp_id == 0) {
        count = (lane < NUM_WARPS_512) ? s_count[lane] : 0.0f;
        mean  = (lane < NUM_WARPS_512) ? s_mean[lane]  : 0.0f;
        M2    = (lane < NUM_WARPS_512) ? s_M2[lane]    : 0.0f;
        warp_welford_reduce_512(count, mean, M2);
    }

    __shared__ float std_shared;
    if (threadIdx.x == 0) {
        std_shared = sqrtf(M2 / (float)h + eps);
    }
    __syncthreads();
    float inv_std = 1.0f / std_shared;

    // Normalize with bfloat162 vectorization
    int out_vec = out_dim / 2;
    int out_rem = out_dim % 2;
    __nv_bfloat162* out_row2 = reinterpret_cast<__nv_bfloat162*>(out_row);
    const __nv_bfloat162* b_new2 = reinterpret_cast<const __nv_bfloat162*>(b_new);

    for (int i = threadIdx.x; i < out_vec; i += BLOCK_SIZE_512) {
        __nv_bfloat162 o = out_row2[i];
        __nv_bfloat162 b = b_new2[i];
        float o0 = __bfloat162float(o.x) * inv_std + __bfloat162float(b.x);
        float o1 = __bfloat162float(o.y) * inv_std + __bfloat162float(b.y);
        out_row2[i] = __halves2bfloat162(__float2bfloat16(o0), __float2bfloat16(o1));
    }
    int out_base = out_vec * 2;
    for (int i = threadIdx.x; i < out_rem; i += BLOCK_SIZE_512) {
        int c = out_base + i;
        float val = __bfloat162float(out_row[c]) * inv_std + __bfloat162float(b_new[c]);
        out_row[c] = __float2bfloat16(val);
    }
}

// ============================================================================
// RMSNorm kernels
// RMSNorm: output = x / rms(x) * gamma, rms(x) = sqrt(mean(x^2) + eps)
// No mean subtraction — simpler than LayerNorm. Single pass suffices.
// The weight absorption is done on CPU: W_new = W * gamma
// So at runtime we compute: raw_output / rms(x) + b_new
// ============================================================================

// V1 RMSNorm FP32: single pass, 256 threads
__global__ void rmsnorm_normalize_fp32_kernel(
    const float* __restrict__ x,         // [rows, h]
    float* __restrict__ raw_output,      // [rows, out_dim] - modified in-place
    const float* __restrict__ b_new,     // [out_dim]
    int rows,
    int h,
    int out_dim,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* x_row = x + row * h;
    float* out_row = raw_output + row * out_dim;
    __shared__ float smem[BLOCK_SIZE / WARP_SIZE];

    // Single pass: sum of squares
    float local_sq = 0.0f;
    int vec_cols = h / 4;
    int remainder = h % 4;
    const float4* x_row4 = reinterpret_cast<const float4*>(x_row);

    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE) {
        float4 v = x_row4[i];
        local_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    int base = vec_cols * 4;
    for (int i = threadIdx.x; i < remainder; i += BLOCK_SIZE) {
        float val = x_row[base + i];
        local_sq += val * val;
    }

    float total_sq = block_reduce_sum(local_sq, smem);

    // rms = sqrt(mean(x^2) + eps)
    __shared__ float rms_shared;
    if (threadIdx.x == 0) {
        rms_shared = sqrtf(total_sq / (float)h + eps);
    }
    __syncthreads();
    float inv_rms = 1.0f / rms_shared;

    // Normalize raw_output in-place
    int out_vec = out_dim / 4;
    int out_rem = out_dim % 4;
    float4* out_row4 = reinterpret_cast<float4*>(out_row);
    const float4* b_new4 = reinterpret_cast<const float4*>(b_new);

    for (int i = threadIdx.x; i < out_vec; i += BLOCK_SIZE) {
        float4 o = out_row4[i];
        float4 b = b_new4[i];
        o.x = o.x * inv_rms + b.x;
        o.y = o.y * inv_rms + b.y;
        o.z = o.z * inv_rms + b.z;
        o.w = o.w * inv_rms + b.w;
        out_row4[i] = o;
    }
    int out_base = out_vec * 4;
    for (int i = threadIdx.x; i < out_rem; i += BLOCK_SIZE) {
        int c = out_base + i;
        out_row[c] = out_row[c] * inv_rms + b_new[c];
    }
}

// V1 RMSNorm FP16
__global__ void rmsnorm_normalize_fp16_kernel(
    const __half* __restrict__ x,
    __half* __restrict__ raw_output,
    const __half* __restrict__ b_new,
    int rows,
    int h,
    int out_dim,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const __half* x_row = x + row * h;
    __half* out_row = raw_output + row * out_dim;
    __shared__ float smem[BLOCK_SIZE / WARP_SIZE];

    float local_sq = 0.0f;
    int vec_cols = h / 2;
    int remainder = h % 2;
    const __half2* x_row2 = reinterpret_cast<const __half2*>(x_row);

    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE) {
        __half2 v = x_row2[i];
        float v0 = __half2float(v.x), v1 = __half2float(v.y);
        local_sq += v0 * v0 + v1 * v1;
    }
    if (remainder && threadIdx.x == 0) {
        float v = __half2float(x_row[h - 1]);
        local_sq += v * v;
    }

    float total_sq = block_reduce_sum(local_sq, smem);

    __shared__ float rms_shared;
    if (threadIdx.x == 0) {
        rms_shared = sqrtf(total_sq / (float)h + eps);
    }
    __syncthreads();
    float inv_rms = 1.0f / rms_shared;

    int out_vec = out_dim / 2;
    int out_rem = out_dim % 2;
    __half2* out_row2 = reinterpret_cast<__half2*>(out_row);
    const __half2* b_new2 = reinterpret_cast<const __half2*>(b_new);

    for (int i = threadIdx.x; i < out_vec; i += BLOCK_SIZE) {
        __half2 o = out_row2[i];
        __half2 b = b_new2[i];
        float o0 = __half2float(o.x) * inv_rms + __half2float(b.x);
        float o1 = __half2float(o.y) * inv_rms + __half2float(b.y);
        out_row2[i] = __halves2half2(__float2half(o0), __float2half(o1));
    }
    int out_base = out_vec * 2;
    for (int i = threadIdx.x; i < out_rem; i += BLOCK_SIZE) {
        int c = out_base + i;
        float val = __half2float(out_row[c]) * inv_rms + __half2float(b_new[c]);
        out_row[c] = __float2half(val);
    }
}

// V1 RMSNorm BF16
__global__ void rmsnorm_normalize_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ raw_output,
    const __nv_bfloat16* __restrict__ b_new,
    int rows,
    int h,
    int out_dim,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const __nv_bfloat16* x_row = x + row * h;
    __nv_bfloat16* out_row = raw_output + row * out_dim;
    __shared__ float smem[BLOCK_SIZE / WARP_SIZE];

    float local_sq = 0.0f;
    int vec_cols = h / 2;
    int remainder = h % 2;
    const __nv_bfloat162* x_row2 = reinterpret_cast<const __nv_bfloat162*>(x_row);

    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE) {
        __nv_bfloat162 v = x_row2[i];
        float v0 = __bfloat162float(v.x), v1 = __bfloat162float(v.y);
        local_sq += v0 * v0 + v1 * v1;
    }
    if (remainder && threadIdx.x == 0) {
        float v = __bfloat162float(x_row[h - 1]);
        local_sq += v * v;
    }

    float total_sq = block_reduce_sum(local_sq, smem);

    __shared__ float rms_shared;
    if (threadIdx.x == 0) {
        rms_shared = sqrtf(total_sq / (float)h + eps);
    }
    __syncthreads();
    float inv_rms = 1.0f / rms_shared;

    int out_vec = out_dim / 2;
    int out_rem = out_dim % 2;
    __nv_bfloat162* out_row2 = reinterpret_cast<__nv_bfloat162*>(out_row);
    const __nv_bfloat162* b_new2 = reinterpret_cast<const __nv_bfloat162*>(b_new);

    for (int i = threadIdx.x; i < out_vec; i += BLOCK_SIZE) {
        __nv_bfloat162 o = out_row2[i];
        __nv_bfloat162 b = b_new2[i];
        float o0 = __bfloat162float(o.x) * inv_rms + __bfloat162float(b.x);
        float o1 = __bfloat162float(o.y) * inv_rms + __bfloat162float(b.y);
        out_row2[i] = __halves2bfloat162(__float2bfloat16(o0), __float2bfloat16(o1));
    }
    int out_base = out_vec * 2;
    for (int i = threadIdx.x; i < out_rem; i += BLOCK_SIZE) {
        int c = out_base + i;
        float val = __bfloat162float(out_row[c]) * inv_rms + __bfloat162float(b_new[c]);
        out_row[c] = __float2bfloat16(val);
    }
}

// V3 RMSNorm FP32: 512 threads
__global__ void rmsnorm_normalize_512_fp32_kernel(
    const float* __restrict__ x,
    float* __restrict__ raw_output,
    const float* __restrict__ b_new,
    int rows,
    int h,
    int out_dim,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* x_row = x + row * h;
    float* out_row = raw_output + row * out_dim;
    __shared__ float smem[BLOCK_SIZE_512 / WARP_SIZE];

    float local_sq = 0.0f;
    int vec_cols = h / 4;
    int remainder = h % 4;
    const float4* x_row4 = reinterpret_cast<const float4*>(x_row);

    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE_512) {
        float4 v = x_row4[i];
        local_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    int base = vec_cols * 4;
    for (int i = threadIdx.x; i < remainder; i += BLOCK_SIZE_512) {
        float val = x_row[base + i];
        local_sq += val * val;
    }

    // Block reduce with 512-thread shared memory
    float total_sq;
    {
        int lane = threadIdx.x % WARP_SIZE;
        int warp_id = threadIdx.x / WARP_SIZE;
        local_sq = warp_reduce_sum(local_sq);
        if (lane == 0) smem[warp_id] = local_sq;
        __syncthreads();
        constexpr int NUM_WARPS = BLOCK_SIZE_512 / WARP_SIZE;
        if (warp_id == 0) {
            local_sq = (lane < NUM_WARPS) ? smem[lane] : 0.0f;
            local_sq = warp_reduce_sum(local_sq);
        }
        total_sq = local_sq;
    }

    __shared__ float rms_shared;
    if (threadIdx.x == 0) {
        rms_shared = sqrtf(total_sq / (float)h + eps);
    }
    __syncthreads();
    float inv_rms = 1.0f / rms_shared;

    int out_vec = out_dim / 4;
    int out_rem = out_dim % 4;
    float4* out_row4 = reinterpret_cast<float4*>(out_row);
    const float4* b_new4 = reinterpret_cast<const float4*>(b_new);

    for (int i = threadIdx.x; i < out_vec; i += BLOCK_SIZE_512) {
        float4 o = out_row4[i];
        float4 b = b_new4[i];
        o.x = o.x * inv_rms + b.x;
        o.y = o.y * inv_rms + b.y;
        o.z = o.z * inv_rms + b.z;
        o.w = o.w * inv_rms + b.w;
        out_row4[i] = o;
    }
    int out_base = out_vec * 4;
    for (int i = threadIdx.x; i < out_rem; i += BLOCK_SIZE_512) {
        int c = out_base + i;
        out_row[c] = out_row[c] * inv_rms + b_new[c];
    }
}

// V3 RMSNorm FP16: 512 threads
__global__ void rmsnorm_normalize_512_fp16_kernel(
    const __half* __restrict__ x,
    __half* __restrict__ raw_output,
    const __half* __restrict__ b_new,
    int rows,
    int h,
    int out_dim,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const __half* x_row = x + row * h;
    __half* out_row = raw_output + row * out_dim;
    __shared__ float smem[BLOCK_SIZE_512 / WARP_SIZE];

    float local_sq = 0.0f;
    int vec_cols = h / 2;
    int remainder = h % 2;
    const __half2* x_row2 = reinterpret_cast<const __half2*>(x_row);

    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE_512) {
        __half2 v = x_row2[i];
        float v0 = __half2float(v.x), v1 = __half2float(v.y);
        local_sq += v0 * v0 + v1 * v1;
    }
    if (remainder && threadIdx.x == 0) {
        float v = __half2float(x_row[h - 1]);
        local_sq += v * v;
    }

    float total_sq;
    {
        int lane = threadIdx.x % WARP_SIZE;
        int warp_id = threadIdx.x / WARP_SIZE;
        local_sq = warp_reduce_sum(local_sq);
        if (lane == 0) smem[warp_id] = local_sq;
        __syncthreads();
        constexpr int NUM_WARPS = BLOCK_SIZE_512 / WARP_SIZE;
        if (warp_id == 0) {
            local_sq = (lane < NUM_WARPS) ? smem[lane] : 0.0f;
            local_sq = warp_reduce_sum(local_sq);
        }
        total_sq = local_sq;
    }

    __shared__ float rms_shared;
    if (threadIdx.x == 0) {
        rms_shared = sqrtf(total_sq / (float)h + eps);
    }
    __syncthreads();
    float inv_rms = 1.0f / rms_shared;

    int out_vec = out_dim / 2;
    int out_rem = out_dim % 2;
    __half2* out_row2 = reinterpret_cast<__half2*>(out_row);
    const __half2* b_new2 = reinterpret_cast<const __half2*>(b_new);

    for (int i = threadIdx.x; i < out_vec; i += BLOCK_SIZE_512) {
        __half2 o = out_row2[i];
        __half2 b = b_new2[i];
        float o0 = __half2float(o.x) * inv_rms + __half2float(b.x);
        float o1 = __half2float(o.y) * inv_rms + __half2float(b.y);
        out_row2[i] = __halves2half2(__float2half(o0), __float2half(o1));
    }
    int out_base = out_vec * 2;
    for (int i = threadIdx.x; i < out_rem; i += BLOCK_SIZE_512) {
        int c = out_base + i;
        float val = __half2float(out_row[c]) * inv_rms + __half2float(b_new[c]);
        out_row[c] = __float2half(val);
    }
}

// V3 RMSNorm BF16: 512 threads
__global__ void rmsnorm_normalize_512_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ raw_output,
    const __nv_bfloat16* __restrict__ b_new,
    int rows,
    int h,
    int out_dim,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const __nv_bfloat16* x_row = x + row * h;
    __nv_bfloat16* out_row = raw_output + row * out_dim;
    __shared__ float smem[BLOCK_SIZE_512 / WARP_SIZE];

    float local_sq = 0.0f;
    int vec_cols = h / 2;
    int remainder = h % 2;
    const __nv_bfloat162* x_row2 = reinterpret_cast<const __nv_bfloat162*>(x_row);

    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE_512) {
        __nv_bfloat162 v = x_row2[i];
        float v0 = __bfloat162float(v.x), v1 = __bfloat162float(v.y);
        local_sq += v0 * v0 + v1 * v1;
    }
    if (remainder && threadIdx.x == 0) {
        float v = __bfloat162float(x_row[h - 1]);
        local_sq += v * v;
    }

    float total_sq;
    {
        int lane = threadIdx.x % WARP_SIZE;
        int warp_id = threadIdx.x / WARP_SIZE;
        local_sq = warp_reduce_sum(local_sq);
        if (lane == 0) smem[warp_id] = local_sq;
        __syncthreads();
        constexpr int NUM_WARPS = BLOCK_SIZE_512 / WARP_SIZE;
        if (warp_id == 0) {
            local_sq = (lane < NUM_WARPS) ? smem[lane] : 0.0f;
            local_sq = warp_reduce_sum(local_sq);
        }
        total_sq = local_sq;
    }

    __shared__ float rms_shared;
    if (threadIdx.x == 0) {
        rms_shared = sqrtf(total_sq / (float)h + eps);
    }
    __syncthreads();
    float inv_rms = 1.0f / rms_shared;

    int out_vec = out_dim / 2;
    int out_rem = out_dim % 2;
    __nv_bfloat162* out_row2 = reinterpret_cast<__nv_bfloat162*>(out_row);
    const __nv_bfloat162* b_new2 = reinterpret_cast<const __nv_bfloat162*>(b_new);

    for (int i = threadIdx.x; i < out_vec; i += BLOCK_SIZE_512) {
        __nv_bfloat162 o = out_row2[i];
        __nv_bfloat162 b = b_new2[i];
        float o0 = __bfloat162float(o.x) * inv_rms + __bfloat162float(b.x);
        float o1 = __bfloat162float(o.y) * inv_rms + __bfloat162float(b.y);
        out_row2[i] = __halves2bfloat162(__float2bfloat16(o0), __float2bfloat16(o1));
    }
    int out_base = out_vec * 2;
    for (int i = threadIdx.x; i < out_rem; i += BLOCK_SIZE_512) {
        int c = out_base + i;
        float val = __bfloat162float(out_row[c]) * inv_rms + __bfloat162float(b_new[c]);
        out_row[c] = __float2bfloat16(val);
    }
}

// ============================================================================
// Host functions
// ============================================================================

// Host function: dispatches fp32 or fp16 kernel
torch::Tensor compute_denominator_cuda(
    torch::Tensor x,
    c10::optional<int64_t> stream_ptr
) {
    TORCH_CHECK(x.dim() == 2, "Input must be 2D");
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA");

    int rows = x.size(0);
    int cols = x.size(1);

    auto output = torch::empty({rows}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));

    cudaStream_t stream;
    if (stream_ptr.has_value()) {
        stream = reinterpret_cast<cudaStream_t>(stream_ptr.value());
    } else {
        stream = at::cuda::getCurrentCUDAStream().stream();
    }

    dim3 grid(rows);
    dim3 block(BLOCK_SIZE);

    if (x.scalar_type() == torch::kFloat32) {
        denominator_fp32_kernel<<<grid, block, 0, stream>>>(
            x.data_ptr<float>(),
            output.data_ptr<float>(),
            rows, cols
        );
    } else if (x.scalar_type() == torch::kFloat16) {
        denominator_fp16_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            output.data_ptr<float>(),
            rows, cols
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype: only float32 and float16 are supported");
    }

    return output;
}

// V1: Fused denominator + normalize (in-place on raw_output)
void denominator_normalize_cuda(
    torch::Tensor x,
    torch::Tensor raw_output,
    torch::Tensor b_new,
    int h,
    float eps
) {
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(raw_output.dim() == 2, "raw_output must be 2D");
    TORCH_CHECK(x.is_cuda() && raw_output.is_cuda() && b_new.is_cuda(), "All tensors must be on CUDA");

    int rows = x.size(0);
    int out_dim = raw_output.size(1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 grid(rows);
    dim3 block(BLOCK_SIZE);

    if (x.scalar_type() == torch::kFloat32) {
        denominator_normalize_fp32_kernel<<<grid, block, 0, stream>>>(
            x.data_ptr<float>(),
            raw_output.data_ptr<float>(),
            b_new.data_ptr<float>(),
            rows, h, out_dim, eps
        );
    } else if (x.scalar_type() == torch::kFloat16) {
        denominator_normalize_fp16_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(raw_output.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(b_new.data_ptr<at::Half>()),
            rows, h, out_dim, eps
        );
    } else if (x.scalar_type() == torch::kBFloat16) {
        denominator_normalize_bf16_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(raw_output.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(b_new.data_ptr<at::BFloat16>()),
            rows, h, out_dim, eps
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype for V1: only fp32, fp16, bf16 supported");
    }
}

// V2: Welford's single-pass denominator
torch::Tensor compute_denominator_welford_cuda(
    torch::Tensor x,
    c10::optional<int64_t> stream_ptr
) {
    TORCH_CHECK(x.dim() == 2, "Input must be 2D");
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Only fp32 supported for Welford");

    int rows = x.size(0);
    int cols = x.size(1);

    auto output = torch::empty({rows}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));

    cudaStream_t stream;
    if (stream_ptr.has_value()) {
        stream = reinterpret_cast<cudaStream_t>(stream_ptr.value());
    } else {
        stream = at::cuda::getCurrentCUDAStream().stream();
    }

    dim3 grid(rows);
    dim3 block(BLOCK_SIZE);

    denominator_welford_fp32_kernel<<<grid, block, 0, stream>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        rows, cols
    );

    return output;
}

// V3: Combined Welford + fused normalize + 512 threads
void denominator_normalize_welford_512_cuda(
    torch::Tensor x,
    torch::Tensor raw_output,
    torch::Tensor b_new,
    int h,
    float eps
) {
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(raw_output.dim() == 2, "raw_output must be 2D");
    TORCH_CHECK(x.is_cuda() && raw_output.is_cuda() && b_new.is_cuda(), "All tensors must be on CUDA");

    int rows = x.size(0);
    int out_dim = raw_output.size(1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 grid(rows);
    dim3 block(BLOCK_SIZE_512);

    if (x.scalar_type() == torch::kFloat32) {
        denominator_normalize_welford_512_fp32_kernel<<<grid, block, 0, stream>>>(
            x.data_ptr<float>(),
            raw_output.data_ptr<float>(),
            b_new.data_ptr<float>(),
            rows, h, out_dim, eps
        );
    } else if (x.scalar_type() == torch::kFloat16) {
        denominator_normalize_welford_512_fp16_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(raw_output.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(b_new.data_ptr<at::Half>()),
            rows, h, out_dim, eps
        );
    } else if (x.scalar_type() == torch::kBFloat16) {
        denominator_normalize_welford_512_bf16_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(raw_output.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(b_new.data_ptr<at::BFloat16>()),
            rows, h, out_dim, eps
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype for V3: only fp32, fp16, bf16 supported");
    }
}

// RMSNorm V1: fused normalize (256 threads)
void rmsnorm_normalize_cuda(
    torch::Tensor x,
    torch::Tensor raw_output,
    torch::Tensor b_new,
    int h,
    float eps
) {
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(raw_output.dim() == 2, "raw_output must be 2D");
    TORCH_CHECK(x.is_cuda() && raw_output.is_cuda() && b_new.is_cuda(), "All tensors must be on CUDA");

    int rows = x.size(0);
    int out_dim = raw_output.size(1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 grid(rows);
    dim3 block(BLOCK_SIZE);

    if (x.scalar_type() == torch::kFloat32) {
        rmsnorm_normalize_fp32_kernel<<<grid, block, 0, stream>>>(
            x.data_ptr<float>(),
            raw_output.data_ptr<float>(),
            b_new.data_ptr<float>(),
            rows, h, out_dim, eps
        );
    } else if (x.scalar_type() == torch::kFloat16) {
        rmsnorm_normalize_fp16_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(raw_output.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(b_new.data_ptr<at::Half>()),
            rows, h, out_dim, eps
        );
    } else if (x.scalar_type() == torch::kBFloat16) {
        rmsnorm_normalize_bf16_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(raw_output.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(b_new.data_ptr<at::BFloat16>()),
            rows, h, out_dim, eps
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype for RMSNorm V1: only fp32, fp16, bf16 supported");
    }
}

// RMSNorm V3: fused normalize (512 threads)
void rmsnorm_normalize_512_cuda(
    torch::Tensor x,
    torch::Tensor raw_output,
    torch::Tensor b_new,
    int h,
    float eps
) {
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(raw_output.dim() == 2, "raw_output must be 2D");
    TORCH_CHECK(x.is_cuda() && raw_output.is_cuda() && b_new.is_cuda(), "All tensors must be on CUDA");

    int rows = x.size(0);
    int out_dim = raw_output.size(1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 grid(rows);
    dim3 block(BLOCK_SIZE_512);

    if (x.scalar_type() == torch::kFloat32) {
        rmsnorm_normalize_512_fp32_kernel<<<grid, block, 0, stream>>>(
            x.data_ptr<float>(),
            raw_output.data_ptr<float>(),
            b_new.data_ptr<float>(),
            rows, h, out_dim, eps
        );
    } else if (x.scalar_type() == torch::kFloat16) {
        rmsnorm_normalize_512_fp16_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(raw_output.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(b_new.data_ptr<at::Half>()),
            rows, h, out_dim, eps
        );
    } else if (x.scalar_type() == torch::kBFloat16) {
        rmsnorm_normalize_512_bf16_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(raw_output.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(b_new.data_ptr<at::BFloat16>()),
            rows, h, out_dim, eps
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype for RMSNorm V3: only fp32, fp16, bf16 supported");
    }
}
