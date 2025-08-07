#include <metal_atomic>
#include <metal_compute>
#include <metal_integer>
#include <metal_math>
#include <metal_simdgroup>

#include <internal/kernel-args.h>

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)


// Each simdgroup reduces all channels of the input and computes a single channel of the output
// + Efficient synchronization
// + Sequential memory access within a warp
// Each threadgroup computes (simdgroups_per_threadgroup) consecutive output channels
// + Reuse input vector from threadgroup memory
// + Avoid synchronization across warps when doing reduction

kernel void gptoss_f32_bf16w_matmul(
    constant gptoss_matmul_args& args [[ buffer(0) ]],
    const device float4* input [[ buffer(1) ]],
    const device bfloat4* weight [[ buffer(2) ]],
    const device bfloat* bias [[ buffer(3) ]],
    device float* output [[ buffer(4) ]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint simdgroup_tid [[thread_index_in_simdgroup]],
    uint simdgroup_idx [[simdgroup_index_in_threadgroup]],
    uint num_simdgroups [[simdgroups_per_threadgroup]])
{
    const uint simdgroup_size = 32;

    const uint num_column_vecs = args.num_column_vecs;
    const uint row = gid.x * num_simdgroups + simdgroup_idx;

    input += gid.y * num_column_vecs + simdgroup_tid;
    weight += num_column_vecs * row + simdgroup_tid;
    bias += row;
    output += gid.y * args.num_rows + row;

    uint num_iter = (num_column_vecs - simdgroup_tid + (simdgroup_size - 1)) / simdgroup_size;

    float4 sum4 = 0.0f;
    do {
        const bfloat4 w = *weight;
        const float4 i = *input;
        sum4 = metal::fma(static_cast<float4>(w), i, sum4);

        weight += simdgroup_size;
        input += simdgroup_size;
    } while (--num_iter != 0);
    const float2 sum2 = sum4.xy + sum4.zw;
    float sum = sum2.x + sum2.y;
    sum = metal::simd_sum(sum);
    if (metal::simd_is_first()) {
        sum += static_cast<float>(*bias);
        if (args.add) {
            *output += sum;
        } else {
            *output = sum;
        }
    }
}

kernel void gptoss_f32_bf16w_unembedding(
    constant gptoss_unembedding_args& args [[ buffer(0) ]],
    const device float4* input [[ buffer(1) ]],
    const device bfloat4* weight [[ buffer(2) ]],
    device float* output [[ buffer(3) ]],
    device metal::atomic_ulong* argmax [[ buffer(4) ]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint simdgroup_tid [[thread_index_in_simdgroup]],
    uint simdgroup_idx [[simdgroup_index_in_threadgroup]],
    uint num_simdgroups [[simdgroups_per_threadgroup]])
{
    const uint simdgroup_size = 32;
    threadgroup uint2 threadgroup_buffer[32];

    const uint num_column_vecs = args.num_column_vecs;
    const uint row_start = gid.x * args.num_rows_per_threadgroup + simdgroup_idx;
    const uint row_end = metal::min(gid.x * args.num_rows_per_threadgroup + args.num_rows_per_threadgroup, args.num_rows);
    const uint num_iter = (num_column_vecs - simdgroup_tid + (simdgroup_size - 1)) / simdgroup_size;

    input += gid.y * num_column_vecs + simdgroup_tid;
    weight += num_column_vecs * row_start + simdgroup_tid;
    output += gid.y * args.num_rows + row_start;

    uint2 row_sum{0xFFFFFFFFul, 0xFFFFFFFFul};
    for (uint row = row_start; row < row_end; row += num_simdgroups) {
        uint n = num_iter;

        float4 sum4 = 0.0f;
        do {
            const bfloat4 w = *weight;
            const float4 i = *input;

            sum4 = metal::fma(static_cast<float4>(w), i, sum4);

            weight += simdgroup_size;
            input += simdgroup_size;
        } while (--n != 0);
        input -= num_iter * simdgroup_size;
        weight -= num_iter * simdgroup_size;

        const float2 sum2 = sum4.xy + sum4.zw;
        float sum = sum2.x + sum2.y;
        sum = metal::simd_sum(sum);
        uint sum_bits = as_type<uint>(sum);
        if (static_cast<int>(sum_bits) >= 0) {
            sum_bits ^= 0x7FFFFFFFu;
        }
        row_sum = as_type<uint2>(metal::min(as_type<ulong>(row_sum), as_type<ulong>(uint2{row, sum_bits})));
        if (metal::simd_is_first()) {
            *output = sum;
        }

        weight += num_column_vecs * num_simdgroups;
        output += num_simdgroups;
    }
    if (metal::simd_is_first()) {
        threadgroup_buffer[simdgroup_idx] = row_sum;
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    if (simdgroup_idx == 0) {
        // Min-Reduce threadgroup_buffer
        if (simdgroup_tid < num_simdgroups) {
            row_sum = threadgroup_buffer[simdgroup_tid];
        }
        const uint sum_bits = row_sum.y;
        const uint sum_bits_min = metal::simd_min(sum_bits);
        const uint row_min = metal::simd_min(sum_bits == sum_bits_min ? row_sum.x : 0xFFFFFFFFu);
        if (metal::simd_is_first()) {
            const uint2 threadgroup_output{row_min, sum_bits_min};
            atomic_min_explicit(&argmax[gid.y], as_type<ulong>(threadgroup_output), metal::memory_order_relaxed);
        }
    }
}
