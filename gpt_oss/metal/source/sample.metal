#include <metal_compute>
#include <metal_integer>
#include <metal_math>
#include <metal_simdgroup>

#include <internal/kernel-args.h>

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)


kernel void gptoss_f32_softmax(
    constant gptoss_softmax_args& args [[ buffer(0) ]],
    const device float* score [[ buffer(1) ]],
    const device uint2* argmax [[ buffer(2) ]],
    device float* prob [[ buffer(3) ]],
    device float* sum [[ buffer(4) ]],
    uint tidx [[thread_index_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 threadgroup_size [[threads_per_threadgroup]],
    uint simdgroup_tid [[thread_index_in_simdgroup]],
    uint simdgroup_idx [[simdgroup_index_in_threadgroup]],
    uint num_simdgroups [[simdgroups_per_threadgroup]])
{
    threadgroup float threadgroup_sumexp[32];

    score += gid.y * args.num_vecs + gid.x * args.num_vecs_per_threadgroup;
    prob += gid.y * args.num_vecs + gid.x * args.num_vecs_per_threadgroup;
    sum += gid.y * args.max_threadgroups;

    uint max_bits = argmax[gid.y].y;
    if (static_cast<int>(max_bits) >= 0) {
        max_bits ^= 0x7FFFFFFFu;
    }
    const float max_val = as_type<float>(max_bits);
    float sum_exp = 0.0f;
    const uint num_vecs_per_threadgroup = metal::min(args.num_vecs - gid.x * args.num_vecs_per_threadgroup, args.num_vecs_per_threadgroup);
    for (uint i = tidx; i < num_vecs_per_threadgroup; i += threadgroup_size.x) {
        const float score_val = score[i];
        const float prob_val = metal::precise::exp((score_val - max_val) * args.temperature);
        prob[i] = prob_val;
        sum_exp += prob_val;
    }
    sum_exp = metal::simd_sum(sum_exp);
    if (metal::simd_is_first()) {
        threadgroup_sumexp[simdgroup_idx] = sum_exp;
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    if (simdgroup_idx == 0) {
        // Sum-Reduce threadgroup_sumexp
        sum_exp = 0.0f;
        if (simdgroup_tid < num_simdgroups) {
            sum_exp = threadgroup_sumexp[simdgroup_tid];
        }
        sum_exp = metal::simd_sum(sum_exp);
        if (metal::simd_is_first()) {
            sum[gid.x] = sum_exp;
        }
    }
}
