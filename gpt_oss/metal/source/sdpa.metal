#include <metal_geometric>
#include <metal_integer>
#include <metal_math>
#include <metal_compute>
#include <metal_simdgroup>

#include <internal/kernel-args.h>

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

// Each threadgroup handles 8 Q heads / 1 KV head for 1 token

[[max_total_threads_per_threadgroup(32)]]
kernel void gptoss_f32_sdpa_q8_d64(
    constant gptoss_sdpa_args& args [[ buffer(0) ]],
    const device float* q [[ buffer(1) ]],
    const device float* k [[ buffer(2) ]],
    const device float* v [[ buffer(3) ]],
    const device bfloat* s [[ buffer(4) ]],
    device float* output [[ buffer(5) ]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    const uint num_q_heads = 64;
    const uint num_kv_heads = 8;
    const uint head_dim = 64;
    const uint qmul = 8;

    const uint qt = gid.x;  // Q token index
    const uint h = gid.y;   // KV head index

    q += qt * args.qkv_dim + h * (qmul * head_dim);
    k += h * head_dim;
    v += h * head_dim;
    output += qt * (num_q_heads * head_dim) + h * (qmul * head_dim);

    float m0 = static_cast<float>(s[h * qmul + 0]);
    float m1 = static_cast<float>(s[h * qmul + 1]);
    float m2 = static_cast<float>(s[h * qmul + 2]);
    float m3 = static_cast<float>(s[h * qmul + 3]);
    float m4 = static_cast<float>(s[h * qmul + 4]);
    float m5 = static_cast<float>(s[h * qmul + 5]);
    float m6 = static_cast<float>(s[h * qmul + 6]);
    float m7 = static_cast<float>(s[h * qmul + 7]);

    float l0 = 1.0f;
    float l1 = 1.0f;
    float l2 = 1.0f;
    float l3 = 1.0f;
    float l4 = 1.0f;
    float l5 = 1.0f;
    float l6 = 1.0f;
    float l7 = 1.0f;

    float2 out0 = 0.0f;
    float2 out1 = 0.0f;
    float2 out2 = 0.0f;
    float2 out3 = 0.0f;
    float2 out4 = 0.0f;
    float2 out5 = 0.0f;
    float2 out6 = 0.0f;
    float2 out7 = 0.0f;

    float2 q0 = reinterpret_cast<const device float2*>(q + 0 * head_dim)[tid];
    float2 q1 = reinterpret_cast<const device float2*>(q + 1 * head_dim)[tid];
    float2 q2 = reinterpret_cast<const device float2*>(q + 2 * head_dim)[tid];
    float2 q3 = reinterpret_cast<const device float2*>(q + 3 * head_dim)[tid];
    float2 q4 = reinterpret_cast<const device float2*>(q + 4 * head_dim)[tid];
    float2 q5 = reinterpret_cast<const device float2*>(q + 5 * head_dim)[tid];
    float2 q6 = reinterpret_cast<const device float2*>(q + 6 * head_dim)[tid];
    float2 q7 = reinterpret_cast<const device float2*>(q + 7 * head_dim)[tid];

    const uint kt_end = qt + args.num_kv_tokens + 1;
    const uint kt_start = metal::subsat(kt_end, args.window);
    k += 2 * num_kv_heads * head_dim * kt_start;
    v += 2 * num_kv_heads * head_dim * kt_start;
    for (uint kt = kt_start; kt < kt_end; kt++) {
        const float2 kval = reinterpret_cast<const device float2*>(k)[tid];
        k += 2 * num_kv_heads * head_dim;

        float qk0 = metal::dot(q0, kval);
        float qk1 = metal::dot(q1, kval);
        float qk2 = metal::dot(q2, kval);
        float qk3 = metal::dot(q3, kval);
        float qk4 = metal::dot(q4, kval);
        float qk5 = metal::dot(q5, kval);
        float qk6 = metal::dot(q6, kval);
        float qk7 = metal::dot(q7, kval);

        qk0 = metal::simd_sum(qk0);
        qk1 = metal::simd_sum(qk1);
        qk2 = metal::simd_sum(qk2);
        qk3 = metal::simd_sum(qk3);
        qk4 = metal::simd_sum(qk4);
        qk5 = metal::simd_sum(qk5);
        qk6 = metal::simd_sum(qk6);
        qk7 = metal::simd_sum(qk7);

        const float new_m0 = metal::max(m0, qk0);
        const float new_m1 = metal::max(m1, qk1);
        const float new_m2 = metal::max(m2, qk2);
        const float new_m3 = metal::max(m3, qk3);
        const float new_m4 = metal::max(m4, qk4);
        const float new_m5 = metal::max(m5, qk5);
        const float new_m6 = metal::max(m6, qk6);
        const float new_m7 = metal::max(m7, qk7);

        const float alpha0 = metal::fast::exp(m0 - new_m0);
        const float alpha1 = metal::fast::exp(m1 - new_m1);
        const float alpha2 = metal::fast::exp(m2 - new_m2);
        const float alpha3 = metal::fast::exp(m3 - new_m3);
        const float alpha4 = metal::fast::exp(m4 - new_m4);
        const float alpha5 = metal::fast::exp(m5 - new_m5);
        const float alpha6 = metal::fast::exp(m6 - new_m6);
        const float alpha7 = metal::fast::exp(m7 - new_m7);

        qk0 = metal::fast::exp(qk0 - new_m0);
        qk1 = metal::fast::exp(qk1 - new_m1);
        qk2 = metal::fast::exp(qk2 - new_m2);
        qk3 = metal::fast::exp(qk3 - new_m3);
        qk4 = metal::fast::exp(qk4 - new_m4);
        qk5 = metal::fast::exp(qk5 - new_m5);
        qk6 = metal::fast::exp(qk6 - new_m6);
        qk7 = metal::fast::exp(qk7 - new_m7);

        l0 = metal::fma(l0, alpha0, qk0);
        l1 = metal::fma(l1, alpha1, qk1);
        l2 = metal::fma(l2, alpha2, qk2);
        l3 = metal::fma(l3, alpha3, qk3);
        l4 = metal::fma(l4, alpha4, qk4);
        l5 = metal::fma(l5, alpha5, qk5);
        l6 = metal::fma(l6, alpha6, qk6);
        l7 = metal::fma(l7, alpha7, qk7);

        m0 = new_m0;
        m1 = new_m1;
        m2 = new_m2;
        m3 = new_m3;
        m4 = new_m4;
        m5 = new_m5;
        m6 = new_m6;
        m7 = new_m7;

        const float2 vval = reinterpret_cast<const device float2*>(v)[tid];
        v += 2 * num_kv_heads * head_dim;
        out0 = metal::fma(vval, qk0, out0 * alpha0);
        out1 = metal::fma(vval, qk1, out1 * alpha1);
        out2 = metal::fma(vval, qk2, out2 * alpha2);
        out3 = metal::fma(vval, qk3, out3 * alpha3);
        out4 = metal::fma(vval, qk4, out4 * alpha4);
        out5 = metal::fma(vval, qk5, out5 * alpha5);
        out6 = metal::fma(vval, qk6, out6 * alpha6);
        out7 = metal::fma(vval, qk7, out7 * alpha7);
    }
    reinterpret_cast<device float2*>(output + 0 * head_dim)[tid] = out0 / l0;
    reinterpret_cast<device float2*>(output + 1 * head_dim)[tid] = out1 / l1;
    reinterpret_cast<device float2*>(output + 2 * head_dim)[tid] = out2 / l2;
    reinterpret_cast<device float2*>(output + 3 * head_dim)[tid] = out3 / l3;
    reinterpret_cast<device float2*>(output + 4 * head_dim)[tid] = out4 / l4;
    reinterpret_cast<device float2*>(output + 5 * head_dim)[tid] = out5 / l5;
    reinterpret_cast<device float2*>(output + 6 * head_dim)[tid] = out6 / l6;
    reinterpret_cast<device float2*>(output + 7 * head_dim)[tid] = out7 / l7;
}
