#include <metal_common>
#include <metal_math>

#include <internal/kernel-args.h>

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)


// Each thread handles 2 head elements.
// Each simdgroup handles one head (64 head elements).

kernel void gptoss_f32_rope(
    constant gptoss_rope_args& args [[ buffer(0) ]],
    device float2* activations [[ buffer(1) ]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint num_head_dims = 64;
    const float head_idx = static_cast<float>(gid.x % (num_head_dims / 2));
    const uint token_idx = args.token_offset + gid.y;
    activations += gid.y * args.token_stride + gid.x;

    const float2 input_vals = *activations;
    const float inv_extrapolation_freq = metal::precise::exp(head_idx * args.freq_scale);
    const float inv_interpolation_freq = inv_extrapolation_freq * args.interpolation_scale;
    const float alpha = metal::saturate(metal::fma(head_idx, args.yarn_scale, args.yarn_offset));
    const float inv_freq = metal::mix(inv_extrapolation_freq, inv_interpolation_freq, alpha);

    const float phi = static_cast<float>(token_idx) * inv_freq;
    const float yarn_multiplier = args.yarn_multiplier;
    float cosphi;
    const float sinphi = metal::precise::sincos(phi, cosphi) * yarn_multiplier;
    cosphi *= yarn_multiplier;

    const float output_re = metal::fma(-input_vals.y, sinphi, input_vals.x * cosphi);
    const float output_im = metal::fma(input_vals.y, cosphi, input_vals.x * sinphi);
    *activations = (float2) { output_re, output_im };
}
