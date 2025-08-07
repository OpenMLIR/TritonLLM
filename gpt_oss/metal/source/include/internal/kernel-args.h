#pragma once

#if !defined(__METAL_VERSION__)
#include <stdint.h>
#endif

struct gptoss_expert_prediction {
    uint32_t expert_id;
    float score;
};

struct gptoss_topk_args {
    uint32_t num_vecs_per_token;
};

struct gptoss_sdpa_args {
    uint32_t qkv_dim;
    uint32_t num_kv_tokens;
    uint32_t window;
};

struct gptoss_u32_fill_random_args {
    uint64_t num_vecs_per_threadgroup;
    uint64_t num_vecs;
    uint64_t offset;
    uint64_t seed;
};

struct gptoss_f32_fill_random_args {
    uint64_t num_vecs_per_threadgroup;
    uint64_t num_vecs;
    uint64_t offset;
    uint64_t seed;
    float scale;
    float bias;
};

struct gptoss_accumulate_args {
    uint32_t num_vecs_per_expert;
    uint32_t num_vecs_per_threadgroup;
    uint32_t num_vecs;
};

struct gptoss_convert_args {
    uint64_t num_vecs_per_threadgroup;
    uint64_t num_vecs;
};

struct gptoss_embeddings_args {
    uint32_t num_vecs;
};

struct gptoss_rmsnorm_args {
    uint32_t num_vecs;
    float num_channels;
    float epsilon;
};

struct gptoss_matmul_args {
    uint32_t num_column_vecs;
    uint32_t num_rows;
    uint32_t add;
};

struct gptoss_unembedding_args {
    uint32_t num_column_vecs;
    uint32_t num_rows_per_threadgroup;
    uint32_t num_rows;
};

struct gptoss_moe_matmul_swiglu_args {
    uint32_t num_column_vecs;
    uint32_t num_rows;
    uint32_t num_active_experts;
    uint32_t weight_expert_stride;  // in bytes
    uint32_t output_expert_stride;  // in elements
    float swiglu_min;
    float swiglu_max;
};

struct gptoss_moe_matmul_args {
    uint32_t num_column_vecs;
    uint32_t num_rows;
    uint32_t num_active_experts;
    uint32_t input_expert_stride;  // in blocks of 32 elements
    uint32_t weight_expert_stride;  // in bytes
    uint32_t output_expert_stride;  // in elements
};

struct gptoss_rope_args {
    uint32_t token_stride;
    uint32_t token_offset;
    float freq_scale;
    float interpolation_scale;
    float yarn_offset;
    float yarn_scale;
    float yarn_multiplier;
};

struct gptoss_softmax_args {
    uint32_t num_vecs;
    uint32_t num_vecs_per_threadgroup;
    uint32_t max_threadgroups;
    float temperature;
};
