import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(x_ptr, t_ptr, scale_ptr, last_dim, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    x_ptr = x_ptr + row * last_dim
    t_ptr = t_ptr + row * last_dim
    _sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, last_dim, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + cols, mask=cols < last_dim, other=0)
        _sum += x * x
    mean = tl.sum(_sum, axis=0) / last_dim + eps
    pid = tl.program_id(1)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offset, mask=offset < last_dim, other=0)
    scale = tl.load(scale_ptr + offset, mask=offset < last_dim, other=0)
    y = x * tl.math.rsqrt(mean) * scale
    tl.store(t_ptr + offset, y, mask=offset < last_dim)


def rmsnorm_forward(x, scale, eps):
    t = torch.empty_like(x)
    last_dim = x.shape[-1]
    remaining = 1
    for s in x.shape[:-1]:
        remaining *= s
    grid = lambda META: (remaining, triton.cdiv(last_dim, META['BLOCK_SIZE']))
    rmsnorm_kernel[grid](x, t, scale, last_dim, eps, BLOCK_SIZE=128)
    return t


@triton.jit
def rope_kernel(
    query_ptr,
    key_ptr,
    sin_ptr,
    cos_ptr,
    offset_ptr,
    max_context_length,
    num_tokens,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_key_value_heads: tl.constexpr,
    head_dim_div: tl.constexpr,
    TILE_TOKENS: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_t = tl.program_id(1)
    query_ptr += pid * num_tokens * num_heads * head_dim
    key_ptr += pid * num_tokens * num_key_value_heads * head_dim
    offs_token = pid_t * TILE_TOKENS + tl.arange(0, TILE_TOKENS)[:, None]
    offset = tl.load(offset_ptr)
    new_offs_token = (offs_token + offset) % max_context_length
    offs_dim_div = tl.arange(0, head_dim_div)[None, :]
    offs = new_offs_token * head_dim_div + offs_dim_div
    sin_cos_mask = offs_token < num_tokens
    sin = tl.load(sin_ptr + offs, mask=sin_cos_mask)
    sin = sin[:, None, :]
    cos = tl.load(cos_ptr + offs, mask=sin_cos_mask)
    cos = cos[:, None, :]

    offs_token = pid_t * TILE_TOKENS + tl.arange(0, TILE_TOKENS)[:, None, None]
    offs_head = tl.arange(0, num_heads)[None, :, None]
    offs_dim = tl.arange(0, head_dim_div)[None, None, :]
    offs = offs_token * (num_heads * head_dim) + offs_head * head_dim + offs_dim
    q_k_mask = offs_token < num_tokens
    q1 = tl.load(query_ptr + offs, mask=q_k_mask)
    q2 = tl.load(query_ptr + offs + head_dim_div, mask=q_k_mask)
    o1 = q1 * cos - q2 * sin
    o2 = q2 * cos + q1 * sin
    tl.store(query_ptr + offs, o1, mask=q_k_mask)
    tl.store(query_ptr + offs + head_dim_div, o2, mask=q_k_mask)
    stride_0 = num_key_value_heads * head_dim
    offs_num_key_value_heads = tl.arange(0, num_key_value_heads)[None, :, None]
    offs = offs_token * stride_0 + offs_num_key_value_heads * head_dim + offs_dim

    k1 = tl.load(key_ptr + offs, mask=q_k_mask)
    k2 = tl.load(key_ptr + offs + head_dim_div, mask=q_k_mask)
    o1 = k1 * cos - k2 * sin
    o2 = k2 * cos + k1 * sin
    tl.store(key_ptr + offs, o1, mask=q_k_mask)
    tl.store(key_ptr + offs + head_dim_div, o2, mask=q_k_mask)

def rope_forward(query, key, sin, cos, max_context_length, offset):
    batch_size, num_tokens, num_heads, head_dim = query.shape
    batch_size, num_tokens, num_key_value_heads, head_dim = key.shape

    grid = lambda META: (batch_size, triton.cdiv(num_tokens, META['TILE_TOKENS']))
    rope_kernel[grid](
        query,
        key,
        sin,
        cos,
        offset,
        max_context_length,
        num_tokens,
        num_heads,
        head_dim,
        num_key_value_heads,
        head_dim // 2,
        TILE_TOKENS = 1 if num_tokens < 60 else 32,
    )


@triton.jit
def matmul_bh_vh_kernel(
    A_ptr, B_ptr, C_ptr,
    B, V, H,
    stride_am, stride_ah,
    stride_bv, stride_bh,
    stride_cb, stride_cv,
    BLOCK_B: tl.constexpr, BLOCK_V: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch dim
    pid_v = tl.program_id(1)  # vocab dim

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    offs_h = tl.arange(0, BLOCK_H)

    acc = tl.zeros((BLOCK_B, BLOCK_V), dtype=tl.float32)

    for h in range(0, H, BLOCK_H):
        # A: [B,H]
        a_ptrs = A_ptr + (offs_b[:, None] * stride_am + (h + offs_h[None, :]) * stride_ah)
        # B: [V,H] (need transpose to [H,V] when reading)
        b_ptrs = B_ptr + (offs_v[None, :] * stride_bv + (h + offs_h[:, None]) * stride_bh)

        a = tl.load(a_ptrs, mask=(offs_b[:, None] < B) & (h + offs_h[None, :] < H), other=0.).to(tl.float32)
        b = tl.load(b_ptrs, mask=(offs_v[None, :] < V) & (h + offs_h[:, None] < H), other=0.).to(tl.float32)

        acc += tl.dot(a, b)

    # Write C: [B,V]
    c_ptrs = C_ptr + offs_b[:, None] * stride_cb + offs_v[None, :] * stride_cv
    tl.store(c_ptrs, acc, mask=(offs_b[:, None] < B) & (offs_v[None, :] < V))


def unembedding_forward(hidden, weight):
    Bdim, H = hidden.shape
    V, H = weight.shape

    C = torch.empty((Bdim, V), device=hidden.device, dtype=torch.float32)

    BLOCK_B, BLOCK_V, BLOCK_H = 32, 32, 32
    grid = (triton.cdiv(Bdim, BLOCK_B), triton.cdiv(V, BLOCK_V))

    matmul_bh_vh_kernel[grid](
        A, B, C,
        Bdim, V, H,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_B=BLOCK_B, BLOCK_V=BLOCK_V, BLOCK_H=BLOCK_H,
    )
    return C
