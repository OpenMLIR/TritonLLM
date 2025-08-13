import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm(x_ptr, t_ptr, scale_ptr, last_dim, eps, BLOCK_SIZE: tl.constexpr):
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
    grid = (remaining, triton.cdiv(last_dim, 128))
    rmsnorm[grid](x, t, scale, last_dim, eps, BLOCK_SIZE=128)
    return t
