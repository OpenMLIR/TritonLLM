# triton_kernels copy and modify from
# https://github.com/triton-lang/triton/tree/main/python/triton_kernels/triton_kernels

import sys
import tritonllm.triton_kernels as triton_kernels_module

sys.modules['triton_kernels'] = triton_kernels_module
