# triton_kernels copy and modify from
# https://github.com/triton-lang/triton/tree/main/python/triton_kernels/triton_kernels

import sys
import tritonllm.triton_kernels as triton_kernels_module
import tritonllm.gpt_oss as gpt_oss_module

sys.modules['triton_kernels'] = triton_kernels_module
sys.modules['gpt_oss'] = gpt_oss_module

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
