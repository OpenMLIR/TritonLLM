# triton_kernels copy and modify from
# https://github.com/triton-lang/triton/tree/main/python/triton_kernels/triton_kernels

import sys
import tritonllm.triton_kernels as triton_kernels_module
import tritonllm.gpt_oss as gpt_oss_module

sys.modules['triton_kernels'] = triton_kernels_module
sys.modules['gpt_oss'] = gpt_oss_module

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import tritonllm as tllm
triton_llm_bin = os.path.join(tllm.__path__[0], "bin")
os.environ["TIKTOKEN_CACHE_DIR"] = triton_llm_bin

from .utils import save_file_to_triton_llm_bin
save_file_to_triton_llm_bin(triton_llm_bin)
