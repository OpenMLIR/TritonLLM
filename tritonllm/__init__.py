# triton_kernels copy and modify from
# https://github.com/triton-lang/triton/tree/main/python/triton_kernels/triton_kernels
import sys
from tritonllm import gpt_oss, triton_kernels


sys.modules['triton_kernels'] = triton_kernels
sys.modules['gpt_oss'] = gpt_oss

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from pathlib import Path
triton_llm_bin = os.path.join(Path(__file__).parent, "bin")
os.environ["TIKTOKEN_CACHE_DIR"] = triton_llm_bin

from .utils import save_file_to_triton_llm_bin
save_file_to_triton_llm_bin(triton_llm_bin)
