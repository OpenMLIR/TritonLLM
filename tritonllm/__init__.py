# triton_kernels copy and modify from
# https://github.com/triton-lang/triton/tree/main/python/triton_kernels/triton_kernels
import sys
from tritonllm import gpt_oss, triton_kernels


sys.modules['triton_kernels'] = triton_kernels
sys.modules['gpt_oss'] = gpt_oss

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from pathlib import Path
tritonllm_bin_dir = os.path.join(Path(__file__).parent, "bin")
os.environ["TIKTOKEN_CACHE_DIR"] = tritonllm_bin_dir

from .utils import save_file_to_tritonllm_bin_dir
save_file_to_tritonllm_bin_dir(tritonllm_bin_dir)
