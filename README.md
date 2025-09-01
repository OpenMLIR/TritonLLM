<h3 align="center">
LLM Inference via Triton 🚀
</h3>

<h4 align="center">
Flexible and modular LLM inference for mini-batch
</h4>

<p align="center">
<a href="https://tritonllm.top"><b>🔗 tritonllm.top</b></a>
</p>

<p align="center">
<a ><b>English</b></a> | <a href="README.zh.md"><b>中文</b></a>
</p>

Implements modular [Triton](https://github.com/triton-lang/triton)-backed LLM inference with an emphasis on kernel optimization using CUBINs. The initial target is the [gpt-oss](https://github.com/openai/gpt-oss) model, executed via [triton_runner](https://github.com/OpenMLIR/triton_runner) and will be tuned for **RTX 5090** (sm120). Now support an NVIDIA GPU with [compute capability](https://developer.nvidia.com/cuda-gpus) sm120(RTX 5090, RTX PRO 6000, etc.), sm90(H100, H200, H20, etc.), sm80(A800, A100), sm89(RTX 4090, RTX 6000, L40, etc.) and sm86(RTX 3090, A10, etc.). If the GPU memory is greater than or equal to **24 GB**, you can run the **gpt-oss-20b**; if it is greater than or equal to **80 GB**, you can run the **gpt-oss-120b**.

## Installation

```shell
git clone https://github.com/OpenMLIR/tritonllm
cd tritonllm

pip install -e .
```

## example code

```Python
from tritonllm.gpt_oss.chat import chat, get_parser_args


if __name__ == "__main__":
    chat(get_parser_args())
```

## Run

```shell
# test
python examples/generate.py

# chat
python examples/chat.py
```

## Benchmark

I am currently optimizing **Tokens Per Second**(TPS), the number of tokens generated per second during autoregressive decoding.

```shell
python examples/bench_chat.py

# show output
python examples/only_output.py
```

## Run use streamlit with Responses API(has bug)

You can also use Streamlit to interact with the [Responses API](https://github.com/openai/gpt-oss?tab=readme-ov-file#responses-api), providing a convenient web interface for managing the project.

```shell
pip install streamlit

python -m gpt_oss.responses_api.serve

streamlit run streamlit/streamlit_chat.py
```

## triton_kernels

triton_kernels is a set of kernels that enable fast moe on different architectures. These kernels are compatible with different precision (e.g bf16, mxfp4)

Original code here https://github.com/triton-lang/triton/tree/main/python/triton_kernels

The current version is the following commit de4376e90a3c2b5ca30ada25a50cccadeadf7f1a and use BlackwellMXValueLayout with commit 19ca20fda4cfd3ae0d3eabde5e547db581fbb7ee。 

