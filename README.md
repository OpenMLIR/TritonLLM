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

Implements modular Triton-backed LLM inference with an emphasis on kernel optimization using CUBINs. The initial target is the [gpt-oss](https://github.com/openai/gpt-oss) model, executed via [triton_runner](https://github.com/OpenMLIR/triton_runner) and will be tuned for **RTX 5090** (sm120). Now support an NVIDIA GPU with [compute capability](https://developer.nvidia.com/cuda-gpus) sm120(RTX 5090, RTX PRO 6000, etc.), sm90(H100, H200, H20, etc.), sm80(A800, A100), sm89(RTX 4090, RTX 6000, L40, etc.) and sm86(RTX 3090, A10, etc.). If the GPU memory is greater than or equal to **24 GB**, you can run the **gpt-oss-20b**; if it is greater than or equal to **80 GB**, you can run the **gpt-oss-120b**.

## Installation

```bash
git clone https://github.com/OpenMLIR/tritonllm
cd tritonllm

pip install -e .
```

## Download model

```bash
pip install -U huggingface_hub
# or  ~/.local/bin/huggingface-cli, if warning `huggingface-cli: command not found`
huggingface-cli download openai/gpt-oss-20b --include "original/*" --local-dir gpt-oss-20b/
```

## Run

```bash
# test
python examples/generate.py gpt-oss-20b/original/

# chat
python examples/chat.py gpt-oss-20b/original/
```

## Benchmark

I am currently optimizing **Tokens Per Second**(TPS), the number of tokens generated per second during autoregressive decoding.

```bash
python -m bench.bench_chat gpt-oss-20b/original/

# show output
python -m bench.only_output gpt-oss-20b/original/
```

## Run use streamlit with Responses API(has bug)

You can also use Streamlit to interact with the [Responses API](https://github.com/openai/gpt-oss?tab=readme-ov-file#responses-api), providing a convenient web interface for managing the project.

```bash
pip install streamlit

python -m gpt_oss.responses_api.serve

streamlit run streamlit/streamlit_chat.py
```
