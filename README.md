# triton_llm

triton_llm is a lightweight and efficient LLM inference framework powered by custom Triton operators. It leverages [triton_runner](https://github.com/OpenMLIR/triton_runner) as its execution backend and is initially optimized for serving [gpt-oss](https://github.com/openai/gpt-oss) on RTX 5090(sm120). Designed for extensibility and performance, triton_llm aims to make deploying large language models fast, modular, and Triton-friendly.

```bash
git clone https://github.com/OpenMLIR/triton_llm
pip install -e .[triton]
pip install -e triton_kernels
pip install triton==3.4.0
```

run cmd
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m gpt_oss.generate --backend triton gpt-oss-20b/original/
```
