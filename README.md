# triton_llm

implements modular Triton-backed LLM inference with an emphasis on kernel optimization using CUBINs. The initial target is the [gpt-oss](https://github.com/openai/gpt-oss) model, executed via [triton_runner](https://github.com/OpenMLIR/triton_runner) and tuned for RTX 5090 (sm120).

```bash
git clone https://github.com/OpenMLIR/triton_llm
cd triton_llm
pip install -e .[triton]
pip install -e triton_kernels
```

run cmd
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m gpt_oss.generate gpt-oss-20b/original/
```
