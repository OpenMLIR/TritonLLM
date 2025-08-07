# triton_llm

triton_llm is a lightweight and efficient LLM inference framework powered by custom Triton operators. It leverages [triton_runner](https://github.com/OpenMLIR/triton_runner) as its execution backend and is initially optimized for serving [gpt-oss](https://github.com/openai/gpt-oss) on RTX 5090(sm120). Designed for extensibility and performance, triton_llm aims to make deploying large language models fast, modular, and Triton-friendly.
