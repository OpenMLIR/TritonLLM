<h3 align="center">
LLM Inference via Triton 🚀
</h3>

<h4 align="center">
面向小批量低延迟的灵活模块化 LLM 推理
</h4>

<p align="center">
<a href="https://tritonllm.top"><b>🔗 tritonllm.top</b></a>
</p>

<p align="center">
<a href="README.md"><b>English</b></a> | <a><b>中文</b></a>
</p>


以 Triton 算子为核心的 LLM 推理，灵活且模块化。并以 [gpt-oss](https://github.com/openai/gpt-oss) 模型为起点，关注 Triton算子优化后的CUBIN二进制文件并使用[triton_runner](https://github.com/OpenMLIR/triton_runner)进行LLM推理。

将针对**RTX 5090**(Blackwell)进行优化。

## 支持的 GPU

- **sm120**：RTX 5090、RTX PRO 6000 等  
- **sm90**：H100、H200、H20 等  
- **sm80**：A800、A100  
- **sm89**：RTX 4090、RTX 6000、L40 等  
- **sm86**：RTX 3090、A10 等  

## 显存要求

- 若 GPU 显存 **≥ 24 GB**，可运行 **gpt-oss-20b**。  
- 若 GPU 显存 **≥ 80 GB**，可运行 **gpt-oss-120b**。

## 安装

```bash
git clone https://github.com/OpenMLIR/tritonllm
cd tritonllm

pip install -e .
```

## 下载模型

[modelscope](https://www.modelscope.cn)很好用，速度也很快。下载120b模型请自行修改命令。

```bash
pip install modelscope

modelscope download openai-mirror/gpt-oss-20b  --include "original/*" --local_dir gpt-oss-20b/
```

## 运行

使用120b模型请自行修改命令。

```bash
# 测试
python examples/generate.py gpt-oss-20b/original/

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 对话
python examples/chat.py gpt-oss-20b/original/
```

## 性能

我目前在尝试优化 **Tokens Per Second**(TPS)，即每秒生成的Token数量，用来评估模型decode的生成速度。

```bash
python -m bench.bench_chat gpt-oss-20b/original/

# 展示输出，实验性质
python -m bench.only_output gpt-oss-20b/original/
```

## 网页版运行(待修复)

你同样可以使用 streamlit 通过调用 Responses API 来使用这个项目，网页更加直观，且方便共享。

```bash
pip install streamlit

python -m gpt_oss.responses_api.serve

streamlit run streamlit/streamlit_chat.py
```

## 项目文档

[Triton Kernel 优先：全新 LLM 推理方式(47e9dcb)](https://zhuanlan.zhihu.com/p/1939592984820691987)

[5090显卡+Triton，轻松玩转GPT-OSS-20B！(6bb4b91)](https://zhuanlan.zhihu.com/p/1936692690503865129)
