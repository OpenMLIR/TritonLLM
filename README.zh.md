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

## 快速安装

你可以通过 pip 安装 tritonllm 的最新稳定版本

```shell
pip install tritonllm
```

## 命令行界面 (CLI)

快速启动 gpt-oss-20b 模型的对话，将自动从 ModelScope 魔搭下载。

```shell
tritonllm
```

你也可以查看所有可用选项：

```shell
tritonllm --help
```

### 使用方法

```
usage: tritonllm [-h] [-r REASONING_EFFORT] [-a] [-b] [--show-browser-results] [-p]
                 [--developer-message DEVELOPER_MESSAGE] [-c CONTEXT] [--raw]
                 [FILE]
```

## 位置参数

| 参数 | 说明 |
|------|------|
| `FILE` | SafeTensors 检查点文件路径。如果未提供，将自动下载 **20B 模型**。你也可以运行 `tritonllm 120b` 来直接使用 **120B 模型**。 |

## 可选参数

| 参数 | 说明 |
|------|------|
| `-h, --help` | 显示帮助信息并退出。 |
| `-r REASONING_EFFORT, --reasoning-effort REASONING_EFFORT` | 设置推理努力等级（`low` / `medium` / `high`）。默认：`high`。 |
| `-a, --apply-patch` | 使模型可使用内部 `apply_patch` 函数。默认：`False`。 |
| `-b, --browser` | 启用浏览器工具，让模型可以抓取网页内容。默认：`False`。 |
| `--show-browser-results` | 在输出中显示抓取的浏览器结果。默认：`False`。 |
| `-p, --python` | 启用 Python 执行工具（允许模型运行 Python 代码片段）。默认：`False`。 |
| `--developer-message DEVELOPER_MESSAGE` | 提供开发者/系统消息以影响模型行为。 |
| `-c CONTEXT, --context CONTEXT` | 最大上下文长度（Token 数）。默认：`8192`。 |
| `--raw` | 原始模式，禁用 Harmony 编码并输出纯文本。默认：`False`。 |


## 安装

```shell
git clone https://github.com/OpenMLIR/tritonllm
cd tritonllm

pip install -e .
```

## 样例

```Python
from tritonllm.gpt_oss.chat import chat, get_parser_args


if __name__ == "__main__":
    chat(get_parser_args())
```

## 运行

使用120b模型请自行修改命令。

```shell
# 测试
python examples/generate.py

# 对话
python examples/chat.py
```

## 性能

我目前在尝试优化 **Tokens Per Second**(TPS)，即每秒生成的Token数量，用来评估模型decode的生成速度。

```shell
python examples/bench_chat.py

# 展示输出，实验性质
python examples/only_output.py
```

## 网页版运行(待修复)

你同样可以使用 streamlit 通过调用 Responses API 来使用这个项目，网页更加直观，且方便共享。

```shell
pip install streamlit

python -m gpt_oss.responses_api.serve

streamlit run streamlit/streamlit_chat.py
```

## 项目文档

[Triton Kernel 优先：全新 LLM 推理方式(47e9dcb)](https://zhuanlan.zhihu.com/p/1939592984820691987)

[5090显卡+Triton，轻松玩转GPT-OSS-20B！(6bb4b91)](https://zhuanlan.zhihu.com/p/1936692690503865129)

## triton_kernels

triton_kernels 是一组用于在不同架构上实现高速 MoE（Mixture of Experts）的核函数（kernels）。这些内核支持多种精度格式（例如 bf16、mxfp4）。

原始代码在这里：
https://github.com/triton-lang/triton/tree/main/python/triton_kernels

当前版本对应的提交为：de4376e90a3c2b5ca30ada25a50cccadeadf7f1a，
并且使用了 BlackwellMXValueLayout 的提交：19ca20fda4cfd3ae0d3eabde5e547db581fbb7ee。
