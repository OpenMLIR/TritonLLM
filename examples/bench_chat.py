from tritonllm.gpt_oss.bench import HarmonyChatTool

if __name__ == "__main__":
    tool = HarmonyChatTool("gpt-oss-20b/original/", reasoning_effort="high")
    # result = tool.single_inference("Your question here", interactive=True)
    result = tool.benchmark_mode()
    tool.single_inference("Your question here", interactive=True)