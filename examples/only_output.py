from tritonllm.gpt_oss.bench import HarmonyChatTool

if __name__ == "__main__":
    tool = HarmonyChatTool("gpt-oss-20b/original/", reasoning_effort="high")
    result = tool.only_output()
