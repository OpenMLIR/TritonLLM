from tritonllm.gpt_oss.bench import HarmonyChatTool
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HarmonyChatTool")
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="gpt-oss-20b/original/",
        type=str,
        help="Path to the SafeTensors checkpoint (default: %(default)s)"
    )
    args = parser.parse_args()
    tool = HarmonyChatTool(args.checkpoint, reasoning_effort="high")
    result = tool.benchmark_mode()
