from tritonllm.gpt_oss.bench import HarmonyChatTool
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HarmonyChatTool")
    parser.add_argument(
        "--model_path",
        type=str,
        default="gpt-oss-20b/original/",
        help="Path to the model"
    )
    args = parser.parse_args()
    tool = HarmonyChatTool(args.model_path, reasoning_effort="high")
    result = tool.only_output()
