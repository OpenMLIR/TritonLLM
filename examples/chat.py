from tritonllm.gpt_oss.chat import chat, get_parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    chat(args)
