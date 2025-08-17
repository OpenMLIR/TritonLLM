from tritonllm.gpt_oss.generate import generate, get_parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    generate(args)
