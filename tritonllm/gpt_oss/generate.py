# Model parallel inference
# Note: This script is for demonstration purposes only. It is not designed for production use.
#       See gpt_oss.chat for a more complete example with the Harmony parser.
# torchrun --nproc-per-node=4 -m gpt_oss.generate -p "why did the chicken cross the road?" model/

import torch

from gpt_oss.tokenizer import get_tokenizer


def generate(args):
    from gpt_oss.triton.model import TokenGenerator as TritonGenerator
    device = torch.device(f"cuda:0")
    generator = TritonGenerator(args.checkpoint, context=4096, device=device)

    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(args.prompt)
    for token, logprob in generator.generate(tokens,
                                             stop_tokens=[tokenizer.eot_token],
                                             temperature=args.temperature,
                                             max_tokens=args.limit,
                                             return_logprobs=True):
        tokens.append(token)
        decoded_token = tokenizer.decode([token])
        print(decoded_token, end="")
    print()
