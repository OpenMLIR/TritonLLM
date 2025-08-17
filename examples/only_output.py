"""
Harmony chat with tools
"""

import argparse
import datetime
import time


import torch
import termcolor

from openai_harmony import (
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    StreamableParser,
    StreamState,
    SystemContent,
    load_harmony_encoding,
)


REASONING_EFFORT = {
    "high": ReasoningEffort.HIGH,
    "medium": ReasoningEffort.MEDIUM,
    "low": ReasoningEffort.LOW,
}


def once_inference(user_message, messages, encoding, generator):
    MESSAGE_PADDING = 12
    print(termcolor.colored("User:".ljust(MESSAGE_PADDING), "red"), flush=True)
    print(user_message)
    user_message = Message.from_role_and_content(Role.USER, user_message)
    messages.append(user_message)
    conversation = Conversation.from_messages(messages)
    tokens = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
    parser = StreamableParser(encoding, role=Role.ASSISTANT)
    current_output_text = ""
    output_text_delta_buffer = ""
    field_created = False
    token_begin = time.perf_counter()
    token_num = 0
    for predicted_token in generator.generate(
        tokens, encoding.stop_tokens_for_assistant_actions()
    ):
        token_num += 1
        parser.process(predicted_token)

        if parser.state == StreamState.EXPECT_START:
            print("")  # new line
            field_created = False

        if not parser.last_content_delta:
            continue

        if not field_created:
            field_created = True
            if parser.current_channel == "final":
                print(termcolor.colored("Assistant:", "green"), flush=True)
            else:
                print(termcolor.colored("CoT:", "yellow"), flush=True)
        output_text_delta_buffer += parser.last_content_delta
        print(output_text_delta_buffer, end="", flush=True)
        current_output_text += output_text_delta_buffer
        output_text_delta_buffer = ""
    # has 10 parser.last_content_delta
    token_num -=  10
    token_end = time.perf_counter()
    elapsed = token_end - token_begin
    print(termcolor.colored(f'TPS(Tokens Per Second) {token_num / elapsed:.3f}\n\n', "yellow"), flush=True)


def get_file_lines_with_random(file_name):
    import tritonllm as tllm
    import os
    file_path = os.path.join(tllm.__path__[0], "bin", file_name)
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    return lines


def main(args):
    from tritonllm.gpt_oss.triton.model import TokenGenerator as TritonGenerator

    device = torch.device(f"cuda:0")
    generator = TritonGenerator(args.checkpoint, args.context, device)

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    system_message_content = (
        SystemContent.new()
        .with_reasoning_effort(REASONING_EFFORT[args.reasoning_effort])
        .with_conversation_start_date(datetime.datetime.now().strftime("%Y-%m-%d"))
    )

    system_message = Message.from_role_and_content(Role.SYSTEM, system_message_content)

    # System message
    print(termcolor.colored("System Message:", "cyan"), flush=True)
    print(
        termcolor.colored("Model Identity:", "cyan"),
        system_message_content.model_identity,
        flush=True,
    )
    print(
        termcolor.colored("Reasoning Effort:", "cyan"),
        system_message_content.reasoning_effort,
        flush=True,
    )
    print(
        termcolor.colored("Conversation Start Date:", "cyan"),
        system_message_content.conversation_start_date,
        flush=True,
    )
    print(
        termcolor.colored("Knowledge Cutoff:", "cyan"),
        system_message_content.knowledge_cutoff,
        flush=True,
    )
    # Developer message
    print(termcolor.colored("Developer Message:", "yellow"), flush=True)

    output_text = ""
    file_lst = ["prompt_zh.txt", "prompt.txt"]
    for prompt_file in file_lst:
        lines = get_file_lines_with_random(prompt_file)
        for user_message in lines:
            messages = [system_message]
            developer_message_content = DeveloperContent.new().with_instructions(
                args.developer_message
            )
            messages.append(
                Message.from_role_and_content(Role.DEVELOPER, developer_message_content)
            )
            once_inference(user_message, messages, encoding, generator)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chat example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "checkpoint",
        metavar="FILE",
        type=str,
        help="Path to the SafeTensors checkpoint",
    )
    parser.add_argument(
        "-r",
        "--reasoning-effort",
        metavar="REASONING_EFFORT",
        type=str,
        default="high",
        choices=["high", "medium", "low"],
        help="Reasoning effort",
    )
    parser.add_argument(
        "--developer-message",
        default="",
        help="Developer message",
    )
    parser.add_argument(
        "-c",
        "--context",
        metavar="CONTEXT",
        type=int,
        default=8192,
        help="Max context length",
    )
    args = parser.parse_args()
    main(args)
