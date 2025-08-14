"""
Harmony chat with tools
"""

import atexit
import argparse
import asyncio
import datetime
import os
import time
import random

from pathlib import Path

try:
    import gnureadline as readline
except ImportError:
    import readline

import torch
import torch.distributed as dist
import termcolor

from gpt_oss.tools import apply_patch
from gpt_oss.tools.simple_browser import SimpleBrowserTool
from gpt_oss.tools.simple_browser.backend import ExaBackend
from gpt_oss.tools.python_docker.docker_tool import PythonTool
from gpt_oss.tokenizer import get_tokenizer

from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    StreamableParser,
    StreamState,
    SystemContent,
    TextContent,
    ToolDescription,
    load_harmony_encoding,
)


REASONING_EFFORT = {
    "high": ReasoningEffort.HIGH,
    "medium": ReasoningEffort.MEDIUM,
    "low": ReasoningEffort.LOW,
}

def once_inference(user_message, messages, encoding, generator):
    user_message = Message.from_role_and_content(Role.USER, user_message)
    messages.append(user_message)

    conversation = Conversation.from_messages(messages)
    tokens = encoding.render_conversation_for_completion(
        conversation, Role.ASSISTANT
    )
    token_num = 0
    for predicted_token in generator.generate(tokens, encoding.stop_tokens_for_assistant_actions()):
        token_num += 1
    # has 10 parser.last_content_delta
    return token_num - 10

def get_file_lines_with_random(file_name):
    with open(f"bench/{file_name}", "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    random.shuffle(lines)
    return lines

def main(args):
    from gpt_oss.triton.model import TokenGenerator as TritonGenerator
    device = torch.device(f"cuda:0")
    generator = TritonGenerator(args.checkpoint, args.context, device)

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    system_message_content = (
        SystemContent.new()
        .with_reasoning_effort(REASONING_EFFORT[args.reasoning_effort])
        .with_conversation_start_date(datetime.datetime.now().strftime("%Y-%m-%d"))
    )

    system_message = Message.from_role_and_content(Role.SYSTEM, system_message_content)
    messages = [system_message]

    developer_message_content = DeveloperContent.new().with_instructions(args.developer_message)
    messages.append(Message.from_role_and_content(Role.DEVELOPER, developer_message_content))

    # System message
    print(termcolor.colored("System Message:", "cyan"), flush=True)
    print(termcolor.colored("Model Identity:", "cyan"), system_message_content.model_identity, flush=True)
    print(termcolor.colored("Reasoning Effort:", "cyan"), system_message_content.reasoning_effort, flush=True)
    print(termcolor.colored("Conversation Start Date:", "cyan"), system_message_content.conversation_start_date, flush=True)
    print(termcolor.colored("Knowledge Cutoff:", "cyan"), system_message_content.knowledge_cutoff, flush=True)
    # Developer message
    print(termcolor.colored("Developer Message:", "yellow"), flush=True)
    print(developer_message_content.instructions, flush=True)

    file_lst = ["prompt_zh.txt", "prompt.txt"]
    lines = get_file_lines_with_random(file_lst[1])
    once_inference(lines[0], messages, encoding, generator) 
    once_inference(lines[1], messages, encoding, generator) 
    for prompt_file in file_lst:
        lines = get_file_lines_with_random(prompt_file)
        time_sum, token_sum = 0, 0
        for user_message in lines:
            token_begin = time.perf_counter()
            token_num = once_inference(user_message, messages, encoding, generator) 
            elapsed = time.perf_counter() - token_begin
            time_sum, token_sum = time_sum + elapsed, token_sum + token_num
            print(termcolor.colored(f'ITL(Inter-token Latency) {token_num / elapsed:.3f}', "yellow"), flush=True)
        print(termcolor.colored(f'{prompt_file} AVG ITL(Inter-token Latency) {token_sum / time_sum:.3f}', "yellow"), flush=True)

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
