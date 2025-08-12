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


def get_user_input():
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank == 0:
        user_input = input()
    else:
        user_input = ""
    user_input_list = [user_input]
    if torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(user_input_list, 0)
    return user_input_list[0]

def once_inference(user_message, messages, encoding, generator):
    token_num = 0
    user_message = Message.from_role_and_content(Role.USER, user_message)
    messages.append(user_message)

    conversation = Conversation.from_messages(messages)
    tokens = encoding.render_conversation_for_completion(
        conversation, Role.ASSISTANT
    )

    if args.raw:
        # Print the last two tokens, which are the start of the assistant message
        print(encoding.decode(tokens[-2:]), flush=True, end="")

    parser = StreamableParser(encoding, role=Role.ASSISTANT)
    field_created = False
    current_output_text = ""
    output_text_delta_buffer = ""
    for predicted_token in generator.generate(tokens, encoding.stop_tokens_for_assistant_actions()):
        token_num += 1
        parser.process(predicted_token)

        if not parser.last_content_delta:
            continue

        should_send_output_text_delta = True
        output_text_delta_buffer += parser.last_content_delta
        if args.browser:
            updated_output_text, _annotations, has_partial_citations = browser_tool.normalize_citations(current_output_text + output_text_delta_buffer)
            output_text_delta_buffer = updated_output_text[len(current_output_text):]
            if has_partial_citations:
                should_send_output_text_delta = False
        if should_send_output_text_delta:
            current_output_text += output_text_delta_buffer
            output_text_delta_buffer = ""
    # has 10 parser.last_content_delta
    return token_num - 10

def get_file_lines_with_random(file_name):
    with open("bench/promt.txt", "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    random.shuffle(lines)
    return lines

def main(args):
    from gpt_oss.triton.model import TokenGenerator as TritonGenerator
    device = torch.device(f"cuda:0")
    tokenizer = get_tokenizer()
    generator = TritonGenerator(args.checkpoint, args.context, device)

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    system_message_content = (
        SystemContent.new()
        .with_reasoning_effort(REASONING_EFFORT[args.reasoning_effort])
        .with_conversation_start_date(datetime.datetime.now().strftime("%Y-%m-%d"))
    )

    if args.browser:
        backend = ExaBackend(
            source="web",
        )
        browser_tool = SimpleBrowserTool(backend=backend)
        system_message_content = system_message_content.with_tools(browser_tool.tool_config)

    if args.python:
        python_tool = PythonTool()
        system_message_content = system_message_content.with_tools(python_tool.tool_config)

    system_message = Message.from_role_and_content(Role.SYSTEM, system_message_content)
    messages = [system_message]

    if args.apply_patch:
        apply_patch_instructions = Path(apply_patch.__file__).parent / "apply_patch.md"
        developer_message = ""
        if args.developer_message:
            developer_message = args.developer_message + "\n"
        developer_message += apply_patch_instructions.read_text()
        developer_message_content = (
            DeveloperContent.new()
            .with_instructions(developer_message)
            .with_function_tools([
                ToolDescription.new(
                    "apply_patch",
                    "Patch a file",
                    parameters={
                        "type": "string",
                        "description": "Formatted patch code",
                        "default": "*** Begin Patch\n*** End Patch\n",
                    }
                ),
            ])
        )
        messages.append(Message.from_role_and_content(Role.DEVELOPER, developer_message_content))
    else:
        developer_message_content = DeveloperContent.new().with_instructions(args.developer_message)
        messages.append(Message.from_role_and_content(Role.DEVELOPER, developer_message_content))

    if args.raw:
        conversation = Conversation.from_messages(messages)
        tokens = encoding.render_conversation(conversation)
        system_message = encoding.decode(tokens)
        print(system_message, flush=True, end="")
        empty_user_message_tokens = encoding.render(Message.from_role_and_content(Role.USER, ""))
        user_message_start = encoding.decode(empty_user_message_tokens[:-1])
        user_message_end = encoding.decode(empty_user_message_tokens[-1:])
    else:
        # System message
        print(termcolor.colored("System Message:", "cyan"), flush=True)
        print(termcolor.colored("Model Identity:", "cyan"), system_message_content.model_identity, flush=True)
        print(termcolor.colored("Reasoning Effort:", "cyan"), system_message_content.reasoning_effort, flush=True)
        print(termcolor.colored("Conversation Start Date:", "cyan"), system_message_content.conversation_start_date, flush=True)
        print(termcolor.colored("Knowledge Cutoff:", "cyan"), system_message_content.knowledge_cutoff, flush=True)
        print(termcolor.colored("Browser Tool:", "cyan"), "Enabled" if args.browser else "Disabled", flush=True)
        print(termcolor.colored("Python Tool:", "cyan"), "Enabled" if args.python else "Disabled", flush=True)
        print(termcolor.colored("Apply Patch Function:", "cyan"), "Enabled" if args.apply_patch else "Disabled", flush=True)
        # Developer message
        print(termcolor.colored("Developer Message:", "yellow"), flush=True)
        print(developer_message_content.instructions, flush=True)

    # Print the system message and the user message start
    MESSAGE_PADDING = 12
    lines = get_file_lines_with_random("promt.txt")
    once_inference(lines[0], messages, encoding, generator) 
    once_inference(lines[1], messages, encoding, generator) 
    for promt_file in ["promt_zh.txt", "promt.txt"]:
        lines = get_file_lines_with_random("promt.txt")
        time_sum = 0
        token_sum = 0
        for user_message in lines:
            token_begin = time.perf_counter()
            token_num = once_inference(user_message, messages, encoding, generator) 
            token_end = time.perf_counter()
            elapsed = token_end - token_begin
            token_sum += token_num
            time_sum += elapsed
            print(termcolor.colored(f'ITL(Inter-token Latency) {token_num / elapsed:.3f}', "yellow"), flush=True)
        print(termcolor.colored(f'{promt_file} AVG ITL(Inter-token Latency) {token_sum / time_sum:.3f}', "yellow"), flush=True)

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
        default="low",
        choices=["high", "medium", "low"],
        help="Reasoning effort",
    )
    parser.add_argument(
        "-a",
        "--apply-patch",
        action="store_true",
        help="Make apply_patch function available to the model",
    )
    parser.add_argument(
        "-b",
        "--browser",
        default=False,
        action="store_true",
        help="Use browser tool",
    )
    parser.add_argument(
        "--show-browser-results",
        default=False,
        action="store_true",
        help="Show browser results",
    )
    parser.add_argument(
        "-p",
        "--python",
        default=False,
        action="store_true",
        help="Use python tool",
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
    parser.add_argument(
        "--raw",
        default=False,
        action="store_true",
        help="Raw mode (does not render Harmony encoding)",
    )
    args = parser.parse_args()

    if int(os.environ.get("WORLD_SIZE", 1)) == 1:
        histfile = os.path.join(os.path.expanduser("~"), ".chat")
        try:
            readline.read_history_file(histfile)
            readline.set_history_length(10000)
        except FileNotFoundError:
            pass

        atexit.register(readline.write_history_file, histfile)

    main(args)
