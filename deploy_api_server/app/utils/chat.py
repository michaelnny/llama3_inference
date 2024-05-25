from typing import Literal, Optional, List, Dict, Any, Union

from utils.modeling import ChatMessage
from utils.tokenizer import Tokenizer

bos_token = "<|begin_of_text|>"
start_header_token = "<|start_header_id|>"
end_header_token = "<|end_header_id|>"
eot_token = "<|eot_id|>"  # end of turn


TRITON_MODEL_NAME_MAP = {
    "llama3": "ensemble",
}

DEFAULT_MODEL_NAME = "ensemble"


def format_single_turn(message: ChatMessage) -> str:
    chat: str = f"{start_header_token}{message.role.strip()}{end_header_token}\n\n"
    chat += message.content.strip()
    chat += eot_token
    return chat


def apply_chat_template(messages: List[ChatMessage]) -> str:
    result: str = bos_token
    for msg in messages:
        result += format_single_turn(msg)

    # Add the start of an assistant message for the model to complete.
    result += f"{start_header_token}assistant{end_header_token}\n\n"
    return result


def get_triton_server_model_name(model_name: str) -> str:
    # Map model name to triton server model name
    if model_name is None or model_name not in TRITON_MODEL_NAME_MAP:
        return DEFAULT_MODEL_NAME
    else:
        return TRITON_MODEL_NAME_MAP[model_name]


def count_number_of_tokens(tokenizer: Tokenizer, text: str) -> int:
    token_ids = tokenizer.encode(text, allowed_special="all")
    return len(token_ids)


def maybe_prune_chat_history(
    tokenizer: Tokenizer,
    messages: List[ChatMessage],
    max_len: int = 512,
    verbose: bool = False,
) -> None:
    """Try to prune chat history to make sure it not exceeds the context limit."""

    if not messages:
        return  # No messages to prune
    elif len(messages) <= 2:
        return  # too short

    max_len = max(max_len, 512)

    if verbose:
        print(f"Max length: {max_len}")

    # Count the number of tokens for each turn, start from the last turn
    total_tokens = 0
    cut_idx = -1
    half_cut = False
    for i in reversed(range(len(messages))):
        message = messages[i]
        formatted_input = format_single_turn(message)
        n = count_number_of_tokens(tokenizer, formatted_input)
        total_tokens += n

        if total_tokens > max_len * 0.9:
            cut_idx = i
            if n > max_len * 0.7:
                half_cut = True
            break

    if cut_idx >= 0:
        if verbose:
            print(f"Before cut:\n{messages}")

        has_system_msg = messages[0].role == "system"
        for i in range(cut_idx + 1):
            if has_system_msg:
                pop_idx = 1
            else:
                pop_idx = 0

            if half_cut and i == cut_idx:
                # Try to cut the content in a single turn to
                sentences = messages[pop_idx].content.split(" ")
                messages[pop_idx].content = " ".join(
                    sentences[-(len(sentences) // 2) :]
                )
            else:
                messages.pop(pop_idx)

        if verbose:
            print(f"After cut:\n{messages}")
