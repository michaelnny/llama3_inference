from typing import Literal, Optional, List, Dict, Any, Union
import re
import json

from utils.schema import ChatMessage, FunctionTool, ToolCallRequest
from utils.tokenizer import Tokenizer
from utils.strings import decode_json_strings


bos_token = '<|begin_of_text|>'
start_header_token = '<|start_header_id|>'
end_header_token = '<|end_header_id|>'
eot_token = '<|eot_id|>'  # end of turn


def extract_tool_calls(content: str) -> Union[List[ToolCallRequest], None]:
    """Extract tool call information from generated content"""
    try:
        # decode to get a list of objects
        decoded = decode_json_strings(content)
        if decoded is not None and len(decoded) > 0:
            response = decoded[0]
            tool_calls = []
            if 'tool_uses' in response:
                for item in response['tool_uses']:
                    tool_name = item['recipient_name'].split('.')[1]
                    tool_arguments = json.dumps(item['parameters'])

                    tool_call = ToolCallRequest(
                        type='function',
                        function={
                            'name': tool_name,
                            'arguments': tool_arguments,
                        },
                    )
                    tool_calls.append(tool_call)
            return tool_calls

    except Exception as error:
        print(f'Failed to extract tool calls, error: {str(error)}')
        return None


def serialize_function_metadata(func_meta: dict) -> str:
    """
    Convert function metadata object into simplified strings for add to system prompt.

    Example input:
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }

    Output:

    // Get the current weather in a given location
    type get_current_weather = (_: {
    // The city and state, e.g. San Francisco, CA
    location: string,
    unit?: \"celsius\" | \"fahrenheit\",
    }) => any;

    """

    if not isinstance(func_meta, dict) or func_meta.get('type') != 'function':
        return ''

    function_meta = func_meta.get('function', {})
    name = function_meta.get('name', None)
    description = function_meta.get('description', None)
    parameters = function_meta.get('parameters', {})
    properties = parameters.get('properties', {})
    required_args = parameters.get('required', [])

    if not all([name, description, properties]):
        return ''

    # one-line description and the name of the function
    result = f'// {description}\n'
    result += f'type {name} = (_: ' + '{\n'

    # add description and type for each field
    for field, field_meta in properties.items():
        field_description = field_meta.get('description', None)
        if field_description:
            result += f'// {field_description}\n'

        # For optional field, add a '?' mark
        optional_mark = '' if field in required_args else '?'
        field_type = field_meta.get('type', 'any')

        if 'enum' in field_meta:
            enum_values = ' | '.join(f'"{value}"' for value in field_meta['enum'])
            field_type = enum_values

        result += f'{field}{optional_mark}: {field_type},\n'

    result += '}) => any;'
    return result


def insert_functions_to_system_message(functions: List[FunctionTool]) -> str:
    """Insert serialized function metadata into the system message.

    Args:
        functions (list[dict]): contains a list of function metadata object

    Returns:
        a system prompt for function calling
    """

    # Start
    message = """
# Tools

## functions

namespace functions {

"""
    # Simplified function strings
    if functions is not None and functions:
        serialized_functions = [serialize_function_metadata(f.dict()) for f in functions]
        serialized_functions = [item for item in serialized_functions if item and len(item) > 1]
        if serialized_functions:
            message += '\n\n'.join(serialized_functions)

    # End
    message += """

} // namespace functions

## multi_tool_use

// This tool serves as a wrapper for utilizing multiple tools. Each tool that can be used must be specified in the tool sections. Only tools in the functions namespace are permitted.
// Ensure that the parameters provided to each tool are valid according to that tool's specification.
namespace multi_tool_use {

// Use this function to run multiple tools simultaneously, but only if they can operate in parallel. Do this even if the prompt suggests using the tools sequentially.
type parallel = (_: {
// The tools to be executed in parallel. NOTE: only functions tools are permitted
tool_uses: {
// The name of the tool to use. The format should either be just the name of the tool, or in the format namespace.function_name for plugin and function tools.
recipient_name: string,
// The parameters to pass to the tool. Ensure these are valid according to the tool's own specifications.
parameters: object,
}[],
}) => any;

} // namespace multi_tool_use
"""

    return message


def format_single_turn(message: ChatMessage) -> str:
    chat: str = f'{start_header_token}{message.role.strip()}{end_header_token}\n\n'
    chat += message.content.strip()
    chat += eot_token
    return chat


def apply_chat_template(messages: List[ChatMessage]) -> str:
    result: str = bos_token
    for msg in messages:
        result += format_single_turn(msg)

    # Add the start of an assistant message for the model to complete.
    result += f'{start_header_token}assistant{end_header_token}\n\n'
    return result


def count_number_of_tokens(tokenizer: Tokenizer, text: str) -> int:
    token_ids = tokenizer.encode(text, allowed_special='all')
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
        print(f'Max length: {max_len}')

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
            print(f'Before cut:\n{messages}')

        has_system_msg = messages[0].role == 'system'
        for i in range(cut_idx + 1):
            if has_system_msg:
                pop_idx = 1
            else:
                pop_idx = 0

            if half_cut and i == cut_idx:
                # Try to cut the content in a single turn to
                sentences = messages[pop_idx].content.split(' ')
                messages[pop_idx].content = ' '.join(sentences[-(len(sentences) // 2) :])
            else:
                messages.pop(pop_idx)

        if verbose:
            print(f'After cut:\n{messages}')
