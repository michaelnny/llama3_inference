from typing import Literal, Optional, List, Dict, Any, Union

from utils.modeling import ChatMessage 

bos_token = "<|begin_of_text|>"
start_header_token = "<|start_header_id|>"
end_header_token = "<|end_header_id|>"
eot_token = "<|eot_id|>" # end of turn

def apply_chat_template(messages: List[ChatMessage]) -> str:
    assert messages is not None and len(messages) > 0, 'Invalid messages'
    assert messages[-1].role == 'user', 'Messages must ends with user turn'

    result: str = ''

    result += bos_token
    for msg in messages:
        result += f'{start_header_token}{msg.role.strip()}{end_header_token}\n\n'
        result += msg.content.strip()
        result += eot_token
    
    # Add the start of an assistant message for the model to complete.
    result += f'{start_header_token}assistant{end_header_token}\n\n'
    return result


TRITON_MODEL_NAME_MAP = {
    'llama3': 'ensemble',
}

DEFAULT_MODEL_NAME = 'ensemble'


def get_triton_server_model_name(model_name: str) -> str:
    # Map model name to triton server model name
    if model_name is None or model_name not in TRITON_MODEL_NAME_MAP:
        return DEFAULT_MODEL_NAME
    else:
        return TRITON_MODEL_NAME_MAP[model_name]
