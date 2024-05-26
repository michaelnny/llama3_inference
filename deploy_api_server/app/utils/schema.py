"""Common pydantic data models for the API request/responses"""

from typing import Literal, Optional, List, Dict, Any, Union
import time
import bleach
import shortuuid
from pydantic import BaseModel, Field, validator


generate_random_id = lambda: shortuuid.random()


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    code: int


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


Role = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    role: Role
    content: str

    @validator("role")
    def validate_role(cls, value):
        """Field validator function to validate values of the field role"""
        value = bleach.clean(value, strip=True)
        valid_roles = {"user", "assistant", "system"}
        if value.lower() not in valid_roles:
            raise ValueError("Role must be one of 'user', 'assistant', or 'system'")
        return value.lower()

    @validator("content")
    def sanitize_content(cls, v):
        """Field validator function to santize user populated fields from HTML"""
        return bleach.clean(v, strip=True)


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = -1
    max_tokens: Optional[int] = 1024
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    repetition_penalty: Optional[float] = 1
    length_penalty: Optional[float] = 1.0
    seed: Optional[int] = 1

    @validator("messages")
    def validate_messages(cls, v):
        """Field validator function to check valid message turns"""
        if v is None or len(v) == 0:
            raise ValueError("Chat messages can not be none or empty")
        else:
            if v[0].role == "system":
                start_idx = 1
            else:
                start_idx = 0
            if not all([msg.role == "user" for msg in v[start_idx::2]]) or not all(
                [msg.role == "assistant" for msg in v[start_idx + 1 :: 2]]
            ):
                raise ValueError(
                    'Chat messages should start with either "system" or "user", then followed by "assistant", and alternating (user/assistant/...)'
                )
            else:
                return v


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    logprobs: Optional[List[float]] = None
    finish_reason: Optional[Literal["stop", "length"]] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{generate_random_id()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Optional[UsageInfo] = None


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    logprobs: Optional[List[float]] = None
    finish_reason: Optional[Literal["stop", "length"]] = "stop"


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{generate_random_id()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]


class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    encoding_format: Optional[str] = "float"


class Embedding(BaseModel):
    index: int
    embedding: List[float]
    object: Optional[str] = "embedding"


class EmbeddingResponse(BaseModel):
    data: List[Embedding]
