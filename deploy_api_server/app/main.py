import argparse
from typing import AsyncGenerator, Union
from tritonclient.utils import InferenceServerException

import uvicorn
from fastapi import FastAPI 
from fastapi.responses import StreamingResponse


from utils.trtllm_client import GrpcTritonClient, StreamingResponseGenerator
from utils.modeling import ErrorResponse, ChatCompletionRequest, ChatMessage, ChatCompletionResponse, ChatCompletionResponseChoice, ChatCompletionStreamResponse, DeltaMessage, ChatCompletionResponseStreamChoice 
from utils.chat import check_valid_chat_pattern, maybe_prune_chat_history, get_triton_server_model_name, apply_chat_template
from utils.tokenizer import Tokenizer

app = FastAPI()


client: GrpcTritonClient = None
tokenizer: Tokenizer = None
max_input_len: int = 512


async def stream_results(model_name: str, result_queue: StreamingResponseGenerator) -> AsyncGenerator[bytes, None]:
    try:
        for token in result_queue:
            if token == None:
                break
            
            delta_choices = [ChatCompletionResponseStreamChoice(index=0, delta=DeltaMessage(role="assistant", content=token))]
            stream_response = ChatCompletionStreamResponse(model=model_name, choices=delta_choices)
            # yield json.dumps(stream_response.json()).encode("utf-8")

            # The "data:" + <content> + "\n\n" format is important, as frontend SSE libraries will specifically look for this JSON pattern.
            json_bytes = stream_response.json().encode("utf-8")
            yield b"data: " + json_bytes + b"\n\n"
    except Exception as error:
        error_response = ErrorResponse(message=f"Streaming error: {str(error)}", code=500)
        json_bytes = error_response.json().encode("utf-8")
        yield b"data: " + json_bytes + b"\n\n"


@app.post("/v1/chat/completions")
async def generate(request: ChatCompletionRequest) -> Union[ChatCompletionResponse, ChatCompletionResponseStreamChoice, ErrorResponse]:
    assert client is not None
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """

    # Check valid chat patterns, should start with either "system" or "user", then followed by "assistant", and alternating (user/assistant/...)
    # In addition, also handle long context
    try:
        check_valid_chat_pattern(request.messages)
    except ValueError as error:
        print(error)
        return ErrorResponse(object='error', message=f'Chat messages error: {str(error)}', code=400)
    finally:
        maybe_prune_chat_history(tokenizer, request.messages, max_input_len)
        prompt = apply_chat_template(request.messages)

    try:
        result_queue: StreamingResponseGenerator = client.request_streaming(
            model_name=get_triton_server_model_name(request.model),
            prompt=prompt,
            stop_words=request.stop,
            max_tokens=request.max_tokens,
            stream=request.stream,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            length_penalty=request.length_penalty,
            random_seed=request.random_seed,
        )

        if request.stream:
            return StreamingResponse(stream_results(request.model, result_queue), media_type='text/event-stream')

        # If not streaming, return the generated content directly
        generated_content = []
        for response in result_queue:
            if response is None:
                break
            generated_content.append(response)

        choices = [ChatCompletionResponseChoice(index=i, message=ChatMessage(role="assistant", content=content) ) for i, content in enumerate(generated_content)]
        return ChatCompletionResponse(model=request.model, choices=choices)
    except InferenceServerException as error:
        print(error)
        return ErrorResponse(object='error', message=f'Error when try to make inference call: {str(error)}', code=500)
    except Exception as error:
        print(error)
        return ErrorResponse(object='error', message=f'Unknown error: {str(error)}', code=400)


@app.on_event("startup")
async def startup_event():
    global client
    global tokenizer
    global max_input_len

    args = parser.parse_args()
    max_input_len = max(max_input_len, args.max_input_len)
    tokenizer = Tokenizer(args.tokenizer_path)
    client = GrpcTritonClient(url=args.triton_server_url, verbose=args.verbose)

    try:
        # Try to call the Triton server to ensure everything is fine
        print(f'Triton server model repositories: {client.get_model_list()}')
    except Exception as error:
        raise SystemError(error)
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--timeout-keep-alive", type=int, default=5)
    parser.add_argument(
        "--triton-server-url",
        type=str,
        default="localhost:8001",
        help="Triton server URL for the GRPC protocol",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="./utils/tokenizer.model",
        help="Path for the tiktoken tokenizer checkpoint",
    )
    parser.add_argument("--max-input-len", type=int, default=512, help="Limit of the input token length",)
    parser.add_argument("--verbose", action="store_true", required=False, default=False)
    args = parser.parse_args()

    # Start the Uvicorn server
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", timeout_keep_alive=5)

