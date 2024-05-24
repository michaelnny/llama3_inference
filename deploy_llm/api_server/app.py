import argparse
import asyncio
import json
from typing import AsyncGenerator, Optional
from tritonclient.utils import InferenceServerException

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse


from trtllm_client import GrpcTritonClient, StreamingResponseGenerator
from modeling import ChatCompletionRequest, ChatMessage, ChatCompletionResponse, ChatCompletionResponseChoice, ChatCompletionStreamResponse, DeltaMessage, ChatCompletionResponseStreamChoice 
from utils import apply_chat_template

app = FastAPI()


client: GrpcTritonClient = None


@app.post("/v1/chat/completions")
async def generate(request: ChatCompletionRequest) -> ChatCompletionResponse:
    assert client is not None
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """

    prompt = apply_chat_template(request.messages)

    try:
        result_queue = client.request_streaming(
            model_name=request.model,
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
            async def stream_results() -> AsyncGenerator[bytes, None]:
                for token in result_queue:
                    if token == None:
                        break
                        
                    delta_choices = [ChatCompletionResponseStreamChoice(index=0, delta=DeltaMessage(role="assistant", content=token))]
                    stream_response = ChatCompletionStreamResponse(model=request.model, choices=delta_choices)
                    # yield json.dumps(stream_response.json()).encode("utf-8")

                    # The "data:" + <content> + "\n\n" format is important, as frontend SSE libraries will specifically look for this JSON pattern.
                    json_bytes = stream_response.json().encode("utf-8")
                    yield b"data: " + json_bytes + b"\n\n"

            return StreamingResponse(stream_results(), media_type='text/event-stream')

        # If not streaming, return the generated content directly
        generated_content = []
        for token in result_queue:
            if token is None:
                break
            generated_content.append(token)

        choices = [ChatCompletionResponseChoice(index=i, message=ChatMessage(role="assistant", content=content) ) for i, content in enumerate(generated_content)]
        return ChatCompletionResponse(model=request.model, choices=choices)

    except InferenceServerException as error:
        print(error)
        return JSONResponse({"text": 'error'})


async def main(args):

    global client

    client = GrpcTritonClient(url=args.triton_server_url, verbose=args.verbose)

    try:
        # Try to call the Triton server to ensure everything is fine
        print(f'Triton server model repositories: {client.get_model_list()}')
    except Exception as error:
        raise SystemError(error)

    config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        timeout_keep_alive=args.timeout_keep_alive,
    )
    await uvicorn.Server(config).serve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--timeout_keep_alive", type=int, default=5)
    parser.add_argument(
        "--triton_server_url",
        type=str,
        default="localhost:8001",
        help="Triton server URL for the GRPC protocol",
    )
    parser.add_argument("--verbose", action="store_true", required=False, default=False)
    args = parser.parse_args()

    asyncio.run(main(args))
