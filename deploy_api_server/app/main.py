import argparse
import logging
from typing import List, AsyncGenerator, Union
from contextlib import asynccontextmanager
import traceback
import json
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY
from tritonclient.utils import InferenceServerException
from fastapi.encoders import jsonable_encoder

from utils.triton_client import GrpcTritonClient, StreamingResponseGenerator
from utils.schema import (
    ErrorResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    Embedding,
    ChatCompletionRequest,
    ChatMessage,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponse,
    DeltaMessage,
    ChatCompletionResponseStreamChoice,
)
from utils.prompt import (
    maybe_prune_chat_history,
    insert_functions_to_system_message,
    apply_chat_template,
    decode_json_strings,
    extract_tool_calls,
)
from utils.server import get_triton_server_model_name
from utils.tokenizer import Tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global variables
client: GrpcTritonClient = None
tokenizer: Tokenizer = None
max_input_len: int = 512


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info('Initialize application')

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

    yield

    logger.info('Closing connection to database')
    await app.state.db_pool.close()
    logger.info('Exit application')
    pass


app = FastAPI(lifespan=lifespan)

# Allow all access in local development
origins = ['*']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content={'detail': jsonable_encoder(exc.errors(), exclude={'input'})},
    )


@app.post(
    '/v1/chat/completions',
    response_model=Union[ChatCompletionResponse, ChatCompletionResponseStreamChoice, ErrorResponse],
)
async def chat_completions(
    request: ChatCompletionRequest,
) -> Union[ChatCompletionResponse, ChatCompletionResponseStreamChoice, ErrorResponse]:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - messages: a list of chat history.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """

    try:

        use_stream = request.stream

        # TODO: merge consecutive 'tool' results into a single message.

        # Serialize content, this will tidy up function calls structure into normal 'content'
        for item in request.messages:
            item.serialize_content()

        if request.tools:
            use_stream = False
            # Inject tools to system message
            tools_sys_msg = insert_functions_to_system_message(request.tools)
            if request.messages[0].role == 'system':
                mixed_sys_msg = tools_sys_msg + '\n\n' + request.messages[0].content
                request.messages[0].content = mixed_sys_msg
            else:
                request.messages.insert(0, ChatMessage(role='system', content=tools_sys_msg))

        maybe_prune_chat_history(tokenizer, request.messages, max_input_len)
        prompt = apply_chat_template(request.messages)

        result_queue: StreamingResponseGenerator = client.request_completion_streaming(
            model_name=get_triton_server_model_name(request.model, for_embedding=False),
            prompt=prompt,
            stop_words=request.stop,
            max_tokens=request.max_tokens,
            stream=use_stream,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            length_penalty=request.length_penalty,
            seed=request.seed,
        )

        if use_stream:

            async def stream_results() -> AsyncGenerator[bytes, None]:
                try:
                    for token in result_queue:
                        if token is None:
                            break

                        delta_choices = [
                            ChatCompletionResponseStreamChoice(
                                index=0,
                                delta=DeltaMessage(role='assistant', content=token),
                            )
                        ]
                        stream_response = ChatCompletionStreamResponse(model=request.model, choices=delta_choices)

                        # The "data:" + <content> + "\n\n" format is important, as frontend SSE libraries will specifically look for this JSON pattern.
                        # json_bytes = stream_response.json().encode("utf-8")
                        # yield b"data: " + json_bytes + b"\n\n"
                        yield 'data: ' + str(stream_response.json()) + '\n\n'
                except Exception as error:
                    client.stop_stream()
                    traceback.print_exception(error)
                    error_response = ErrorResponse(message=f'Streaming error: {str(error)}', code=500)
                    yield 'data: ' + str(error_response.json()) + '\n\n'

            return StreamingResponse(stream_results(), media_type='text/event-stream')
        else:
            # If not streaming, return the generated content directly
            generated_content = result_queue.get_all_items()
            if request.tools:
                tool_calls = extract_tool_calls(generated_content)
                if tool_calls:
                    # create tool call response
                    response_message = ChatMessage(role='assistant', content=None, tool_calls=tool_calls)
                    choices = [ChatCompletionResponseChoice(index=0, message=response_message)]
                    return ChatCompletionResponse(model=request.model, choices=choices)

            response_message = ChatMessage(role='assistant', content=generated_content)
            choices = [ChatCompletionResponseChoice(index=0, message=response_message)]
            return ChatCompletionResponse(model=request.model, choices=choices)
    except InferenceServerException as error:
        traceback.print_exception(error)
        print(error)
        return ErrorResponse(
            object='error',
            message=f'Error when try to make inference call: {str(error)}',
            code=500,
        )
    except Exception as error:
        traceback.print_exception(error)
        print(error)
        return ErrorResponse(object='error', message=f'Unknown error: {str(error)}', code=500)


@app.post('/v1/embeddings', response_model=Union[EmbeddingResponse, ErrorResponse])
async def create_embeddings(
    request: EmbeddingRequest,
) -> Union[EmbeddingResponse, ErrorResponse]:
    """Generate embeddings for the request.

    The request should be a JSON object with the following fields:
    - input: the input text to compute embedding.
    - name: embedding model name.
    """

    try:
        result_queue = client.request_embedding(
            model_name=get_triton_server_model_name(request.model, for_embedding=True),
            input=request.input,
        )

        result = result_queue.get_all_items()

        if result is None:
            return ErrorResponse(
                object='error',
                message='Call remote embedding function failed without return any value',
                code=500,
            )
        else:
            embeddings = [Embedding(index=i, embedding=result[i]) for i in range(len(result))]
            response = EmbeddingResponse(data=embeddings)
            return response
    except InferenceServerException as error:
        traceback.print_exception(error)
        print(error)
        return ErrorResponse(
            object='error',
            message=f'Error when try to make inference call: {str(error)}',
            code=500,
        )
    except Exception as error:
        traceback.print_exception(error)
        print(error)
        return ErrorResponse(object='error', message=f'Unknown error: {str(error)}', code=500)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=3000)
    parser.add_argument('--timeout-keep-alive', type=int, default=5)
    parser.add_argument(
        '--triton-server-url',
        type=str,
        default='localhost:8001',
        help='Triton server URL for the GRPC protocol',
    )
    parser.add_argument(
        '--tokenizer-path',
        type=str,
        default='./utils/tokenizer.model',
        help='Path for the tiktoken tokenizer checkpoint',
    )
    parser.add_argument(
        '--max-input-len',
        type=int,
        default=512,
        help='Limit of the input token length',
    )
    parser.add_argument('--verbose', action='store_true', required=False, default=False)
    args = parser.parse_args()

    # Start the Uvicorn server
    uvicorn.run(app, host=args.host, port=args.port, log_level='info', timeout_keep_alive=5)
