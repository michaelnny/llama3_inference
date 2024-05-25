"""

Code adapted from NVIDIA's GenerativeAIExamples:

GenerativeAIExamples/RetrievalAugmentedGeneration/llm-inference-server/model_server_client/trt_llm.py

https://github.com/NVIDIA/GenerativeAIExamples
"""

import abc
import json
import queue
import random
import time
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Type, Union

import google.protobuf.json_format
import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.grpc.service_pb2 import ModelInferResponse
from tritonclient.utils import np_to_triton_dtype


DEFAULT_STOP_WORDS = ["</s>", "<|end_of_text|>", "<|eot_id|>"]

class StreamingResponseGenerator(queue.Queue[Optional[str]]):
    """A Generator that provides the inference results from an LLM."""

    def __init__(
        self,
        client: "GrpcTritonClient",
        request_id: str,
        force_batch: bool = False,
        stop_words: List[str] = ["</s>", "<|end_of_text|>", "<|eot_id|>"],
    ) -> None:
        """Instantiate the generator class."""
        super().__init__()
        self._client = client
        self.request_id = request_id
        self._batch = force_batch
        self.stop_words = stop_words

    def __iter__(self) -> "StreamingResponseGenerator":
        """Return self as a generator."""
        return self

    def __next__(self) -> str:
        """Return the next retrieved token."""
        val = self.get()
        if val is None or val in self.stop_words:
            self._stop_stream()
            raise StopIteration()
        return val

    def _stop_stream(self) -> None:
        """Drain and shutdown the Triton stream."""
        self._client.stop_stream(
            "tensorrt_llm", self.request_id, signal=not self._batch
        )


class BaseTritonClient(abc.ABC):
    """An abstraction of the connection to a triton inference server."""

    def __init__(self, url: str, verbose: bool = False) -> None:
        """Initialize the client."""
        self._client = self._inference_server_client(url=url, verbose=verbose)

    @property
    @abc.abstractmethod
    def _inference_server_client(
        self,
    ) -> Union[
        Type[grpcclient.InferenceServerClient], Type[httpclient.InferenceServerClient]
    ]:
        """Return the prefered InferenceServerClient class."""

    @property
    @abc.abstractmethod
    def _infer_input(
        self,
    ) -> Union[Type[grpcclient.InferInput], Type[httpclient.InferInput]]:
        """Return the preferred InferInput."""

    @property
    @abc.abstractmethod
    def _infer_output(
        self,
    ) -> Union[
        Type[grpcclient.InferRequestedOutput], Type[httpclient.InferRequestedOutput]
    ]:
        """Return the preferred InferRequestedOutput."""

    def load_model(self, model_name: str, timeout: int = 3000) -> None:
        """Load a model into the server."""
        if self._client.is_model_ready(model_name):
            return

        self._client.load_model(model_name)
        t0 = time.perf_counter()
        t1 = t0
        while not self._client.is_model_ready(model_name) and t1 - t0 < timeout:
            t1 = time.perf_counter()

        if not self._client.is_model_ready(model_name):
            raise RuntimeError(f"Failed to load {model_name} on Triton in {timeout}s")

    def get_model_list(self) -> List[dict]:
        """Get a list of models loaded in the triton server."""
        res = self._client.get_model_repository_index(as_json=True)
        models = res["models"] if "models" in res else []
        return models

    def get_model_config(self, model_name: str) -> dict:
        """Get the model config."""
        return self._client.get_model_config(model_name, as_json=True)["config"]

    def get_model_statistics(self, model_name: str) -> dict:
        """Get the model statistics."""
        return self._client.get_inference_statistics(model_name, as_json=True)[
            "model_stats"
        ]

    def _generate_stop_signals(
        self,
    ) -> List[Union[grpcclient.InferInput, httpclient.InferInput]]:
        """Generate the signal to stop the stream."""
        inputs = [
            self._infer_input("input_ids", [1, 1], "INT32"),
            self._infer_input("input_lengths", [1, 1], "INT32"),
            self._infer_input("request_output_len", [1, 1], "UINT32"),
            self._infer_input("stop", [1, 1], "BOOL"),
        ]
        inputs[0].set_data_from_numpy(np.empty([1, 1], dtype=np.int32))
        inputs[1].set_data_from_numpy(np.zeros([1, 1], dtype=np.int32))
        inputs[2].set_data_from_numpy(np.array([[0]], dtype=np.uint32))
        inputs[3].set_data_from_numpy(np.array([[True]], dtype="bool"))
        return inputs

    def _generate_outputs(
        self,
    ) -> List[Union[grpcclient.InferRequestedOutput, httpclient.InferRequestedOutput]]:
        """Generate the expected output structure."""
        return [self._infer_output("text_output")]

    def _prepare_tensor(
        self, name: str, input_data: Any
    ) -> Union[grpcclient.InferInput, httpclient.InferInput]:
        """Prepare an input data structure."""
        t = self._infer_input(
            name, input_data.shape, np_to_triton_dtype(input_data.dtype)
        )
        t.set_data_from_numpy(input_data)
        return t

    def _generate_inputs(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        prompt: str,
        stop_words: List[str] = ["</s>", "<|end_of_text|>", "<|eot_id|>"],
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_k: float = 1,
        top_p: float = 0,
        beam_width: int = 1,
        repetition_penalty: float = 1,
        length_penalty: float = 1.0,
        stream: bool = True,
        random_seed: int = 1,
    ) -> List[Union[grpcclient.InferInput, httpclient.InferInput]]:
        """Create the input for the triton inference server."""
        input_query = np.array([prompt]).astype(object).reshape((1, -1))
        input_stop_words = np.array([stop_words]).astype(object).reshape((1, -1))
        input_max_tokens = np.array([max_tokens]).astype(np.int32).reshape((1, -1))
        input_top_k = np.array([top_k]).astype(np.int32).reshape((1, -1))
        input_top_p = np.array([top_p]).astype(np.float32).reshape((1, -1))
        input_temperature = np.array([temperature]).astype(np.float32).reshape((1, -1))
        input_len_penalty = (
            np.array([length_penalty]).astype(np.float32).reshape((1, -1))
        )
        input_repeat_penalty = (
            np.array([repetition_penalty]).astype(np.float32).reshape((1, -1))
        )
        input_random_seed = np.array([random_seed]).astype(np.uint64).reshape((1, -1))
        input_beam_width = np.array([beam_width]).astype(np.int32).reshape((1, -1))
        input_stream = np.array([stream], dtype=bool).reshape((1, -1))
        input_stream = np.array([stream], dtype=bool).reshape((1, -1))

        inputs = [
            self._prepare_tensor("text_input", input_query),
            self._prepare_tensor("stop_words", input_stop_words),
            self._prepare_tensor("max_tokens", input_max_tokens),
            self._prepare_tensor("top_k", input_top_k),
            self._prepare_tensor("top_p", input_top_p),
            self._prepare_tensor("temperature", input_temperature),
            self._prepare_tensor("length_penalty", input_len_penalty),
            self._prepare_tensor("repetition_penalty", input_repeat_penalty),
            self._prepare_tensor("random_seed", input_random_seed),
            self._prepare_tensor("beam_width", input_beam_width),
            self._prepare_tensor("stream", input_stream),
        ]
        return inputs

    def _trim_batch_response(self, result_str: str, stop_words: List[str]) -> str:
        """Trim the resulting response from a batch request by removing provided prompt and extra generated text."""
        # extract the generated part of the prompt

        # for llama2
        if "[/INST]" in result_str:
            split = result_str.split("[/INST]", 1)
            generated = split[-1]
        else:
            generated = result_str

        end_token_idx = -1
        for stop_word in stop_words:
            end_token_idx = generated.find(stop_word)
            if end_token_idx != -1:
                return generated[:end_token_idx].strip()
        return generated


class GrpcTritonClient(BaseTritonClient):
    """GRPC connection to a triton inference server."""

    @property
    def _inference_server_client(
        self,
    ) -> Type[grpcclient.InferenceServerClient]:
        """Return the prefered InferenceServerClient class."""
        return grpcclient.InferenceServerClient  # type: ignore

    @property
    def _infer_input(self) -> Type[grpcclient.InferInput]:
        """Return the preferred InferInput."""
        return grpcclient.InferInput  # type: ignore

    @property
    def _infer_output(
        self,
    ) -> Type[grpcclient.InferRequestedOutput]:
        """Return the preferred InferRequestedOutput."""
        return grpcclient.InferRequestedOutput  # type: ignore

    def _send_stop_signals(self, model_name: str, request_id: str) -> None:
        """Send the stop signal to the Triton Inference server."""
        stop_inputs = self._generate_stop_signals()
        self._client.async_stream_infer(
            model_name,
            stop_inputs,
            request_id=request_id,
            parameters={"Streaming": True},
        )

    @staticmethod
    def _process_result(result: Dict[str, str]) -> str:
        """Post-process the result from the server."""
        message = ModelInferResponse()
        google.protobuf.json_format.Parse(json.dumps(result), message)
        infer_result = grpcclient.InferResult(message)
        np_res = infer_result.as_numpy("text_output")
        # items = infer_result.as_numpy("input_token_len")
        # print(items)
        generated_text: str = ""
        if np_res is not None:
            generated_text = "".join([token.decode() for token in np_res])

        return generated_text

    def _stream_callback(
        self,
        result_queue: StreamingResponseGenerator,
        force_batch: bool,
        result: Any,
        error: str,
    ) -> None:
        """Add streamed result to queue."""
        if error:
            result_queue.put(error)
        else:
            response_raw = result.get_response(as_json=True)
            if "outputs" in response_raw:
                # the very last response might have no output, just the final flag
                response = self._process_result(response_raw)
                if force_batch:
                    response = self._trim_batch_response(
                        response, result_queue.stop_words
                    )

                if response in result_queue.stop_words:
                    result_queue.put(None)
                else:
                    result_queue.put(response)

            if response_raw["parameters"]["triton_final_response"]["bool_param"]:
                # end of the generation
                result_queue.put(None)

    # pylint: disable-next=too-many-arguments
    def _send_prompt_streaming(
        self,
        model_name: str,
        request_inputs: Any,
        request_outputs: Optional[Any],
        request_id: str,
        result_queue: StreamingResponseGenerator,
        force_batch: bool = False,
    ) -> None:
        """Send the prompt and start streaming the result."""
        self._client.start_stream(
            callback=partial(self._stream_callback, result_queue, force_batch)
        )
        self._client.async_stream_infer(
            model_name=model_name,
            inputs=request_inputs,
            outputs=request_outputs,
            request_id=request_id,
        )

    def request_streaming(
        self,
        model_name: str,
        prompt: str,
        stop_words: Union[str, List[str]] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_k: float = 1,
        top_p: float = 0,
        beam_width: int = 1,
        repetition_penalty: float = 1,
        length_penalty: float = 1.0,
        stream: bool = True,
        random_seed: int = 1,
        request_id: Optional[str] = None,
    ) -> StreamingResponseGenerator:
        """Request a streaming connection."""
        if not self._client.is_model_ready(model_name):
            raise RuntimeError("Cannot request streaming, model is not loaded")

        if not request_id:
            request_id = str(random.randint(1, 9999999))  # nosec

        if stop_words is None:
            stop_words = DEFAULT_STOP_WORDS
        elif isinstance(stop_words, str):
            stop_words = [stop_words]

        if max_tokens is None:
            max_tokens = 1024

        result_queue = StreamingResponseGenerator(
            self, request_id, force_batch=not stream, stop_words=stop_words
        )
        inputs = self._generate_inputs(
            prompt=prompt,
            stop_words=stop_words,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            beam_width=beam_width,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            stream=stream,
            random_seed=random_seed,
        )
        outputs = self._generate_outputs()
        self._send_prompt_streaming(
            model_name,
            inputs,
            outputs,
            request_id,
            result_queue,
            force_batch=not stream,
        )
        return result_queue

    def stop_stream(
        self, model_name: str, request_id: str, signal: bool = True
    ) -> None:
        """Close the streaming connection."""
        if signal:
            self._send_stop_signals(model_name, request_id)
        self._client.stop_stream()
