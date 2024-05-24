#!/usr/bin/env python
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import queue
import sys
import uuid
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Type, Union
import json
import google.protobuf.json_format
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.grpc.service_pb2 import ModelInferResponse
from tritonclient.utils import InferenceServerException



chat_template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{0}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

STOP_WORDS = ["</s>", "<|end_of_text|>", "<|eot_id|>"]

FLAGS = None







class StreamingResponseGenerator(queue.Queue[Optional[str]]):
    """A Generator that provides the inference results from an LLM."""

    def __init__(
        self, client: grpcclient.InferenceServerClient, request_id: str, force_batch: bool
    ) -> None:
        """Instantiate the generator class."""
        super().__init__()
        self._client = client
        self.request_id = request_id
        self._batch = force_batch

    def __iter__(self) -> "StreamingResponseGenerator":
        """Return self as a generator."""
        return self

    def __next__(self) -> str:
        """Return the next retrieved token."""
        val = self.get()
        if val is None or val in STOP_WORDS:
            self._stop_stream()
            raise StopIteration()
        return val

    def _stop_stream(self) -> None:
        """Drain and shutdown the Triton stream."""
        # self._client.stop_stream(
        #     "tensorrt_llm", self.request_id, signal=not self._batch
        # )
        self._client.stop_stream()



# Define the callback function. Note the last two parameters should be
# result and error. InferenceServerClient would povide the results of an
# inference as grpcclient.InferResult in result. For successful
# inference, error will be None, otherwise it will be an object of
# tritonclientutils.InferenceServerException holding the error details
def callback(result_queue: queue.Queue[Union[Optional[Dict[str, str]], str]], result, error):
    if error:
        result_queue.put(error)
    else:
        response_raw = result.get_response(as_json=True)
        if "outputs" in response_raw:
            # the very last response might have no output, just the final flag
            response = process_result(response_raw)
            
            if response in STOP_WORDS:
                result_queue.put(None)
            else:
                result_queue.put(response)
        
        if response_raw["parameters"]["triton_final_response"]["bool_param"]:
            # end of the generation
            result_queue.put(None)


def async_stream_send(
    triton_client, query_text, model_name, max_gen_len, stream, request_id
):
    
    prompt = chat_template.format(query_text)
    input_prompt = np.asarray([prompt]).astype(object).reshape((1, -1))
    input_max_len = np.array([max_gen_len]).astype(np.int32).reshape((1, -1))
    input_stop_words = np.asarray(["</s>", "<|end_of_text|>", "<|eot_id|>"]).astype(object).reshape((1, -1))
    input_stream = np.asarray([stream]).astype(bool).reshape((1, -1))
    
    inputs = []

    inputs.append(grpcclient.InferInput('text_input', input_prompt.shape, "BYTES"))
    inputs.append(grpcclient.InferInput('max_tokens', input_max_len.shape, "INT32"))
    inputs.append(grpcclient.InferInput('stop_words', input_stop_words.shape, "BYTES"))
    inputs.append(grpcclient.InferInput('stream', input_stream.shape, "BOOL"))
    inputs[0].set_data_from_numpy(input_prompt)
    inputs[1].set_data_from_numpy(input_max_len)
    inputs[2].set_data_from_numpy(input_stop_words)
    inputs[3].set_data_from_numpy(input_stream)

    outputs = []
    outputs.append(grpcclient.InferRequestedOutput("text_output"))

    triton_client.async_stream_infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
            request_id=request_id,
        )


def process_result(result: Dict[str, str]) -> str:
    """Post-process the result from the server."""
    message = ModelInferResponse()
    generated_text: str = ""
    google.protobuf.json_format.Parse(json.dumps(result), message)
    infer_result = grpcclient.InferResult(message)
    np_res = infer_result.as_numpy("text_output")

    generated_text = ""
    if np_res is not None:
        generated_text = "".join([token.decode() for token in np_res])

    return generated_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )
    parser.add_argument(
        "--url",
        type=str,
        required=False,
        default="localhost:8001",
        help="Inference server URL and it gRPC port. Default is localhost:8001.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=False,
        default="ensemble",
        help="Inference server model name.",
    )
    parser.add_argument(
        "--query-text",
        type=str,
        required=False,
        default="Tell me a short story about a dog and his cat friend.",
        help="Inference server model name.",
    )
    parser.add_argument(
        "--max-gen-len",
        type=int,
        required=False,
        default=256,
        help="",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        required=False,
        default=False,
        help="Stream output.",
    )
    parser.add_argument(
        "--stream-timeout",
        type=float,
        required=False,
        default=None,
        help="Stream timeout in seconds. Default is None.",
    )

    FLAGS = parser.parse_args()

    string_result0_list = []


    

    # It is advisable to use client object within with..as clause
    # when sending streaming requests. This ensures the client
    # is closed when the block inside with exits.
    with grpcclient.InferenceServerClient(
        url=FLAGS.url, verbose=FLAGS.verbose
    ) as triton_client:
        try:
            
            request_id = '1234565'
            result_queue = StreamingResponseGenerator(triton_client, request_id, False)

            # Establish stream
            triton_client.start_stream(
                callback=partial(callback, result_queue),
                stream_timeout=FLAGS.stream_timeout,
            )

            # Now send the inference sequences...
            async_stream_send(
                triton_client,
                FLAGS.query_text,
                FLAGS.model_name,
                FLAGS.max_gen_len,
                FLAGS.stream,
                request_id,
            )
            
        except InferenceServerException as error:
            print(error)
            sys.exit(1)

        # We then retrieve the results...
        for token in result_queue:
            print(token)
            if token == None:
                break

    print("PASS: Sequence")
