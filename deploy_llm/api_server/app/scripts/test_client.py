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


# Simple hack to support running without installing as a package, so we can import the tokenizer
from pathlib import Path
import sys
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from utils.trtllm_client import StreamingResponseGenerator, GrpcTritonClient

chat_template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{0}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


FLAGS = None


def async_stream_send(
    triton_client, query_text, model_name, max_gen_len, stream, request_id
):

    prompt = chat_template.format(query_text)
    input_prompt = np.asarray([prompt]).astype(object)  # .reshape((1, -1))
    input_max_len = np.array([max_gen_len]).astype(np.int32).reshape((1, -1))
    input_stop_words = np.asarray(["<|eot_id|>"]).astype(object).reshape((1, -1))
    input_stream = np.asarray([stream]).astype(bool).reshape((1, -1))

    inputs = []

    inputs.append(grpcclient.InferInput("text_input", input_prompt.shape, "BYTES"))
    inputs.append(grpcclient.InferInput("max_tokens", input_max_len.shape, "INT32"))
    inputs.append(grpcclient.InferInput("stop_words", input_stop_words.shape, "BYTES"))
    inputs.append(grpcclient.InferInput("stream", input_stream.shape, "BOOL"))
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
        default="Fix grammar error in the following sentence:\n\nYou're a very famous comediant that tells great store to entertain people.",
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

    prompt = chat_template.format(FLAGS.query_text)

    client = GrpcTritonClient(url=FLAGS.url, verbose=FLAGS.verbose)

    # Display all available models
    print(f'Model repositories: {client.get_model_list()}')
    print(f'Model config: {client.get_model_config(FLAGS.model_name)}')
    print(f'Model statistics: {client.get_model_statistics(FLAGS.model_name)}')
    
    try:
        result_queue = client.request_streaming(
            model_name=FLAGS.model_name,
            prompt=prompt,
            stop_words=["</s>", "<|end_of_text|>", "<|eot_id|>"],
            max_tokens=FLAGS.max_gen_len,
            stream=FLAGS.stream,
            # temperature=0,
            # top_k=1,
            # top_p=0,
            # repetition_penalty=1,
            # length_penalty=1.0,
            # random_seed=1,
        )

        # We then retrieve the results...
        for token in result_queue:
            print(token)
            if token == None:
                break


        print(f'Model statistics: {client.get_model_statistics(FLAGS.model_name)}')

    except InferenceServerException as error:
        print(error)
        sys.exit(1)