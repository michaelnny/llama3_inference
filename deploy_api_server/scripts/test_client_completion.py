import argparse
import sys
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


# Simple hack to support running without installing as a package, so we can import the GrpcTritonClient
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from app.utils.triton_client import GrpcTritonClient

chat_template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{0}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


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
        default="llama3_ensemble",
        help="Inference server model name.",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=False,
        default="Fix grammar error in the following sentence:\n\nYou're a very famous comediant that tells great store to entertain people.",
        help="Query text",
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
    
    prompt = chat_template.format(FLAGS.query)

    client = GrpcTritonClient(url=FLAGS.url, verbose=FLAGS.verbose)

    # Display all available models
    print(f"Model repositories: {client.get_model_list()}")
    print(f"Model config: {client.get_model_config(FLAGS.model_name)}")
    print(f"Model statistics: {client.get_model_statistics(FLAGS.model_name)}")

    try:
        result_queue = client.request_completion_streaming(
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
            # seed=1,
        )

        # We then retrieve the results...
        for token in result_queue:
            print(token)
            if token == None:
                break

        print(f"Model statistics: {client.get_model_statistics(FLAGS.model_name)}")

    except InferenceServerException as error:
        print(error)
        sys.exit(1)
