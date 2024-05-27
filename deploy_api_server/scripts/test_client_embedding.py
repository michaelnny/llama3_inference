import argparse
import sys
import torch
from tritonclient.utils import InferenceServerException


# Simple hack to support running without installing as a package, so we can import the GrpcTritonClient
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from app.utils.triton_client import GrpcTritonClient


FLAGS = None


def main():
    FLAGS = parser.parse_args()
    input_texts = [
        "Fix grammar error in the following sentence",
        "Fix grammar error in the following sentence",
        "You are a very famous comedian that tells great stories to entertain people",
        "You are a comedian that enjoys telling stories to cheer up people",
    ]

    client = GrpcTritonClient(url=FLAGS.url, verbose=FLAGS.verbose)

    # Display all available models
    print(f"Model repositories: {client.get_model_list()}")
    print(f"Model config: {client.get_model_config(FLAGS.model_name)}")
    print(f"Model statistics: {client.get_model_statistics(FLAGS.model_name)}")

    try:
        result = client.request_embedding(
            model_name=FLAGS.model_name,
            input=input_texts,
        )

        embeddings = result.get_all_items()

        assert len(embeddings) == 4

        embed_matrix = torch.Tensor(embeddings)

        # one and two are identical
        score_1 = torch.cosine_similarity(embed_matrix[0], embed_matrix[1], dim=0)

        score_2 = torch.cosine_similarity(embed_matrix[0], embed_matrix[2], dim=0)
        score_3 = torch.cosine_similarity(embed_matrix[1], embed_matrix[3], dim=0)

        # last pair are similar
        score_4 = torch.cosine_similarity(embed_matrix[2], embed_matrix[3], dim=0)

        print(score_1)
        print(score_2)
        print(score_3)
        print(score_4)

        assert round(score_1.item(), 4) == 1.0
        assert round(score_2.item(), 4) < 0.3
        assert round(score_3.item(), 4) < 0.3
        assert round(score_4.item(), 4) >= 0.7

        print("PASS")

        print(f"Model statistics: {client.get_model_statistics(FLAGS.model_name)}")

    except InferenceServerException as error:
        print(error)
        sys.exit(1)


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
        default="st_ensemble",
        help="Inference server model name.",
    )

    main()
