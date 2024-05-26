# Copyright (c) 2023-2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any
import functools
import inspect
import logging
import os
import typing
import argparse
import torch
import torch.nn.functional as F
from google.protobuf import text_format

from transformers import AutoModel, AutoTokenizer
from tritonclient.grpc.model_config_pb2 import (
    DataType,
    ModelConfig,
    ModelInput,
    ModelOutput,
    ModelOptimizationPolicy,
    ModelInstanceGroup,
    ModelParameter,
)
from tritonclient.utils import np_to_triton_dtype

try:
    import onnx
except ImportError as exc:
    raise RuntimeError(
        "Please install onnx to use this feature. Run `pip3 install onnx`"
    ) from exc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomEmbeddingModel(torch.nn.Module):
    # pylint: disable=abstract-method
    def __init__(self, model_name: str):
        super().__init__()

        self.inner_model = AutoModel.from_pretrained(model_name)
        self._output_dim = self.inner_model.config.hidden_size

        sig = inspect.signature(self.inner_model.forward)

        ordered_list_keys = list(sig.parameters.keys())
        if ordered_list_keys[0] == "self":
            ordered_list_keys = ordered_list_keys[1:]

        # Save the idx of the attention mask because exporting prefers arguments over kwargs
        self._attention_mask_idx = ordered_list_keys.index("attention_mask")

        # Wrap the original function so the export can find the original signature
        @functools.wraps(self.inner_model.forward)
        def forward(*args, **kwargs):
            return self._forward(*args, **kwargs)

        self.forward = forward

    @property
    def output_dim(self):
        return self._output_dim

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        # Adapted from https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
        # First element of model_output contains all token embeddings
        last_hidden_state = model_output[
            "last_hidden_state"
        ]  # [batch_size, seq_length, hidden_size]

        alternate = True

        if alternate:
            last_hidden = last_hidden_state.masked_fill(
                ~attention_mask[..., None].bool(), 0.0
            )
            return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

        # Transpose to make broadcasting possible
        last_hidden_state = torch.transpose(
            last_hidden_state, 0, 2
        )  # [hidden_size, seq_length, batch_size]

        input_mask_expanded = torch.transpose(
            attention_mask.unsqueeze(-1).float(), 0, 2
        )  # [1, seq_length, batch_size]

        num = torch.sum(
            last_hidden_state * input_mask_expanded, 1
        )  # [hidden_size, batch_size]
        denom = torch.clamp(input_mask_expanded.sum(1), min=1e-9)  # [1, batch_size]

        return torch.transpose(num / denom, 0, 1)  # [batch_size, hidden_size]

    def normalize(self, embeddings):

        alternate = False

        if alternate:
            return F.normalize(embeddings, p=2, dim=1)

        # Use the same trick here to broadcast to avoid using the expand operator which breaks dynamic axes
        denom = torch.transpose(
            embeddings.norm(2, 1, keepdim=True).clamp_min(1e-12), 0, 1
        )

        return torch.transpose(torch.transpose(embeddings, 0, 1) / denom, 0, 1)

    def _forward(self, *args, **kwargs):

        if "attention_mask" in kwargs:
            attention_mask = kwargs["attention_mask"]
        elif len(args) > self._attention_mask_idx:
            # Lookup from positional
            attention_mask = args[self._attention_mask_idx]
        else:
            raise RuntimeError("Cannot determine attention mask")

        model_outputs = self.inner_model(*args, **kwargs)

        sentence_embeddings = self.mean_pooling(model_outputs, attention_mask)

        sentence_embeddings = self.normalize(sentence_embeddings)

        return sentence_embeddings


def export_onnx_model(model, sample_input: dict, output_model_path: str):

    # Ensure our input is a dictionary, not a batch encoding
    args = (dict(sample_input.items()),)

    inspect.signature(model.forward)

    torch.onnx.export(
        model,
        args,
        output_model_path,
        opset_version=14,
        input_names=["input_ids", "attention_mask"],
        output_names=["output"],
        dynamic_axes={
            "input_ids": {
                0: "batch_size",
                1: "seq_length",
            },  # variable length axes
            "attention_mask": {
                0: "batch_size",
                1: "seq_length",
            },
            "output": {
                0: "batch_size",
                1: "hidden_size",
            },
        },
        verbose=False,
    )

    onnx_model = onnx.load(output_model_path)

    onnx.checker.check_model(onnx_model)


def build_onnx_model_config(
    model_name: str, max_batch_size: int, sample_input: Any, test_output: Any
):

    # Make the config file
    config: typing.Any = typing.cast(typing.Any, ModelConfig())

    config.name = model_name
    config.backend = "onnxruntime"
    config.max_batch_size = max_batch_size

    # pylint: disable=no-member
    inputs = []
    for input_name, input_data in sample_input.data.items():
        inputs.append(
            ModelInput(
                name=input_name,
                data_type=DataType.Value(
                    f"TYPE_{np_to_triton_dtype(input_data.cpu().numpy().dtype)}"
                ),
                dims=[input_data.shape[1]],
            )
        )

    config.input.extend(inputs)

    config.output.extend(
        [
            ModelOutput(
                name="output",
                data_type=DataType.Value(
                    f"TYPE_{np_to_triton_dtype(test_output.cpu().numpy().dtype)}"
                ),
                dims=[test_output.shape[1]],
            )
        ]
    )

    def _powers_of_2(max_val: int):
        val = 1

        while val <= max_val:
            yield val
            val *= 2

    config.dynamic_batching.preferred_batch_size.extend(
        [x for x in _powers_of_2(max_batch_size)]
    )
    config.dynamic_batching.max_queue_delay_microseconds = 50000

    config.optimization.execution_accelerators.gpu_execution_accelerator.extend(
        [
            ModelOptimizationPolicy.ExecutionAccelerators.Accelerator(
                name="tensorrt",
                parameters={
                    "precision_mode": "FP16",
                    "max_workspace_size_bytes": "2147483648",
                },
            )
        ]
    )

    config.instance_group.extend([ModelInstanceGroup(count=1, kind="KIND_GPU")])

    return config


def build_preprocessing_config(
    model_name: str = "preprocessing",
    max_batch_size: int = 128,
    tokenizer_dir: str = "/app/local_tokenizer/bert",
):
    return ModelConfig(
        name=model_name,
        backend="python",
        max_batch_size=max_batch_size,
        input=[{"name": "QUERY", "data_type": "TYPE_STRING", "dims": [-1]}],
        output=[
            {"name": "INPUT_ID", "data_type": "TYPE_INT64", "dims": [-1]},
            {"name": "ATTENTION_MASK", "data_type": "TYPE_INT64", "dims": [-1]},
        ],
        instance_group=[{"count": 1, "kind": "KIND_CPU"}],
        parameters={"tokenizer_dir": ModelParameter(string_value=tokenizer_dir)},
    )


def build_ensemble_config(
    model_name: str = "ensemble",
    preprocessing_name: str = "preprocessing",
    embedding_name: str = "embedding",
    max_batch_size: int = 128,
    output_size: int = 384,
):
    return ModelConfig(
        name=model_name,
        backend="ensemble",
        max_batch_size=max_batch_size,
        input=[{"name": "input", "data_type": "TYPE_STRING", "dims": [-1]}],
        output=[{"name": "embedding", "data_type": "TYPE_FP32", "dims": [output_size]}],
        ensemble_scheduling={
            "step": [
                {
                    "model_name": preprocessing_name,
                    "model_version": -1,
                    "input_map": {"QUERY": "input"},
                    "output_map": {"INPUT_ID": "_INPUT_ID"},
                    "output_map": {"ATTENTION_MASK": "_ATTENTION_MASK"},
                },
                {
                    "model_name": embedding_name,
                    "model_version": -1,
                    "input_map": {"input_ids": "_INPUT_ID"},
                    "input_map": {"attention_mask": "_ATTENTION_MASK"},
                    "output_map": {"output": "embedding"},
                },
            ]
        },
    )


def main(args):
    hf_model_name = args.hf_model_name
    model_seq_length = args.model_seq_length
    max_batch_size = args.max_batch_size
    triton_repo = args.triton_repo
    embedding_model_name = args.embedding_model_name
    preprocess_model_name = args.preprocessing_model_name
    ensemble_model_name = args.ensemble_model_name
    tokenizer_dir = args.tokenizer_dir

    if embedding_model_name is None:
        embedding_model_name = hf_model_name

    embedding_model_dir = os.path.join(triton_repo, embedding_model_name)
    preprocess_model_dir = os.path.join(triton_repo, preprocess_model_name)
    embedding_model_version_dir = os.path.join(embedding_model_dir, "1")
    preprocess_model_version_dir = os.path.join(preprocess_model_dir, "1")
    # Make sure we create the directory if it does not exist
    os.makedirs(triton_repo, exist_ok=True)
    os.makedirs(embedding_model_dir, exist_ok=True)
    os.makedirs(embedding_model_version_dir, exist_ok=True)
    os.makedirs(preprocess_model_dir, exist_ok=True)
    os.makedirs(preprocess_model_version_dir, exist_ok=True)
    os.makedirs(tokenizer_dir, exist_ok=True)

    device = torch.device("cuda")
    hf_model_name = f"{hf_model_name}".strip()

    logger.info(f"Initialize model {hf_model_name}")
    model = CustomEmbeddingModel(hf_model_name)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

    test_texts = [
        "This is text one which is longer",
        "This is text two",
    ]

    sample_input = tokenizer(
        test_texts,
        max_length=model_seq_length,
        padding="max_length",
        truncation=True,
        return_token_type_ids=False,
        return_tensors="pt",
    ).to(device)

    # print(sample_input)

    test_output = model(**(sample_input.to(device))).detach()

    onnx_config = build_onnx_model_config(
        model_name=embedding_model_name,
        max_batch_size=max_batch_size,
        sample_input=sample_input,
        test_output=test_output,
    )

    config_path = os.path.join(embedding_model_dir, "config.pbtxt")

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(str(onnx_config))

    logger.info(f"Created Triton Model Configuration file at {config_path}")

    output_model_path = os.path.join(embedding_model_version_dir, "model.onnx")

    export_onnx_model(model, sample_input, output_model_path=output_model_path)
    logger.info(f"Created Triton Model at {embedding_model_dir}")

    # Preprocessing model
    triton_tokenizer_dir = tokenizer_dir.split("deploy_triton_server")[1]
    preprocess_config = build_preprocessing_config(
        model_name=preprocess_model_name,
        max_batch_size=max_batch_size,
        tokenizer_dir=triton_tokenizer_dir,
    )
    preprocess_config_path = os.path.join(preprocess_model_dir, "config.pbtxt")
    with open(preprocess_config_path, "w", encoding="utf-8") as f:
        # f.write(str(preprocess_config))
        text_format.PrintMessage(
            preprocess_config, f, use_short_repeated_primitives=True
        )

    # # Ensemble model
    # ensemble_config = build_ensemble_config(
    #     model_name=ensemble_model_name,
    #     preprocessing_name=preprocess_model_name,
    #     embedding_name=embedding_model_name,
    #     max_batch_size=max_batch_size,
    #     output_size=test_output.shape[1],
    # )
    # ensemble_config_path = os.path.join(ensemble_model_dir, "config.pbtxt")
    # with open(ensemble_config_path, "w", encoding="utf-8") as f:
    #     # f.write(str(preprocess_config))
    #     text_format.PrintMessage(ensemble_config, f, use_short_repeated_primitives=False)

    # logger.info(f"Created Triton Model Configuration file at {ensemble_config_path}")

    tokenizer.save_pretrained(tokenizer_dir)
    logger.info(f"Created tokenizer checkpoint at {tokenizer_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Export HuggingFace open-source embedding models to ONNX"
    )
    parser.add_argument(
        "--hf-model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace embedding model name",
    )
    parser.add_argument(
        "--model-seq-length",
        type=int,
        default=512,
        help="Maximum input sequence length",
    )
    parser.add_argument(
        "--max-batch-size", type=int, default=128, help="Maximum batch size"
    )
    parser.add_argument(
        "--triton-repo",
        type=str,
        default="../deploy_triton_server/app/model_repos",
        help="Path to the Triton server model repository",
    )
    parser.add_argument(
        "--embedding-model-name",
        type=str,
        default="st_embedding",
        help="Name of the exported ONNX embedding model directory inside the Triton model repo",
    )
    parser.add_argument(
        "--preprocessing-model-name",
        type=str,
        default="st_preprocessing",
        help="Name of the preprocessing model directory inside the Triton model repo",
    )
    parser.add_argument(
        "--ensemble-model-name",
        type=str,
        default="st_ensemble",
        help="Name of the ensemble model directory inside the Triton model repo",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=str,
        default="../deploy_triton_server/app/local_tokenizers/bert",
        help="Name of the tokenizer checkpoint directory",
    )

    args = parser.parse_args()

    main(args)
