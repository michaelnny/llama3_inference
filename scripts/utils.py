# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path
from typing import Optional


# Simple hack to support running without installing as a package, so we can import the tokenizer
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

# CHANGE: using native Tiktoken tokenizer
from libs.tokenizer import Tokenizer

import tensorrt_llm

# TODO(enweiz): Update for refactored models
DEFAULT_HF_MODEL_DIRS = {
    'BaichuanForCausalLM': 'baichuan-inc/Baichuan-13B-Chat',
    'BloomForCausalLM': 'bigscience/bloom-560m',
    'ChatGLMForCausalLM': 'THUDM/chatglm3-6b',
    'FalconForCausalLM': 'tiiuae/falcon-rw-1b',
    'gpt': 'gpt2-medium',
    'GPTJForCausalLM': 'EleutherAI/gpt-j-6b',
    'GPTNeoXForCausalLM': 'EleutherAI/gpt-neox-20b',
    'InternLMForCausalLM': 'internlm/internlm-chat-7b',
    'LlamaForCausalLM': 'meta-llama/Llama-2-7b-hf',
    'MPTForCausalLM': 'mosaicml/mpt-7b',
    'PhiForCausalLM': 'microsoft/phi-2',
    'OPTForCausalLM': 'facebook/opt-350m',
    'qwen': 'Qwen/Qwen-7B',
}

DEFAULT_PROMPT_TEMPLATES = {
    'InternLMForCausalLM':
    "<|User|>:{input_text}<eoh>\n<|Bot|>:",
    'qwen':
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n",
    # CHANGE: add a prompt template for llama3
    'Llama3Chat':
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
}


def read_model_name(engine_dir: str):
    engine_version = tensorrt_llm.runtime.engine.get_engine_version(engine_dir)

    with open(Path(engine_dir) / "config.json", 'r') as f:
        config = json.load(f)

    if engine_version is None:
        return config['builder_config']['name'], None

    model_arch = config['pretrained_config']['architecture']
    model_version = None
    if model_arch == 'ChatGLMForCausalLM':
        model_version = config['pretrained_config']['chatglm_version']
    return model_arch, model_version


def throttle_generator(generator, stream_interval):
    for i, out in enumerate(generator):
        if not i % stream_interval:
            yield out

    if i % stream_interval:
        yield out


def load_tokenizer(tokenizer_ckpt: Optional[str] = None) -> Tokenizer:

    # CHANGE: using native Tiktoken tokenizer
    tokenizer = Tokenizer(model_path=tokenizer_ckpt)

    pad_id = tokenizer.pad_id
    end_id = tokenizer.eos_id

    return tokenizer, pad_id, end_id
