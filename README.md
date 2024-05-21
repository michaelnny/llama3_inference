# 1. Preparation

## 1.1 Download Llama3 model and tokenizer checkpoints

Please refer to Meta's llama3 repository on how to download the native model and tokenizer checkpoints (not based on HuggingFace).


## 1.2 Install Docker on Host OS

https://en.opensuse.org/Docker

```bash
# Install Docker pages:
zypper install docker docker-compose docker-compose-switch

# To start the docker daemon during boot:
sudo systemctl enable docker

# To join the docker group that is allowed to use the docker daemon:
sudo usermod -G docker -a $USER

# Log in to the docker group:
newgrp docker

# Restart the docker daemon:
sudo systemctl restart docker

# Verify docker is running:
docker version
```

Install Nvidia container runtime
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-zypper

```bash
sudo zypper ar https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo

sudo zypper modifyrepo --enable nvidia-container-toolkit-experimental

sudo zypper --gpg-auto-import-keys install nvidia-container-toolkit
```

Configure Docker runtime

```bash
sudo nvidia-ctk runtime configure --runtime=docker


sudo systemctl restart docker
```

# 2. Build a TensorRT engine for Llama3 model

```
git clone xxx

cd llama3_inference
```

The files inside the `scripts` contains all the code we need to build the TensorRT engine, the code was adapted from the original TensorRT-LLM project, under the `examples/llama`.

You can install the TensorRT-LLM on your host machine, or using a Docker image. In practice, using a Docker to do this is much easier.
Pull and build Docker image, this will take a few minutes.

```bash
cd llama3_inference

docker run --rm --gpus all --volume ${PWD}:/workspace --entrypoint /bin/bash -it --workdir /workspace nvidia/cuda:12.1.0-devel-ubuntu22.04



Unable to find image 'nvidia/cuda:12.1.0-devel-ubuntu22.04' locally
12.1.0-devel-ubuntu22.04: Pulling from nvidia/cuda
aece8493d397: Pull complete
45f7ea5367fe: Pull complete
3d97a47c3c73: Pull complete
12cd4d19752f: Pull complete
da5a484f9d74: Pull complete
5e5846364eee: Pull complete
fd355de1d1f2: Pull complete
3480bb79c638: Pull complete
e7016935dd60: Pull complete
99541166a133: Pull complete
8999112df5b0: Pull complete
Digest: sha256:e3a8f7b933e77ecee74731198a2a5483e965b585cea2660675cf4bb152237e9b
Status: Downloaded newer image for nvidia/cuda:12.1.0-devel-ubuntu22.04

```


Install packages inside the docker run-time image, these are required to build the TensorRT engine

```bash
# Install dependencies, TensorRT-LLM requires Python 3.10
apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev

# Install the stable version (corresponding to the cloned branch) of TensorRT-LLM.
pip3 install tensorrt_llm==0.8.0 -U --extra-index-url https://pypi.nvidia.com

# Optional, for testing runs
pip3 install tiktoken blobfile
```

Tips: we can save the update docker image to a custom namespace, so we don't have to repeadtly install the packages if something goes wrong and we want to start over.

```bash
# find the docker instance id
docker ps -a -q

# create a new image and save the changes to 
docker commit c0241ede9323 my-tensorrtllm-image:latest

```

We can then load the custom image using this command:
```bash

docker run --rm --gpus all --volume ${PWD}:/workspace --entrypoint /bin/bash -it --workdir /workspace my-tensorrtllm-image:latest

```


To build the TensorRT engine, we first need to convert the regular llama model checkpoint to a TensorRT-LLM compatible format.
The `scripts/convert_checkpoint.py` file can be used to do this, and it was adapted from the original TensorRT-LLM repository under the `examples/llama/` folder.


The following command will load the original llama model checkpoint from `./checkpoints/Meta-Llama-3-8B-Instruct`, convert to `bfloat16`, and save the files to `./checkpoints/trtllm-Llama-3-8B-Instruct-1gpu-bf16` so we can later us it to build the TensorRT engine.

```bash
python3 scripts/convert_checkpoint.py --meta_ckpt_dir ./checkpoints/Meta-Llama-3-8B-Instruct \
            --output_dir ./tmp/trtllm-Llama-3-8B-Instruct-1gpu-bf16 \
            --dtype bfloat16 \
            --vocab_size 128256 \
            --n_kv_head 8 



[TensorRT-LLM] TensorRT-LLM version: 0.8.00.8.0
Total time of converting checkpoints: 00:00:24
```



Inside the TensorRT-LLM container, build the TensorRT engine using the checkpoint from the above step, note here we need to remove `--gemm_plugin` in order to use in-flight batching. 
```bash
trtllm-build --checkpoint_dir ./tmp/trtllm-Llama-3-8B-Instruct-1gpu-bf16 \
            --output_dir ./tmp/trt_engines/llama3/8B/bf16/1-gpu \
            --paged_kv_cache enable \
            --context_fmha enable \
            --remove_input_padding enable \
            --gpt_attention_plugin bfloat16 \
            --max_batch_size 64 \
            --max_input_len 1024 \
            --max_output_len 1024 \
            --max_num_tokens 12288 \
            --strongly_typed



[TensorRT-LLM] TensorRT-LLM version: 0.8.0[05/21/2024-14:00:33] [TRT-LLM] [I] Set bert_attention_plugin to float16.
[05/21/2024-14:00:33] [TRT-LLM] [I] Set gpt_attention_plugin to bfloat16.
[05/21/2024-14:00:33] [TRT-LLM] [I] Set gemm_plugin to None.
[05/21/2024-14:00:33] [TRT-LLM] [I] Set lookup_plugin to None.
[05/21/2024-14:00:33] [TRT-LLM] [I] Set lora_plugin to None.
[05/21/2024-14:00:33] [TRT-LLM] [I] Set context_fmha to True.
[05/21/2024-14:00:33] [TRT-LLM] [I] Set context_fmha_fp32_acc to False.
[05/21/2024-14:00:33] [TRT-LLM] [I] Set paged_kv_cache to True.
[05/21/2024-14:00:33] [TRT-LLM] [I] Set remove_input_padding to True.
[05/21/2024-14:00:33] [TRT-LLM] [I] Set use_custom_all_reduce to True.
[05/21/2024-14:00:33] [TRT-LLM] [I] Set multi_block_mode to False.
[05/21/2024-14:00:33] [TRT-LLM] [I] Set enable_xqa to True.
[05/21/2024-14:00:33] [TRT-LLM] [I] Set attention_qk_half_accumulation to False.
[05/21/2024-14:00:33] [TRT-LLM] [I] Set tokens_per_block to 128.
[05/21/2024-14:00:33] [TRT-LLM] [I] Set use_paged_context_fmha to False.
[05/21/2024-14:00:33] [TRT-LLM] [I] Set use_context_fmha_for_generation to False.
[05/21/2024-14:00:33] [TRT] [I] [MemUsageChange] Init CUDA: CPU +14, GPU +0, now: CPU 155, GPU 269 (MiB)
[05/21/2024-14:00:34] [TRT] [I] [MemUsageChange] Init builder kernel library: CPU +1810, GPU +312, now: CPU 2101, GPU 581 (MiB)
[05/21/2024-14:00:34] [TRT-LLM] [I] Set nccl_plugin to None.
[05/21/2024-14:00:34] [TRT-LLM] [I] Set use_custom_all_reduce to True.
[05/21/2024-14:00:34] [TRT-LLM] [I] Build TensorRT engine Unnamed Network 0
[05/21/2024-14:00:34] [TRT] [W] Unused Input: position_ids
[05/21/2024-14:00:34] [TRT] [W] [RemoveDeadLayers] Input Tensor position_ids is unused or used only at compile-time, but is not being removed.
[05/21/2024-14:00:34] [TRT] [I] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +8, now: CPU 2127, GPU 603 (MiB)
[05/21/2024-14:00:34] [TRT] [I] [MemUsageChange] Init cuDNN: CPU +1, GPU +10, now: CPU 2128, GPU 613 (MiB)
[05/21/2024-14:00:34] [TRT] [W] TensorRT was linked against cuDNN 8.9.6 but loaded cuDNN 8.9.2
[05/21/2024-14:00:34] [TRT] [I] Global timing cache in use. Profiling results in this builder pass will be stored.
[05/21/2024-14:02:21] [TRT] [I] [GraphReduction] The approximate region cut reduction algorithm is called.
[05/21/2024-14:02:21] [TRT] [I] Detected 106 inputs and 1 output network tensors.
[05/21/2024-14:02:21] [TRT] [I] Total Host Persistent Memory: 22560
[05/21/2024-14:02:21] [TRT] [I] Total Device Persistent Memory: 0
[05/21/2024-14:02:21] [TRT] [I] Total Scratch Memory: 1358954496
[05/21/2024-14:02:21] [TRT] [I] [BlockAssignment] Started assigning block shifts. This will take 168 steps to complete.
[05/21/2024-14:02:21] [TRT] [I] [BlockAssignment] Algorithm ShiftNTopDown took 2.32292ms to assign 12 blocks to 168 nodes requiring 1811942912 bytes.
[05/21/2024-14:02:21] [TRT] [I] Total Activation Memory: 1811942912
[05/21/2024-14:02:26] [TRT] [I] Total Weights Memory: 16060522752
[05/21/2024-14:02:26] [TRT] [I] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +8, now: CPU 2189, GPU 15943 (MiB)
[05/21/2024-14:02:26] [TRT] [I] [MemUsageChange] Init cuDNN: CPU +0, GPU +10, now: CPU 2189, GPU 15953 (MiB)
[05/21/2024-14:02:26] [TRT] [W] TensorRT was linked against cuDNN 8.9.6 but loaded cuDNN 8.9.2
[05/21/2024-14:02:26] [TRT] [I] Engine generation completed in 111.412 seconds.
[05/21/2024-14:02:26] [TRT] [I] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 0 MiB, GPU 15317 MiB
[05/21/2024-14:02:26] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in building engine: CPU +0, GPU +15317, now: CPU 0, GPU 15317 (MiB)
[05/21/2024-14:02:32] [TRT] [I] [MemUsageStats] Peak memory usage during Engine building and serialization: CPU: 38513 MiB
[05/21/2024-14:02:32] [TRT-LLM] [I] Total time of building Unnamed Network 0: 00:01:57
[05/21/2024-14:02:33] [TRT-LLM] [I] Serializing engine to ./tmp/trt_engines/llama3/8B/bf16/1-gpu/rank0.engine...
[05/21/2024-14:02:42] [TRT-LLM] [I] Engine serialized. Total time: 00:00:09
[05/21/2024-14:02:42] [TRT-LLM] [I] Total time of building all engines: 00:02:09

```


Test the generated TensorRT engine



The code from the original TensorRT-LLM only provide code to test the engine when using HuggingFace transformers (models and tokenizers). We have made some simple changes to the code to load the native Tiktoken tokenizer.

Inside the TensorRT-LLM container, run the following command:

```bash

python3 scripts/test_run.py --engine_dir ./tmp/trt_engines/llama3/8B/bf16/1-gpu --max_output_len 100 --tokenizer_dir ./checkpoints/Meta-Llama-3-8B-Instruct --input_text "How do I count to nine in French?"



[TensorRT-LLM][INFO] Engine version 0.8.0 found in the config file, assuming engine(s) built by new builder API.
[TensorRT-LLM][WARNING] [json.exception.type_error.302] type must be array, but is null
[TensorRT-LLM][WARNING] Optional value for parameter lora_target_modules will not be set.
[TensorRT-LLM][WARNING] Parameter max_draft_len cannot be read from json:
[TensorRT-LLM][WARNING] [json.exception.out_of_range.403] key 'max_draft_len' not found
[TensorRT-LLM][WARNING] [json.exception.type_error.302] type must be string, but is null
[TensorRT-LLM][WARNING] Optional value for parameter quant_algo will not be set.
[TensorRT-LLM][WARNING] [json.exception.type_error.302] type must be string, but is null
[TensorRT-LLM][WARNING] Optional value for parameter kv_cache_quant_algo will not be set.
[TensorRT-LLM][INFO] MPI size: 1, rank: 0
[TensorRT-LLM][INFO] Loaded engine size: 15320 MiB
[TensorRT-LLM][INFO] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +8, now: CPU 15519, GPU 15607 (MiB)
[TensorRT-LLM][INFO] [MemUsageChange] Init cuDNN: CPU +1, GPU +10, now: CPU 15520, GPU 15617 (MiB)
[TensorRT-LLM][WARNING] TensorRT was linked against cuDNN 8.9.6 but loaded cuDNN 8.9.2
[TensorRT-LLM][INFO] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +15316, now: CPU 0, GPU 15316 (MiB)
[TensorRT-LLM][INFO] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +8, now: CPU 15543, GPU 15835 (MiB)
[TensorRT-LLM][INFO] [MemUsageChange] Init cuDNN: CPU +0, GPU +8, now: CPU 15543, GPU 15843 (MiB)
[TensorRT-LLM][WARNING] TensorRT was linked against cuDNN 8.9.6 but loaded cuDNN 8.9.2
[TensorRT-LLM][INFO] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +0, now: CPU 0, GPU 15316 (MiB)
[TensorRT-LLM][INFO] Allocate 7952400384 bytes for k/v cache. 
[TensorRT-LLM][INFO] Using 60672 tokens in paged KV cache.
[TensorRT-LLM] TensorRT-LLM version: 0.8.0Input [Text 0]: "<|begin_of_text|><|begin_of_text|><|start_header_id|>user<|end_header_id|>

How do I count to nine in French?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
Output [Text 0 Beam 0]: " 

To count to nine in French, you can use the following numbers:

1. Un (one)
2. Deux (two)
3. Trois (three)
4. Quatre (four)
5. Cinq (five)
6. Six (six)
7. Sept (seven)
8. Huit (eight)
9. Neuf (nine)

I hope that helps! Let me know if you have any other questions.<|eot_id|><|start_header_id|>assistant

You're welcome! I hope"

```





# 3. Deploying with Triton Inference Server

Create a model registry for TensorRT-LLM backend

Back to the host os, clone a the TensorRT-LLM backend repository.

```bash
cd /tmp

git clone https://github.com/triton-inference-server/tensorrtllm_backend.git

```

Go to the host os workspace, at the root of the project folder, create the file-system structure and copy the TensorRT engine to the Triton inference server

```bash
cd ~/llama3_inference

# Create a workspace for triton inference server
mkdir -p triton_server/model_repos
mkdir -p triton_server/scripts
mkdir -p triton_server/tools


# Copy the TRT engine to the model repository
cp ./tmp/trt_engines/llama3/8B/bf16/1-gpu/* triton_server/model_repos/tensorrt_llm/1/


# Copy the example models to the model repository
cp -r /tmp/tensorrtllm_backend/all_models/inflight_batcher_llm/* triton_server/model_repos/



# Copy triton server and other scripts
cp /tmp/tensorrtllm_backend/scripts/launch_triton_server.py triton_server/scripts/launch_triton_server.py 

cp -r /tmp/tensorrtllm_backend/tools/* triton_server/tools/



# Create folder for tiktoken tokenizer, so we only need to mount `triton_server` to the container
mkdir -p triton_server/llama3

# Copy tiktoken tokenizer class and checkpoint to Triton inference workspace
cp libs/tokenizer.py triton_server/llama3
cp checkpoints/Meta-Llama-3-8B-Instruct/tokenizer.model triton_server/llama3


```

Now we have everything we need to run the Triton inference server, before we start the server, we also need to manually change the configuration files for the TensorRT-LLM models.

Update config.pgtxt file for the different models
```bash
cd triton_server

# Note here the /workspace is the path inside the Docker container, as we'll be mount 'triton_server' from our host server to the container
# The location of the tiktoken tokenizer
LLAMA_CKPT_DIR=/workspace/llama3
# The location of the compiled TensorRT engine
ENGINE_PATH=/workspace/model_repos/tensorrt_llm/1


python3 tools/fill_template.py -i model_repos/preprocessing/config.pbtxt \
        tokenizer_dir:${LLAMA_CKPT_DIR},tokenizer_type:auto,triton_max_batch_size:64,preprocessing_instance_count:1


python3 tools/fill_template.py -i model_repos/postprocessing/config.pbtxt \
        tokenizer_dir:${LLAMA_CKPT_DIR},tokenizer_type:auto,triton_max_batch_size:64,postprocessing_instance_count:1


python3 tools/fill_template.py -i model_repos/tensorrt_llm_bls/config.pbtxt \
        triton_max_batch_size:64,decoupled_mode:False,bls_instance_count:1,accumulate_tokens:False


python3 tools/fill_template.py -i model_repos/ensemble/config.pbtxt triton_max_batch_size:64


python3 tools/fill_template.py -i model_repos/tensorrt_llm/config.pbtxt \
        triton_backend:tensorrtllm,triton_max_batch_size:64,decoupled_mode:False,max_beam_width:1,engine_dir:${ENGINE_PATH},max_tokens_in_paged_kv_cache:2560,max_attention_window_size:2560,kv_cache_free_gpu_mem_fraction:0.25,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:50000

```

Next, we need to update the code in `preprocessing` and `postprocessing` models, essentially, we need to replace the tokenizer with the native tiktoken tokenizer, also change how to text was encoded/decoded.


```bash
cd triton_server

# Set the workspace as the same folder we are in
docker run -it --rm --gpus all --network host --shm-size=2g \
--volume $(pwd):/workspace \
--workdir /workspace \
nvcr.io/nvidia/tritonserver:24.03-trtllm-python-py3


Unable to find image 'nvcr.io/nvidia/tritonserver:24.03-trtllm-python-py3' locally
24.03-trtllm-python-py3: Pulling from nvidia/tritonserver
a48641193673: Pulling fs layer 
4505fcb34058: Pulling fs layer 
4f4fb700ef54: Pulling fs layer 
61c70c86753a: Waiting 
2e71696a7bbd: Pull complete 
2f61de369480: Pull complete 
232727b13aa7: Pull complete 
7a0e546cb493: Pull complete 
b4e102e4d681: Pull complete 
5f8055ebdf56: Pull complete 
93a7bdb52326: Pull complete 
c62aa9613aa2: Pull complete 
2d6fed706135: Pull complete 
573e602e6564: Pull complete 
379fe37161b3: Pull complete 
26aec69ed155: Pull complete 
4ce82283ffe6: Pull complete 
11d2607b81e7: Pull complete 
1d7d9ff4a1af: Pull complete 
0bf978d1d9c7: Pull complete 
d98db3a93c00: Pull complete 
c1c8a4945a10: Pull complete 
bc70024cf70f: Pull complete 
fe8724bdd50d: Pull complete 
4e6c78908a34: Pull complete 
dfa26552209a: Pull complete 
61e2d3d2f6f3: Pull complete 
8b07fa9c4696: Pull complete 
c4f784b11868: Pull complete 
b78b51da6bc5: Pull complete 
d0231c32cb70: Pull complete 
4cb8472819ff: Pull complete 
ef20ad9cf5d9: Pull complete 
Digest: sha256:fb1a2a07b9796246f3a24c85e1d89aa3f39b15038379cd212a514397000aeddb
Status: Downloaded newer image for nvcr.io/nvidia/tritonserver:24.03-trtllm-python-py3

=============================
== Triton Inference Server ==
=============================

NVIDIA Release 24.03 (build 86102893)
Triton Server Version 2.44.0

Copyright (c) 2018-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

Various files include modifications (c) NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license
```


Install required package in the Triton docker instance
```bash

# Install tiktoken and other packages
pip3 install protobuf tiktoken blobfile

```


Tips: we can save the update docker image to a custom namespace, so we don't have to repeadtly install the packages if something goes wrong and we want to start over.

```bash
# find the docker instance id
docker ps -a -q

# create a new image and save the changes to 
docker commit 70f030a33c38 my-triton-server-image:latest

```

We can then reuse the image for subsequent runs
```bash
cd triton_server

docker run -it --rm --gpus all --network host --shm-size=2g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --volume $(pwd):/workspace \
  --workdir /workspace \
  my-triton-server-image:latest

```


Now start the Triton inference server inside the docker container
```bash
# Launch Server
python3 scripts/launch_triton_server.py \
    --model_repo /workspace/model_repos \
    --world_size 1 \
    --log
    

root@linux-pc:/workspace# I0521 10:03:57.685707 107 pinned_memory_manager.cc:275] Pinned memory pool is created at '0x7fc5a4000000' with size 268435456
I0521 10:03:57.685907 107 cuda_memory_manager.cc:107] CUDA memory pool is created on device 0 with size 67108864
I0521 10:03:57.688373 107 model_lifecycle.cc:469] loading: tensorrt_llm_bls:1
I0521 10:03:57.688401 107 model_lifecycle.cc:469] loading: tensorrt_llm:1
I0521 10:03:57.688418 107 model_lifecycle.cc:469] loading: preprocessing:1
I0521 10:03:57.688436 107 model_lifecycle.cc:469] loading: postprocessing:1
I0521 10:03:57.726402 107 python_be.cc:2391] TRITONBACKEND_ModelInstanceInitialize: tensorrt_llm_bls_0_0 (CPU device 0)
I0521 10:03:57.835575 107 python_be.cc:2391] TRITONBACKEND_ModelInstanceInitialize: preprocessing_0_0 (CPU device 0)
[TensorRT-LLM][WARNING] gpu_device_ids is not specified, will be automatically set
I0521 10:03:57.836328 107 python_be.cc:2391] TRITONBACKEND_ModelInstanceInitialize: postprocessing_0_0 (CPU device 0)
[TensorRT-LLM][WARNING] batch_scheduler_policy parameter was not found or is invalid (must be max_utilization or guaranteed_no_evict)
[TensorRT-LLM][WARNING] enable_chunked_context is not specified, will be set to false.
[TensorRT-LLM][WARNING] enable_trt_overlap is not specified, will be set to false
[TensorRT-LLM][WARNING] normalize_log_probs is not specified, will be set to true
[TensorRT-LLM][INFO] Engine version 0.8.0 found in the config file, assuming engine(s) built by new builder API.
[TensorRT-LLM][WARNING] [json.exception.type_error.302] type must be array, but is null
[TensorRT-LLM][WARNING] Optional value for parameter lora_target_modules will not be set.
[TensorRT-LLM][WARNING] Parameter max_draft_len cannot be read from json:
[TensorRT-LLM][WARNING] [json.exception.out_of_range.403] key 'max_draft_len' not found
[TensorRT-LLM][WARNING] [json.exception.type_error.302] type must be string, but is null
[TensorRT-LLM][WARNING] Optional value for parameter quant_algo will not be set.
[TensorRT-LLM][WARNING] [json.exception.type_error.302] type must be string, but is null
[TensorRT-LLM][WARNING] Optional value for parameter kv_cache_quant_algo will not be set.
[TensorRT-LLM][INFO] Initializing MPI with thread mode 1
I0521 10:03:57.911020 107 model_lifecycle.cc:835] successfully loaded 'tensorrt_llm_bls'
[TensorRT-LLM][INFO] MPI size: 1, rank: 0
[TensorRT-LLM][INFO] Rank 0 is using GPU 0
I0521 10:03:58.257018 107 model_lifecycle.cc:835] successfully loaded 'preprocessing'
I0521 10:03:58.263239 107 model_lifecycle.cc:835] successfully loaded 'postprocessing'
[TensorRT-LLM][INFO] TRTGptModel maxNumSequences: 32
[TensorRT-LLM][INFO] TRTGptModel maxBatchSize: 32
[TensorRT-LLM][INFO] TRTGptModel mMaxAttentionWindowSize: 2560
[TensorRT-LLM][INFO] TRTGptModel enableTrtOverlap: 0
[TensorRT-LLM][INFO] TRTGptModel normalizeLogProbs: 1
[TensorRT-LLM][INFO] Loaded engine size: 15320 MiB
[TensorRT-LLM][INFO] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +8, now: CPU 15366, GPU 15671 (MiB)
[TensorRT-LLM][INFO] [MemUsageChange] Init cuDNN: CPU +2, GPU +10, now: CPU 15368, GPU 15681 (MiB)
[TensorRT-LLM][INFO] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +15316, now: CPU 0, GPU 15316 (MiB)
[TensorRT-LLM][INFO] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +1, GPU +8, now: CPU 15391, GPU 21627 (MiB)
[TensorRT-LLM][INFO] [MemUsageChange] Init cuDNN: CPU +0, GPU +8, now: CPU 15391, GPU 21635 (MiB)
[TensorRT-LLM][INFO] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +0, now: CPU 0, GPU 15316 (MiB)
[TensorRT-LLM][WARNING] Both freeGpuMemoryFraction (aka kv_cache_free_gpu_mem_fraction) and maxTokens (aka max_tokens_in_paged_kv_cache) are set (to 0.500000 and 2560, respectively). The smaller value will be used.
[TensorRT-LLM][INFO] Allocate 335544320 bytes for k/v cache. 
[TensorRT-LLM][INFO] Using 2560 total tokens in paged KV cache, and 20 blocks per sequence
I0521 10:04:12.749494 107 model_lifecycle.cc:835] successfully loaded 'tensorrt_llm'
I0521 10:04:12.749694 107 server.cc:607] 
+------------------+------+
| Repository Agent | Path |
+------------------+------+
+------------------+------+

I0521 10:04:12.749735 107 server.cc:634] 
+-------------+----------------------------------------------+----------------------------------------------+
| Backend     | Path                                         | Config                                       |
+-------------+----------------------------------------------+----------------------------------------------+
| python      | /opt/tritonserver/backends/python/libtriton_ | {"cmdline":{"auto-complete-config":"false"," |
|             | python.so                                    | backend-directory":"/opt/tritonserver/backen |
|             |                                              | ds","min-compute-capability":"6.000000","shm |
|             |                                              | -region-prefix-name":"prefix0_","default-max |
|             |                                              | -batch-size":"4"}}                           |
|             |                                              |                                              |
| tensorrtllm | /opt/tritonserver/backends/tensorrtllm/libtr | {"cmdline":{"auto-complete-config":"false"," |
|             | iton_tensorrtllm.so                          | backend-directory":"/opt/tritonserver/backen |
|             |                                              | ds","min-compute-capability":"6.000000","def |
|             |                                              | ault-max-batch-size":"4"}}                   |
|             |                                              |                                              |
+-------------+----------------------------------------------+----------------------------------------------+

I0521 10:04:12.750993 107 server.cc:677] 
+------------------+---------+--------+
| Model            | Version | Status |
+------------------+---------+--------+
| postprocessing   | 1       | READY  |
| preprocessing    | 1       | READY  |
| tensorrt_llm     | 1       | READY  |
| tensorrt_llm_bls | 1       | READY  |
+------------------+---------+--------+

I0521 10:04:12.765492 107 metrics.cc:877] Collecting metrics for GPU 0: NVIDIA GeForce RTX 3090
I0521 10:04:12.767069 107 metrics.cc:770] Collecting CPU metrics
I0521 10:04:12.767162 107 tritonserver.cc:2538] 
+----------------------------------+------------------------------------------------------------------------+
| Option                           | Value                                                                  |
+----------------------------------+------------------------------------------------------------------------+
| server_id                        | triton                                                                 |
| server_version                   | 2.44.0                                                                 |
| server_extensions                | classification sequence model_repository model_repository(unload_depen |
|                                  | dents) schedule_policy model_configuration system_shared_memory cuda_s |
|                                  | hared_memory binary_tensor_data parameters statistics trace logging    |
| model_repository_path[0]         | /workspace/model_repos                                                 |
| model_control_mode               | MODE_NONE                                                              |
| strict_model_config              | 1                                                                      |
| rate_limit                       | OFF                                                                    |
| pinned_memory_pool_byte_size     | 268435456                                                              |
| cuda_memory_pool_byte_size{0}    | 67108864                                                               |
| min_supported_compute_capability | 6.0                                                                    |
| strict_readiness                 | 1                                                                      |
| exit_timeout                     | 30                                                                     |
| cache_enabled                    | 0                                                                      |
+----------------------------------+------------------------------------------------------------------------+

I0521 10:04:12.771724 107 grpc_server.cc:2466] Started GRPCInferenceService at 0.0.0.0:8001
I0521 10:04:12.771885 107 http_server.cc:4636] Started HTTPService at 0.0.0.0:8000
I0521 10:04:12.812951 107 http_server.cc:320] Started Metrics Service at 0.0.0.0:8002

```

Test runs
```bash

curl -X POST localhost:8000/v2/models/ensemble/generate -d \
'{
    "text_input": "Tell me a short joke about llamas",
    "parameters": {
      "max_tokens": 128,
      "stop_words":["<|eot_id|>"]
    }
}'


# Output is not so great because we didn't apply chat template
{
    "context_logits": 0.0,
    "cum_log_probs": 0.0,
    "generation_logits": 0.0,
    "model_name": "ensemble",
    "model_version": "1",
    "output_log_probs": [
        0.0,
        0.0,
        0.0,
        0.0,
        ...
    ],
    "sequence_end": false,
    "sequence_id": 0,
    "sequence_start": false,
    "text_output": ".\nWhy did the llama refuse to play poker?\nBecause it always got fleeced! (get fleeced means to be cheated or swindled) \nTell me a short joke about llamas.\nWhy did the llama refuse to play poker?\nBecause it always got fleeced! (get fleeced means to be cheated or swindled) \nTell me a short joke about llamas.\nWhy did the llama refuse to play poker?\nBecause it always got fleeced! (get fleeced means to be cheated or swindled) \nTell me a short joke about llamas.\nWhy did the llama refuse to play poker?\nBecause"
}
```

```bash

curl -X POST localhost:8000/v2/models/ensemble/generate -d \
'{
    "text_input": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nTell me a short joke about llamas<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "parameters": {
      "max_tokens": 128,
      "stop_words":["<|eot_id|>"]
    }
}'


# Output is much better if we apply the chat template
{
    "context_logits": 0.0,
    "cum_log_probs": 0.0,
    "generation_logits": 0.0,
    "model_name": "ensemble",
    "model_version": "1",
    "output_log_probs": [
        0.0,
        0.0,
        0.0,
        0.0,
        ...
    ],
    "sequence_end": false,
    "sequence_id": 0,
    "sequence_start": false,
    "text_output": "Why did the llama go to the party?\n\nBecause it was a hair-raising good time!<|eot_id|>"
}
```


Streaming API


To use the streaming API, we need to set the following properties inside the `tensorrt_llm\config.pbtxt` and `tensorrt_llm_bls\config.pbtxt`.
```
model_transaction_policy {
  decoupled: True
}
```


We can then run the test using this command:
```bash


curl -X POST localhost:8000/v2/models/ensemble/generate_stream -d \
'{
    "text_input": "How do I count to nine in French?",
    "parameters": {
      "max_tokens": 128,
      "stream": true,
      "stop_words":["<|eot_id|>"]
    }
}'


data: {"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":0.0,"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":"**\n"}

data: {"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":"To"}

data: {"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":" count"}

data: {"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":" to"}

data: {"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":" nine"}

data: {"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":" in"}

data: {"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":" French"}

data: {"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":","}

data: {"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":" you"}

data: {"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":" can"}

data: {"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":" use"}

...

data: {"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":" if"}

data: {"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":" you"}

data: {"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":" have"}

data: {"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":" any"}

data: {"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":" other"}

data: {"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":" questions"}

data: {"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":"."}

data: {"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":"<|eot_id|>"}

```



Concurrency test

The power of TensorRT-LLM backend is the support of in-flight batching, where the response will be send to the client as soon as finished, without need to wait the full batch to be finished.

We can use this simple script to run concurrency test:
```bash

cd llama3_inference

python3 scripts/test_inference.py --num_concurrency 256 --stream

```

# Troubleshooting

- docker: Error response from daemon: unknown or invalid runtime name: nvidia.
  remove `--runtime=nvidia`
- docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].
  Install nvidia container toolkit https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-zypper
