# Deploy a TensorRT-LLM model engine to Triton inference server


**Prerequisites**

You need to make sure your host machine has Docker runtime and NVIDIA Container Toolkit installed.
- Install Docker on OpenSuse: https://en.opensuse.org/Docker
- Install NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html

You need to have a working TensorRT-LLM model engine.




## Prepare TensorRT-LLM-Backend for Triton Inference Server

On the host machine, clone the TensorRT-LLM backend repository to a temporary location, make sure we're on the same release as the one we use to build TensorRT-LLM engine.


```bash

cd ~/dl_projects

git clone -b v0.9.0 https://github.com/triton-inference-server/tensorrtllm_backend.git

```

Next, we need to prepare the Triton server workspace, specially we need to define model repository and custom code for load the native Tiktoken tokenizer.

On your host machine, navigate to `deploy_llm` folder.

```bash

cd deploy_llm

mkdir triton_server

cd triton_server

# Llama3 inflight batching model repository
mkdir -p model_repos/llama3_ifb

# Store tiktoken tokenizer class and checkpoint
mkdir -p llama3_libs


# Copy example model folder from tensorrtllm_backend to our repository
cp -r ~/dl_projects/tensorrtllm_backend/all_models/inflight_batcher_llm/* model_repos/llama3_ifb/


# Remove any existing file for `tensorrt_llm` model
rm model_repos/llama3_ifb/tensorrt_llm/1/*


# Copy the TensorRT-LLM engine to the model repository
cp ../../build_llm/tmp/trt_engines/Llama-3-8B-Instruct-1gpu-bf16/* model_repos/llama3_ifb/tensorrt_llm/1/


# Copy the preprocessing and postprocessing model.py files to the model repository, these include modified code to use the native tiktoken
cp ../src/trt_preprocessing_model.py model_repos/llama3_ifb/preprocessing/1/model.py
cp ../src/trt_postprocessing_model.py model_repos/llama3_ifb/postprocessing/1/model.py


# Finally copy the tiktoken class and checkpoint file to the triton server
cp ../src/tokenizer.py llama3_libs
cp ~/models/Llama3/Meta-Llama-3-8B-Instruct/tokenizer.model llama3_libs


# Copy start Triton server script
cp ~/dl_projects/tensorrtllm_backend/scripts/launch_triton_server.py ./start.py

```


Now we need to update the `config.pbtxt` files for the different models in the `model_repos/llama3_ifb` folder. Specifically, we need to set the location of the tokenizer checkpoint, and location of the TensorRT-LLM model engine, amount other things like maximum batch size etc.

On the host machine, use the following command to update these properties. Note the script `src/fill_template.py` was copied from the `tensorrtllm_backend` project, which was located at `tensorrtllm_backend/tools/fill_template.py`.

```bash

# Note here the /workspace is the path of `triton_server` inside the Docker container
# The location of the tiktoken tokenizer
LLAMA_CKPT_DIR=/workspace/llama3_libs

# The location of the compiled TensorRT engine
ENGINE_PATH=/workspace/model_repos/llama3_ifb/tensorrt_llm/1


cd deploy_llm


# Update preprocessing model config
python3 src/fill_template.py -i triton_server/model_repos/llama3_ifb/preprocessing/config.pbtxt \
        tokenizer_dir:${LLAMA_CKPT_DIR},tokenizer_type:auto,triton_max_batch_size:64,preprocessing_instance_count:1


# Update postprocessing model config
python3 src/fill_template.py -i triton_server/model_repos/llama3_ifb/postprocessing/config.pbtxt \
        tokenizer_dir:${LLAMA_CKPT_DIR},tokenizer_type:auto,triton_max_batch_size:64,postprocessing_instance_count:1


# Update BLS model config
python3 src/fill_template.py -i triton_server/model_repos/llama3_ifb/tensorrt_llm_bls/config.pbtxt \
        triton_max_batch_size:64,decoupled_mode:true,bls_instance_count:1,accumulate_tokens:false


# Update ensemble model config
python3 src/fill_template.py -i triton_server/model_repos/llama3_ifb/ensemble/config.pbtxt triton_max_batch_size:64


# Update TensorRT-LLM model config
python3 src/fill_template.py -i triton_server/model_repos/llama3_ifb/tensorrt_llm/config.pbtxt \
        triton_backend:tensorrtllm,triton_max_batch_size:64,decoupled_mode:true,max_beam_width:1,engine_dir:${ENGINE_PATH},max_tokens_in_paged_kv_cache:2560,max_attention_window_size:2560,kv_cache_free_gpu_mem_fraction:0.3,exclude_input_in_output:true,enable_kv_cache_reuse:false,batching_strategy:inflight_fused_batching,batch_scheduler_policy:guaranteed_completion,max_queue_delay_microseconds:5000

```



## Launch Triton inference server container

Now, we can try to pull the Triton docker image and launch our inference server


```bash

cd deploy_llm/triton_server

# Build the Docker image with the custom name
docker-compose build

# Disable cache when force rebuild
docker-compose build --no-cache

```


Now try to exist the Triton docker container and relaunch:
```bash

cd deploy_llm/triton_server


docker-compose up


# Or run in detached mode
docker-compose up -d


# Incase want to open interactive command, here 'triton-server' is the service name inside the 'docker-compose.yaml' file
docker-compose run --rm triton-server

```


Once the container is up and running, it will automatically launch the Triton inference server, which can be manually started with

```bash

python3 start.py \
    --model_repo /workspace/model_repos/llama3_ifb \
    --world_size 1

```




And we can check the logs of the container, you should be able to see the output
```bash

# Find active containers
docker ps
CONTAINER ID   IMAGE                        COMMAND                  CREATED          STATUS          PORTS     NAMES
ba5a4a84f245   triton-server-image:v24.04   "/opt/nvidia/nvidia_…"   16 minutes ago   Up 39 seconds             triton_server-triton-server-1


# Display container logs
docker logs ba5a4a84f245


I0522 14:51:21.484246 381 server.cc:634] 
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

I0522 14:51:21.485254 381 server.cc:677] 
+------------------+---------+--------+
| Model            | Version | Status |
+------------------+---------+--------+
| ensemble         | 1       | READY  |
| postprocessing   | 1       | READY  |
| preprocessing    | 1       | READY  |
| tensorrt_llm     | 1       | READY  |
| tensorrt_llm_bls | 1       | READY  |
+------------------+---------+--------+

I0522 14:51:21.500442 381 metrics.cc:877] Collecting metrics for GPU 0: NVIDIA GeForce RTX 3090
I0522 14:51:21.502515 381 metrics.cc:770] Collecting CPU metrics
I0522 14:51:21.502673 381 tritonserver.cc:2538] 
+----------------------------------+------------------------------------------------------------------------+
| Option                           | Value                                                                  |
+----------------------------------+------------------------------------------------------------------------+
| server_id                        | triton                                                                 |
| server_version                   | 2.45.0                                                                 |
| server_extensions                | classification sequence model_repository model_repository(unload_depen |
|                                  | dents) schedule_policy model_configuration system_shared_memory cuda_s |
|                                  | hared_memory binary_tensor_data parameters statistics trace logging    |
| model_repository_path[0]         | /workspace/model_repos/llama3_ifb                            |
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

I0522 14:51:21.508040 381 grpc_server.cc:2463] Started GRPCInferenceService at 0.0.0.0:8001
I0522 14:51:21.508186 381 http_server.cc:4692] Started HTTPService at 0.0.0.0:8000
I0522 14:51:21.549483 381 http_server.cc:362] Started Metrics Service at 0.0.0.0:8002

```


We can check if the server is up and running

```bash

curl -v localhost:8000/v2/health/ready


# output

* Request completely sent off
< HTTP/1.1 200 OK
< Content-Length: 0
< Content-Type: text/plain


```


## Test runs

We can now start to test the Triton inference server by send HTTP request to the endpoint. On the host machine, run the following command to send a very simple request. You may notice the output is not so great, the reason is because we didn't apply chat template to the input text.

```bash

curl -H "Content-Type: application/json" \
    -X POST localhost:8000/v2/models/ensemble/generate -d \
    '{
        "text_input": "Tell me a short joke about a dog and a cat.",
        "parameters": {
        "max_tokens": 256,
        "stop_words":["<|eot_id|>"]
        }
    }'


# # Output 
{
    "context_logits": 0.0,
    "cum_log_probs": 0.0,
    "generation_logits": 0.0,
    "model_name": "ensemble",
    "model_version": "1",
    "output_log_probs": [
        0.0,
        0.0,
        ...,
        0.0,
        0.0
    ],
    "sequence_end": false,
    "sequence_id": 0,
    "sequence_start": false,
    "text_output": " I'll try to come up with a joke on the spot.\n\nHere's a joke: Why did the dog and cat go to the vet?\n\nBecause they were feeling a little paws-itive!\n\nNow it's your turn! Can you come up with a joke about a dog and a cat? Go ahead and give it a try!<|eot_id|>"
}


```


We can try to add the proper chat template format to the text, which gives much better results.
```bash

curl -H "Content-Type: application/json" \
    -X POST localhost:8000/v2/models/ensemble/generate -d \
    '{
        "text_input": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nTell me a short joke about a dog and a cat.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "parameters": {
        "max_tokens": 256,
        "stop_words":["<|eot_id|>"]
        }
    }'


# Output
{
    "context_logits": 0.0,
    "cum_log_probs": 0.0,
    "generation_logits": 0.0,
    "model_name": "ensemble",
    "model_version": "1",
    "output_log_probs": [
        0.0,
        0.0,
        ...,
        0.0,
        0.0
    ],
    "sequence_end": false,
    "sequence_id": 0,
    "sequence_start": false,
    "text_output": "Why did the dog and cat go to the vet?\n\nBecause the cat was feeling paws-itive it had a doggone cold!<|eot_id|>"
}


```


Run more test
```bash


curl -H "Content-Type: application/json" \
    -X POST localhost:8000/v2/models/ensemble/generate -d \
    '{
        "text_input": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHow do I count to nine in French?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "parameters": {
        "max_tokens": 256,
        "stop_words":["<|eot_id|>"]
        }
    }'



# output
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
        ...,
        0.0,
        0.0
    ],
    "sequence_end": false,
    "sequence_id": 0,
    "sequence_start": false,
    "text_output": "Counting to nine in French is easy! Here it goes:\n\n1. Un (one)\n2. Deux (two)\n3. Trois (three)\n4. Quatre (four)\n5. Cinq (five)\n6. Six (six)\n7. Sept (seven)\n8. Huit (eight)\n9. Neuf (nine)\n\nVoilà!<|eot_id|>"
}
```

Streaming API


To use the streaming API, we need to set the following properties inside the `tensorrt_llm\config.pbtxt`.
```
model_transaction_policy {
  decoupled: True
}
```


We can then run the test using this command:
```bash


curl -H "Content-Type: application/json" \
    -X POST localhost:8000/v2/models/ensemble/generate_stream -d \
    '{
        "text_input": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHow do I count to nine in French?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "parameters": {
        "max_tokens": 256,
        "stream": true,
        "stop_words":["<|eot_id|>"]
        }
    }'


# Output

data: {"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":0.0,"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":"Count"}

data: {"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":"ing"}

data: {"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":" to"}

data: {"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":" nine"}

data: {"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":" in"}

data: {"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":" French"}

data: {"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":" is"}

data: {"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":" easy"}

data: {"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":"!"}

...,

data: {"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":" Ne"}

data: {"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":"uf"}

data: {"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":" ("}

data: {"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":"nine"}

data: {"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":")\n\n"}

data: {"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":"Vo"}

data: {"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":"il"}

data: {"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":"à"}

data: {"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":"!"}

data: {"context_logits":0.0,"cum_log_probs":0.0,"generation_logits":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":"<|eot_id|>"}

```




## Benchmark the Triton inference server

On the host machine or other client, run the following command to do a quick concurrency test:

```bash

cd deploy_llm


# Run concurrency benchmark tests
python3 scripts/benchmark_concurrency.py --max_gen_len 256


# Plot concurrency benchmark test results
python3 scripts/plot_concurrency_results.py

```


We can monitor the status by open a new terminal on the host machine and run the following command to query the Triton server metrics

```bash

curl localhost:8002/metrics

```