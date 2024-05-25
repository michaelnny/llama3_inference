# Build an optimized TensorRT-LLM engine

Part one of the project mainly focus on turning an trained LLM model like Llama3 into an optimized TensorRT-LLM engine.

TensorRT-LLM is build on top of TensorRT, where the later is an accelerated backend architecture for Nvidia's Triton inference server solution.

TensorRT-LLM implements some transformer specific kernels like FlashAttention which target transformer based LLM.

To get started, we first need to install the TensorRT-LLM package, it's recommended to user a Docker container for the process.

**Prerequisites**

You need to make sure your host machine has Docker runtime and NVIDIA Container Toolkit installed.

- Install Docker on OpenSuse: https://en.opensuse.org/Docker
- Install NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html

## Prepare TensorRT-LLM workspace

The TensorRT-LLM project contains additional script to do testing and benchmarking, where you can't find inside the `pip` installed package. On your host machine, clone the TensorRT-LLM project, notice here we use the same version (v0.9.0) as we will install the package inside the container (specified in the Dockerfile).

```bash

cd ~/dl_projects

git clone -b v0.9.0 https://github.com/NVIDIA/TensorRT-LLM.git

```

Now lets copy some custom testing scripts to the examples folder inside the TensorRT-LLM project, these custom scripts are adjusted to use native tiktoken tokenizer.

```bash

cd TensorRT-LLM

# Copy our custom testing code to the TensorRT-LLM workspace
cp -r ~/dl_projects/llama3_inference/build_trtllm_engine/src/tokenizer.py ./examples
cp -r ~/dl_projects/llama3_inference/build_trtllm_engine/src/run_llama3.py ./examples
cp -r ~/dl_projects/llama3_inference/build_trtllm_engine/src/dolly-15k.jsonl ./examples


# Notice as of May 2024, the v0.9.0 has multiple bug of using fixed vocab size inside the model class
cp -r ~/dl_projects/llama3_inference/build_trtllm_engine/src/trtllm_v0.9.0_llama_model.py ./tensorrt_llm/models/llama/model.py

```

## Build the Docker container for TensorRT-LLM

We start by pulling the NVIDIA CUDA docker image, the process are maintained inside the `build_trtllm_engine/Dockerfile`. On your host machine, navigate to `build_trtllm_engine` folder, and run the following command to build the TensorRT-LLM container and save it to a image called `tensorrtllm-image:v0.9.0`.

```bash

# Build the Docker image with the custom name
docker-compose build


# Or we can use docker build to manually specify the name
docker build -t tensorrtllm-image:v0.9.0 .

```

## Launch TensorRT-LLM container

We can then launch the container using `docker-compose run` command, notice we need to maintain the volume paths for the `TensorRT-LLM v0.9.0` project and the `llama3 checkpoints` inside the `docker-compose.yaml` file.

```bash

docker-compose run --rm tensorrtllm

```

Or we can start the container directly use `docker run`, as shown in this example:

```bash

docker run --rm --gpus all --volume ${PWD}:/workspace \
    --volume /home/michael/dl_projects/TensorRT-LLM:/TensorRT-LLM \
    --volume /home/michael/models/Llama3:/checkpoints \
    --entrypoint /bin/bash -it --workdir /TensorRT-LLM \
    tensorrtllm-image:v0.9.0

```

## Build TensorRT-LLM model engine

**All commands in this section should be executed inside the TensorRT-LLM container**

To build the TensorRT engine, we first need to convert the regular llama3 model checkpoint to a TensorRT-LLM compatible format.
The `TensorRT-LLM/examples/llama/convert_checkpoint.py` file can be used to do this.

The following command will load the original Meta llama model checkpoint from `/checkpoints/Meta-Llama-3-8B-Instruct`, convert to `bfloat16`, and save the files to `/workspace/tmp/trtllm-Llama-3-8B-Instruct-1gpu-bf16` so we can later us it to build the TensorRT engine.

```bash
python3 examples/llama/convert_checkpoint.py --meta_ckpt_dir /checkpoints/Meta-Llama-3-8B-Instruct \
            --output_dir /workspace/tmp/trtllm-Llama-3-8B-Instruct-1gpu-bf16 \
            --dtype bfloat16 \
            --tp_size 1



[TensorRT-LLM] TensorRT-LLM version: 0.9.0
0.9.0
Total time of converting checkpoints: 00:00:20
```

We can then build the TensorRT-LLM model engine using the converted checkpoint from the above step. We may have to try different configurations to build the optimal engine depending on the environment. You can refer to the `benchmark section` to evaluate the performance of an TensorRT-LLM model engine.

Note if using v0.8.0 version, we may need to remove `--gemm_plugin` option in order to use in-flight batching.

```bash
trtllm-build --checkpoint_dir /workspace/tmp/trtllm-Llama-3-8B-Instruct-1gpu-bf16 \
            --output_dir /workspace/tmp/trt_engines/Llama-3-8B-Instruct-1gpu-bf16 \
            --paged_kv_cache enable \
            --context_fmha enable \
            --remove_input_padding enable \
            --gpt_attention_plugin bfloat16 \
            --gemm_plugin bfloat16 \
            --max_batch_size 64 \
            --max_input_len 1024 \
            --max_output_len 1024 \
            --max_num_tokens 16384 \
            --strongly_typed



[TensorRT-LLM] TensorRT-LLM version: 0.9.0
[05/22/2024-14:16:52] [TRT-LLM] [I] Set bert_attention_plugin to float16.
[05/22/2024-14:16:52] [TRT-LLM] [I] Set gpt_attention_plugin to bfloat16.
[05/22/2024-14:16:52] [TRT-LLM] [I] Set gemm_plugin to bfloat16.
[05/22/2024-14:16:52] [TRT-LLM] [I] Set lookup_plugin to None.
[05/22/2024-14:16:52] [TRT-LLM] [I] Set lora_plugin to None.
[05/22/2024-14:16:52] [TRT-LLM] [I] Set moe_plugin to float16.
[05/22/2024-14:16:52] [TRT-LLM] [I] Set mamba_conv1d_plugin to float16.
[05/22/2024-14:16:52] [TRT-LLM] [I] Set context_fmha to True.
[05/22/2024-14:16:52] [TRT-LLM] [I] Set context_fmha_fp32_acc to False.
[05/22/2024-14:16:52] [TRT-LLM] [I] Set paged_kv_cache to True.
[05/22/2024-14:16:52] [TRT-LLM] [I] Set remove_input_padding to True.
[05/22/2024-14:16:52] [TRT-LLM] [I] Set use_custom_all_reduce to True.
[05/22/2024-14:16:52] [TRT-LLM] [I] Set multi_block_mode to False.
[05/22/2024-14:16:52] [TRT-LLM] [I] Set enable_xqa to True.
[05/22/2024-14:16:52] [TRT-LLM] [I] Set attention_qk_half_accumulation to False.
[05/22/2024-14:16:52] [TRT-LLM] [I] Set tokens_per_block to 128.
[05/22/2024-14:16:52] [TRT-LLM] [I] Set use_paged_context_fmha to False.
[05/22/2024-14:16:52] [TRT-LLM] [I] Set use_fp8_context_fmha to False.
[05/22/2024-14:16:52] [TRT-LLM] [I] Set use_context_fmha_for_generation to False.
[05/22/2024-14:16:52] [TRT-LLM] [I] Set multiple_profiles to False.
[05/22/2024-14:16:52] [TRT-LLM] [I] Set paged_state to True.
[05/22/2024-14:16:52] [TRT-LLM] [I] Set streamingllm to False.
[05/22/2024-14:16:52] [TRT-LLM] [W] remove_input_padding is enabled, while opt_num_tokens is not set, setting to max_batch_size*max_beam_width.

[05/22/2024-14:16:52] [TRT-LLM] [W] Fail to infer cluster key, use A100-SXM-80GB as fallback.
[05/22/2024-14:16:52] [TRT] [I] [MemUsageChange] Init CUDA: CPU +14, GPU +0, now: CPU 167, GPU 269 (MiB)
[05/22/2024-14:16:54] [TRT] [I] [MemUsageChange] Init builder kernel library: CPU +1810, GPU +312, now: CPU 2113, GPU 581 (MiB)
[05/22/2024-14:16:54] [TRT-LLM] [I] Set nccl_plugin to None.
[05/22/2024-14:16:54] [TRT-LLM] [I] Set use_custom_all_reduce to True.
[05/22/2024-14:16:54] [TRT-LLM] [I] Build TensorRT engine Unnamed Network 0
[05/22/2024-14:16:54] [TRT] [W] Unused Input: position_ids
[05/22/2024-14:16:54] [TRT] [W] [RemoveDeadLayers] Input Tensor position_ids is unused or used only at compile-time, but is not being removed.
[05/22/2024-14:16:54] [TRT] [I] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +8, now: CPU 2141, GPU 603 (MiB)
[05/22/2024-14:16:54] [TRT] [I] [MemUsageChange] Init cuDNN: CPU +2, GPU +10, now: CPU 2143, GPU 613 (MiB)
[05/22/2024-14:16:54] [TRT] [W] TensorRT was linked against cuDNN 8.9.6 but loaded cuDNN 8.9.2
[05/22/2024-14:16:54] [TRT] [I] Global timing cache in use. Profiling results in this builder pass will be stored.
[05/22/2024-14:16:56] [TRT] [I] [GraphReduction] The approximate region cut reduction algorithm is called.
[05/22/2024-14:16:56] [TRT] [I] Detected 13 inputs and 1 output network tensors.
[05/22/2024-14:17:03] [TRT] [I] Total Host Persistent Memory: 52976
[05/22/2024-14:17:03] [TRT] [I] Total Device Persistent Memory: 0
[05/22/2024-14:17:03] [TRT] [I] Total Scratch Memory: 268500992
[05/22/2024-14:17:03] [TRT] [I] [BlockAssignment] Started assigning block shifts. This will take 558 steps to complete.
[05/22/2024-14:17:03] [TRT] [I] [BlockAssignment] Algorithm ShiftNTopDown took 13.1927ms to assign 15 blocks to 558 nodes requiring 1577063424 bytes.
[05/22/2024-14:17:03] [TRT] [I] Total Activation Memory: 1577062400
[05/22/2024-14:17:03] [TRT] [I] Total Weights Memory: 16060522496
[05/22/2024-14:17:03] [TRT] [I] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +8, now: CPU 2161, GPU 15939 (MiB)
[05/22/2024-14:17:03] [TRT] [I] [MemUsageChange] Init cuDNN: CPU +0, GPU +10, now: CPU 2161, GPU 15949 (MiB)
[05/22/2024-14:17:03] [TRT] [W] TensorRT was linked against cuDNN 8.9.6 but loaded cuDNN 8.9.2
[05/22/2024-14:17:03] [TRT] [I] Engine generation completed in 9.05298 seconds.
[05/22/2024-14:17:03] [TRT] [I] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 0 MiB, GPU 15317 MiB
[05/22/2024-14:17:03] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in building engine: CPU +0, GPU +15317, now: CPU 0, GPU 15317 (MiB)
[05/22/2024-14:17:09] [TRT] [I] [MemUsageStats] Peak memory usage during Engine building and serialization: CPU: 34537 MiB
[05/22/2024-14:17:09] [TRT-LLM] [I] Total time of building Unnamed Network 0: 00:00:15
[05/22/2024-14:17:09] [TRT] [I] Serialized 59 bytes of code generator cache.
[05/22/2024-14:17:09] [TRT] [I] Serialized 147429 bytes of compilation cache.
[05/22/2024-14:17:09] [TRT] [I] Serialized 5 timing cache entries
[05/22/2024-14:17:09] [TRT-LLM] [I] Timing cache serialized to model.cache
[05/22/2024-14:17:09] [TRT-LLM] [I] Serializing engine to ./tmp/trt_engines/Llama-3-8B-Instruct-1gpu-bf16/rank0.engine...
[05/22/2024-14:17:16] [TRT-LLM] [I] Engine serialized. Total time: 00:00:06
[05/22/2024-14:17:17] [TRT-LLM] [I] Total time of building all engines: 00:00:24

```

## Test TensorRT-LLM model engine

To manually test the engine, run the following command inside the container.

```bash

# Ask a simple question
python3 examples/run_llama3.py --max_output_len=256 \
                  --tokenizer_dir /checkpoints/Meta-Llama-3-8B-Instruct \
                  --engine_dir /workspace/tmp/trt_engines/Llama-3-8B-Instruct-1gpu-bf16 \
                  --input_text "Can you explain in simple words what's a black box?"




Input [Text 0]: "<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Can you explain in simple words what's a black box?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"
Output [Text 0 Beam 0]: "A "black box" is a term used to describe a system, process, or device that is mysterious or unknown, and its inner workings are not easily understood or visible.

Imagine a box that you can't open or see inside. You can't see what's happening inside the box, but you can see what's happening outside the box. For example, you might put a toy car into the box and then close the lid. When you open the box again, you might find that the toy car has changed shape or color. But you wouldn't know how it got that way, because you can't see what's happening inside the box.

In the same way, a black box in technology or science is a system that takes in some input (like data or a signal), does something to it, and then produces an output (like a result or a response). But the exact steps that the system takes to get from the input to the output are unknown or not easily understood.

For example, a black box might be a complex computer program that takes in some data and produces a prediction or recommendation. You might be able to use the program and get useful results, but you wouldn't know exactly how it's making those predictions or recommendations, because the inner workings of the"


```

We can also run the test on a predefined dataset, by give the `--input_file`, notice the file must be a valid `jsonl` file, where each row is a complete sample dict, and has the `instruct` property.

```bash

# Run with a prompt dataset
python3 examples/run_llama3.py --max_output_len=256 \
                  --tokenizer_dir /checkpoints/Meta-Llama-3-8B-Instruct \
                  --engine_dir /workspace/tmp/trt_engines/Llama-3-8B-Instruct-1gpu-bf16 \
                  --input_file examples/dolly-15k.jsonl \
                  --max_samples 64




Input [Text 61]: "<|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|begin_of_text|><|start_header_id|>user<|end_header_id|>

Is beauty objective or subjective?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"
Output [Text 61 Beam 0]: "The age-old question!

The answer is a resounding "it's complicated"!

In philosophy, beauty is often considered a subjective experience, meaning that it is a personal and individual perception. What one person finds beautiful, another person may not. This is because beauty is often tied to cultural, social, and personal values and experiences.

For example, what one person finds beautiful, another person might find ugly. This is because beauty is often a combination of physical, emotional, and cultural factors, such as:

1. **Physical characteristics**: Beauty can be influenced by physical features, such as symmetry, proportion, and proportions.
2. **Cultural and social norms**: Beauty can be influenced by cultural and social norms, values, and expectations.
3. **Personal experiences**: Beauty can be influenced by personal experiences, emotions, and memories.

However, there are also some aspects of beauty that can be considered objective, such as:

1. **Mathematical proportions**: In art, architecture, and design, beauty can be measured and quantified using mathematical proportions and proportions.
2. **Biological and psychological responses**: Research has shown that certain physical features, such as symmetry and facial attractiveness, can elicit specific emotional and psychological responses in humans.

So, is beauty objective or"
Input [Text 62]: "<|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|begin_of_text|><|start_header_id|>user<|end_header_id|>

Identify which instrument is string or percussion: Kpanlogo, Shamisen<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"
Output [Text 62 Beam 0]: "Based on the names, I would identify the instruments as follows:

* Kpanlogo: Percussion (Kpanlogo is a type of drum)
* Shamisen: String (Shamisen is a type of stringed instrument, similar to a lute or a harp)"
Input [Text 63]: "<|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|begin_of_text|><|start_header_id|>user<|end_header_id|>

Why do potato chip bags become stale after opening?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"
Output [Text 63 Beam 0]: "The eternal question! 🥔

Potato chips, like many other snack foods, can become stale or stale after opening due to a combination of factors. Here are some reasons why:

1. **Moisture loss**: When potato chips are packaged, they contain a certain level of moisture to maintain their texture and flavor. When you open the bag, the air inside the bag can cause the moisture to escape, leading to a loss of crispiness and freshness.
2. **Oxidation**: When potato chips are exposed to air, the starches on the surface of the chips can oxidize, leading to a loss of flavor and texture. This is more pronounced when the chips are exposed to heat, light, or oxygen.
3. **Starch retrogradation**: Starches in potato chips can retrograde (or break down) when they are exposed to heat, light, or moisture. This can cause the chips to become stale, dry, and crumbly.
4. **Packaging**: The type of packaging used for potato chips can also contribute to staleness. For example, if the bag is not airtight or if the seal is compromised, moisture can enter the bag and cause the chips to become stale faster.
5. **Storage"

```

## Benchmark TensorRT-LLM model engine

**Note**

Before start run benchmark, we need to add the llama3 model configuration to `benchmarks/python/allowed_configs.py`

Here's an example of the llama3 7B model, you can fine the parameters inside `config.json` located at the engine dir.

```python


"llama3_7b": ModelConfig(name="llama3_7b",
                family="llama",
                benchmark_type="gpt",
                build_config=BuildConfig(
                    num_layers=32,
                    num_heads=32,
                    hidden_size=4096,
                    vocab_size=128256,
                    hidden_act='silu',
                    n_positions=2048,
                    inter_size=14336,
                    max_batch_size=128,
                    max_input_len=512,
                    max_output_len=512,
                    builder_opt=None,
                )),

```

Inside the container, run the following command to benchmark the engine

```bash

cd benchmarks/python

# Run test over a single batch
python3 benchmark.py --model llama3_7b \
                --engine_dir /workspace/tmp/trt_engines/Llama-3-8B-Instruct-1gpu-bf16 \
                --mode plugin \
                --batch_size 32 \
                --max_input_len 128 \
                --max_output_len 256 \
                --input_output_len "128,256" \
                --dtype bfloat16


# Or we can run a series of tests to compare the performance on different batch size
for batch_size in 1 2 4 8 16 32 64
do
    python3 benchmark.py --model llama3_7b \
                    --engine_dir /workspace/tmp/trt_engines/Llama-3-8B-Instruct-1gpu-bf16 \
                    --mode plugin \
                    --batch_size $batch_size \
                    --max_input_len 128 \
                    --max_output_len 256 \
                    --input_output_len "128,256" \
                    --dtype bfloat16 \
                    --csv
done

```

We can then check the results in the csv file. Depend on the result, we may need to repeat the `build engine -> benchmark` process until we have find a good model engine. Then we can move on to deploy the model engine to Triton inference server.
