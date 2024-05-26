# Use the NVIDIA CUDA base image
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# # Set environment variables for proxy (if needed), here I have proxy server running on my host machine on port 1081
# ENV https_proxy=http://172.17.0.1:1081
# ENV http_proxy=http://172.17.0.1:1081


# Install necessary packages
RUN apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev git

# Install tensorrt_llm v0.9.0 and other Python packages
RUN pip3 install tensorrt_llm==0.9.0 -U --extra-index-url https://pypi.nvidia.com \
    && pip3 install tiktoken blobfile datasets evaluate rouge_score sentencepiece


# # Undo set environment variables for proxy
# ENV https_proxy=''
# ENV http_proxy=''


# Replace the llama model class to fix hard-coded vocab size problem
# This is only required for tensorrt_llm v0.9.0
COPY ./src/trtllm_v0.9.0_llama_model.py /usr/local/lib/python3.10/dist-packages/tensorrt_llm/models/llama/model.py

# Set the entrypoint to bash (optional, for interactive use)
ENTRYPOINT ["/bin/bash"]