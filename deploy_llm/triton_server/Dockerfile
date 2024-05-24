# Use the NVIDIA Triton Inference Server image
# here `24.04` standars for release `yy.mm`
FROM nvcr.io/nvidia/tritonserver:24.04-trtllm-python-py3

# Set environment variables for proxy (if needed), here I have proxy server running on my host machine on port 1081
ENV https_proxy=http://172.17.0.1:1081
ENV http_proxy=http://172.17.0.1:1081
ENV NO_PROXY localhost,127.0.0.1,::1

# Install tiktoken packages
RUN pip3 install protobuf tiktoken blobfile


# Set the working directory
WORKDIR /workspace

# # Copy the llama3 model engine and other TensorRT-LLM models, we disabled it to save space
# COPY . /workspace


# Expose any necessary ports (if needed)
EXPOSE 8000 8001 8002


# Default command to run when container starts
CMD ["python3", "start.py", "--model_repo", "/workspace/model_repos/llama3_ifb", "--world_size", "1"]
