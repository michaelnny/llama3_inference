# Use the NVIDIA Triton Inference Server image
# here `24.04` standars for release `yy.mm`
# Notice, this iamge only has 'TensorRT-LLM' and 'Python' backend
FROM nvcr.io/nvidia/tritonserver:24.04-trtllm-python-py3

# Support for Tensorflow, PyTorch, TensorRT, ONNX and OpenVINO backends
FROM nvcr.io/nvidia/tritonserver:24.04-py3


# # Set environment variables for proxy (if needed), here I have proxy server running on my host machine on port 1081
ENV https_proxy=http://172.17.0.1:1081
ENV http_proxy=http://172.17.0.1:1081


COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade -r requirements.txt


# # Undo set environment variables for proxy
ENV https_proxy=''
ENV http_proxy=''


# Copy ONNX backend to container as this is required for the Stentence Transformer model  
COPY ./onnxruntime /opt/tritonserver/backends/onnxruntime


# Set the working directory
WORKDIR /app

# # Copy the llama3 model engine and other TensorRT-LLM models, we disabled it to save space
# COPY ./app /app


# Expose any necessary ports (if needed)
EXPOSE 8000 8001 8002


# Default command to run when container starts
CMD ["python3", "start.py", "--model_repo", "./model_repos", "--world_size", "1", "--tensorrt_llm_model_name", "llama3_tensorrt_llm"]
