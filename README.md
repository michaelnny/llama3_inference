# 1. Preparation


## 1.1 Install Docker on Host OS

Install the Docker runtime on the host machine, the following is an example of installing docker on OpenSUSE linux.

Source: https://en.opensuse.org/Docker

```bash
# Install Docker packages:
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

## 1.3 Install NVIDIA Container Toolkit

In order to user NVIDIA GPUs inside docker container, we need to install NVIDIA Container Toolkit. The following is an example of installation process on OpenSUSE linux.

Source: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-zypper

```bash
sudo zypper ar https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo

sudo zypper modifyrepo --enable nvidia-container-toolkit-experimental

sudo zypper --gpg-auto-import-keys install nvidia-container-toolkit
```

After the installation is completed, we need to configure Docker runtime and restart the docker service.

```bash
sudo nvidia-ctk runtime configure --runtime=docker


sudo systemctl restart docker
```

# 2. Build optimized model engines

Before we can serve our LLM model for inference using Triton inference server, we need to build an optimized TensorRT engine (more precisely TensorRT-LLM engine). An engine is simple the optimized computation model/graph, along with all the weights. This process is often a trial-and-error process, which may involve multiple iterations.

In addition to build a engine for llama3 chat model, we will also need to build an open-source text embedding model, such as the one from SentenceTransformers, we'll export the SentenceTransformers model to ONNX format.

Please refer to `build_models`, where you can follow the instructions inside `build_models/README.md` on how to build a optimal TensorRT-LLM engine.

# 3. Deploy the model engines to Triton inference server

Once we have an complied TensorRT-LLM model engine, we can then deploy the engine to Triton inference server.
The Triton server will utilize TensorRT-LLM-Backend to serve the engine, along with some preprocessing and postprocessing tasks (text encode and token decode).

In addition, to deploy TensorRT-LLM model engine, we'll also deploy the ONNX model for the text embedding model.

Please refer to `deploy_triton_server`, where you can follow the instructions inside `deploy_triton_server/README.md` on how to deploy the engine with Triton inference server.

# 4. Deploy API Gateway server

Since the Triton inference server focus on low-level computation and optimization, it does not provide a very easy to use API, especially if we want to use openAI API's client to connect to our system.

To solve this issue, we create a simple API gateway server using FastAPI, which provide similar API design to the openAI API.

Please refer to `deploy_api_server`, where you can follow the instructions inside `deploy_api_server/README.md` on how to deploy the API gateway server based on FastAPI for openAI API compatibilities.
