

# 1. Preparation

## 1.1 Download Llama3 model and tokenizer checkpoints

Please refer to Meta's llama3 repository on how to download the native model and tokenizer checkpoints (not based on HuggingFace).


## 1.2 Install Docker on Host OS

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


# 2. Build an optimized TensorRT-LLM model engine

Before we can serve our LLM model for inference using Triton inference server, we need to build an optimized TensorRT engine (more precisely TensorRT-LLM engine). This process is often a trial-and-error process, which may involve multiple iterations.

Please refer to `PART_ONE.md` on how to build a such engine.



# 3. Serve the TensorRT-LLM model engine using Triton inference server

Once we have an complied TensorRT-LLM model engine, we can then deploy the engine to Triton inference server.
The Triton server will utilize TensorRT-LLM-Backend to serve the engine, along with some preprocessing and postprocessing tasks (text encode and token decode).

Please refer to `PART_TWO.md` on how to deploy the engine with Triton inference server.

