# Use the NVIDIA PyTorch base image
FROM nvcr.io/nvidia/pytorch:24.07-py3

# Install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Update pip to the latest version
RUN pip install --no-cache-dir --upgrade pip

# Uninstall apex first
RUN pip uninstall -y apex

# # Install flash_attn separately with --use-pep517 flag
# RUN pip install --no-cache-dir --use-pep517 flash-attn==2.6.3 flask

RUN pip install -i https://mirrors.cloud.tencent.com/pypi/simple xfuser==0.3.1


RUN apt update \
    && apt install -y --no-install-recommends gnupg \
    && echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list \
    && apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub \
    && apt update \
    && apt install -y nsight-systems-cli

COPY . /workspace

WORKDIR /workspace