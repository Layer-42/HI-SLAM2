FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    wget \
    curl \
    ca-certificates \
    build-essential \
    gcc-11 \
    g++-11 \
    cmake \
    ninja-build \
    pkg-config \
    python3 \
    python3-pip \
    python3-dev \
    unzip \
    libgl1 \
    libegl1 \
    libglib2.0-0 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libglfw3 \
    libsm6 \
    libxrandr2 \
    libxi6 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 100

RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace/HI-SLAM2

COPY . .

RUN bash scripts/install.sh

CMD ["/bin/bash"]