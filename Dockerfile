FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ARG COLMAP_VERSION=3.8
ARG CUDA_ARCHITECTURES="75;80;86;89"

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6+PTX"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    wget \
    curl \
    ca-certificates \
    build-essential \
    gcc-10 \
    g++-10 \
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
    libusb-1.0-0 \
    libglib2.0-0 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libglfw3 \
    libsm6 \
    libxrandr2 \
    libxi6 \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone \
    --branch "${COLMAP_VERSION}" \
    --depth 1 \
    https://github.com/colmap/colmap.git \
    /tmp/colmap && \
    CC=/usr/bin/gcc-10 \
    CXX=/usr/bin/g++-10 \
    CUDAHOSTCXX=/usr/bin/g++-10 \
    cmake \
    -S /tmp/colmap \
    -B /tmp/colmap/build \
    -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
    -DCUDA_ENABLED=ON && \
    cmake --build /tmp/colmap/build --parallel && \
    cmake --install /tmp/colmap/build && \
    rm -rf /tmp/colmap

RUN update-alternatives \
    --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 && \
    update-alternatives \
    --install /usr/bin/g++ g++ /usr/bin/g++-11 100 && \
    update-alternatives \
    --install /usr/bin/python python /usr/bin/python3 100

RUN python -m pip install --upgrade \
    pip \
    setuptools \
    wheel

WORKDIR /workspace/HI-SLAM2

COPY . .

RUN python -m pip install \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118

RUN python -m pip install \
    numpy==1.26.4 \
    setuptools==69.5.1 \
    scipy \
    opencv-python \
    tqdm \
    matplotlib \
    pyyaml \
    lightning \
    wheel \
    torchmetrics \
    pyrender \
    imgviz \
    timm \
    open3d \
    evo \
    munch \
    plyfile \
    rich \
    glfw \
    PyGLM \
    huggingface_hub

RUN python -m pip install \
    git+https://github.com/eriksandstroem/evaluate_3d_reconstruction_lib.git

RUN python -m pip install \
    --no-build-isolation \
    torch-scatter \
    -f https://data.pyg.org/whl/torch-2.1.2+cu118.html

RUN python -m pip install \
    --no-build-isolation \
    ./thirdparty/simple-knn

RUN python -m pip install \
    --no-build-isolation \
    ./thirdparty/diff-gaussian-rasterization

RUN python setup.py install

RUN python - <<'PY'
import torch
import torch_scatter
import simple_knn
import diff_gaussian_rasterization
import droid_backends
import lietorch

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("torch-scatter: OK")
print("simple-knn: OK")
print("diff-gaussian-rasterization: OK")
print("droid_backends: OK")
print("lietorch: OK")
PY

ARG USERNAME=hislam2
ARG UID=1000
ARG GID=1000

RUN groupadd --gid "${GID}" "${USERNAME}" \
    && useradd \
        --uid "${UID}" \
        --gid "${GID}" \
        --create-home \
        --shell /bin/bash \
        "${USERNAME}" \
    && if getent group video >/dev/null; then usermod -aG video "${USERNAME}"; fi \
    && if getent group render >/dev/null; then usermod -aG render "${USERNAME}"; fi \
    && mkdir -p \
        "/home/${USERNAME}/.cache/huggingface" \
        "/home/${USERNAME}/.cache/torch" \
    && chown -R \
        "${USERNAME}:${USERNAME}" \
        "/home/${USERNAME}" \
        /workspace/HI-SLAM2

USER ${USERNAME}

ENV HOME=/home/hislam2
ENV PATH=/home/hislam2/.local/bin:${PATH}
ENV HF_HOME=/home/hislam2/.cache/huggingface
ENV TORCH_HOME=/home/hislam2/.cache/torch
ENV XDG_CACHE_HOME=/home/hislam2/.cache

CMD ["/bin/bash"]