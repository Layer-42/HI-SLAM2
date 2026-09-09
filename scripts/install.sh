#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "Installing HI-SLAM2"

nvcc --version
gcc --version | head -n 1
python --version

git config --global --add safe.directory '*'
git submodule sync --recursive
git submodule update --init --recursive

echo "Installing PyTorch"

python -m pip install \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118

echo "Installing Python dependencies"

python -m pip install \
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

python -m pip install \
    git+https://github.com/eriksandstroem/evaluate_3d_reconstruction_lib.git

echo "Installing CUDA extensions"

python -m pip install \
    --no-build-isolation \
    torch-scatter \
    -f https://data.pyg.org/whl/torch-2.1.2+cu118.html

python -m pip install \
    --no-build-isolation \
    ./thirdparty/simple-knn

python -m pip install \
    --no-build-isolation \
    ./thirdparty/diff-gaussian-rasterization

echo "Building HI-SLAM2"

python setup.py install

echo "Downloading pretrained models"

mkdir -p pretrained_models

download_model() {
    local filename="$1"
    local output="pretrained_models/${filename}"
    local tmp="${output}.part"

    if [ -s "${output}" ]; then
        echo "${filename} already exists"
        return
    fi

    rm -f "${tmp}"

    if wget \
        --tries=3 \
        --timeout=30 \
        "https://zenodo.org/records/10447888/files/${filename}" \
        -O "${tmp}"; then
        mv "${tmp}" "${output}"
        return
    fi

    rm -f "${tmp}"

    hf download \
        clay3d/omnidata \
        "${filename}" \
        --local-dir pretrained_models
}

download_model "omnidata_dpt_normal_v2.ckpt"
download_model "omnidata_dpt_depth_v2.ckpt"

echo "Checking installation"

python - <<'PY'
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

echo "HI-SLAM2 installed"