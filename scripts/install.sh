#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "Installing HI-SLAM2"

nvcc --version
gcc --version | head -n 1
python --version
colmap -h >/dev/null

git config --global --add safe.directory '*'
if [ -d "${ROOT_DIR}/.git" ]; then
    git submodule sync --recursive || true
    git submodule update --init --recursive || true
fi

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

TMP_DIR="$(mktemp -d)"

cp -a thirdparty/simple-knn \
    "${TMP_DIR}/simple-knn"

python -m pip install \
    --no-build-isolation \
    "${TMP_DIR}/simple-knn"

cp -a thirdparty/diff-gaussian-rasterization \
    "${TMP_DIR}/diff-gaussian-rasterization"

python -m pip install \
    --no-build-isolation \
    "${TMP_DIR}/diff-gaussian-rasterization"

rm -rf "${TMP_DIR}"

echo "Building HI-SLAM2"

python setup.py install

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