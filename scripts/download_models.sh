#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${ROOT_DIR}/pretrained_models"

mkdir -p "${MODEL_DIR}"

download_model() {
    filename="$1"
    output="${MODEL_DIR}/${filename}"

    if [ -s "${output}" ]; then
        echo "${filename} already exists"
        return
    fi

    echo "Downloading ${filename}"

    wget \
        --tries=3 \
        --timeout=30 \
        "https://zenodo.org/records/10447888/files/${filename}" \
        -O "${output}"
}

download_model "omnidata_dpt_normal_v2.ckpt"
download_model "omnidata_dpt_depth_v2.ckpt"

if [ ! -s "${MODEL_DIR}/droid.pth" ]; then
    echo "Downloading droid.pth"
    if command -v gdown &>/dev/null; then
        gdown "https://drive.google.com/uc?id=1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh" -O "${MODEL_DIR}/droid.pth" || true
    else
        wget --tries=3 --timeout=30 "https://huggingface.org/datasets/yihua7/HI-SLAM2/resolve/main/droid.pth" -O "${MODEL_DIR}/droid.pth" || true
    fi
fi

echo "Pretrained models ready at ${MODEL_DIR}"