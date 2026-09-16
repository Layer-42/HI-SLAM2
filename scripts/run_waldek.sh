#!/usr/bin/env bash
set -eo pipefail

VIDEO="data/waldek_data/video.MP4"
DATA_DIR="data/waldek_data"
OUTPUT_DIR="outputs"

# preprocess
python scripts/preprocess_owndata.py \
  "${VIDEO}" \
  "${DATA_DIR}" \
  1 \
  584


# Demo
python demo.py \
  --imagedir "${DATA_DIR}/images" \
  --calib "${DATA_DIR}/calib.txt" \
  --config config/waldek_performance.yaml \
  --output "${OUTPUT_DIR}" \
  --scene waldek_data \
  --buffer 400 \
  --undistort