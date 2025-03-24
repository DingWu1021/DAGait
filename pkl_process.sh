#!/bin/bash

# Change the following to your path
SIL_PATH="/data/wzx/Gait3D_transform"
OUTPUT_PATH="/data/wzx/Gait3D_transform_pkl"

# 执行 Python 脚本
python datasets/Gait3D/process_gait3d_alignment.py \
    --sil_path "$SIL_PATH" \
    --output_path "$OUTPUT_PATH"