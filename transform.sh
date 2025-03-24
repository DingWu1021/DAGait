#!/bin/bash

# Change the following to your path
SKELETON_ROOT="/data/Gait3d/2D_Poses"
SILHOUETTE_ROOT="/data/Gait3d/2D_Silhouettes"
OUTPUT_ROOT="/data/Gait3D_transform"
LOG_DIR="/data/wzx/output"
SCORE_THRESHOLD=0.3

python transform.py \
    --skeleton_root "$SKELETON_ROOT" \
    --silhouette_root "$SILHOUETTE_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --log_dir "$LOG_DIR" \
    --score_threshold "$SCORE_THRESHOLD"