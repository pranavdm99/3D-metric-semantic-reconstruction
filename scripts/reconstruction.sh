#!/bin/bash
# Quick launcher for Reconstruction: Neural Reconstruction

set -e

echo "🚀 Starting Reconstruction: Neural Reconstruction & Surface Alignment"
echo "=================================================="

# Check if video file exists
if [ ! -f "/workspace/data/raw_video/input.mp4" ]; then
    echo "❌ Error: No input video found at /workspace/data/raw_video/input.mp4"
    echo "Please place your multi-view video in data/raw_video/ and name it input.mp4"
    exit 1
fi

# Check GPU
nvidia-smi || { echo "❌ Error: NVIDIA GPU not found"; exit 1; }

# Run the pipeline
python /workspace/src/reconstruction.py \
    --config /workspace/configs/pipeline_config.json

echo "✅ Reconstruction complete!"
