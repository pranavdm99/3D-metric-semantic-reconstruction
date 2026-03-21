#!/bin/bash
# Quick launcher for Reconstruction: Neural Reconstruction

set -e

echo "🚀 Starting Reconstruction: Neural Reconstruction & Surface Alignment"
echo "=================================================="

# Check GPU
nvidia-smi || { echo "❌ Error: NVIDIA GPU not found"; exit 1; }

# Check pipeline configuration
if [ ! -f "configs/pipeline_config.json" ]; then
    echo "No pipeline configuration found at configs/pipeline_config.json"
    echo ""
    echo "Please place your pipeline configuration there first:"
    echo "  cp /path/to/your/pipeline_config.json configs/pipeline_config.json"
    exit 1
fi

# Check if input file paths in the config are valid
if [ ! -f "$(jq -r .reconstruction.input_video configs/pipeline_config.json)" ]; then
    echo "No input video found at $(jq -r .reconstruction.input_video configs/pipeline_config.json)"
    echo ""
    echo "Please place your multi-view video there first:"
    echo "  cp /path/to/your/video.mp4 $(jq -r .reconstruction.input_video configs/pipeline_config.json)"
    exit 1
fi

# Check if AR data path is valid
if [ ! -d "$(jq -r .reconstruction.ar_data_path configs/pipeline_config.json)" ]; then
    echo "No AR data found at $(jq -r .reconstruction.ar_data_path configs/pipeline_config.json)"
    echo ""
    echo "Please place your AR data there first:"
    echo "  cp -r /path/to/your/ar_data $(jq -r .reconstruction.ar_data_path configs/pipeline_config.json)"
    exit 1
fi

# Run the pipeline
python /workspace/src/reconstruction.py \
    --config /workspace/configs/pipeline_config.json

echo "✅ Reconstruction complete!"
