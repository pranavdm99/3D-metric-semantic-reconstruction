#!/bin/bash
# Master script to run all phases sequentially

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   Semantic Scene Reconstruction Pipeline - Full Execution  ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check prerequisites
if ! command -v docker &> /dev/null; then
    echo "docker not found. Please install Docker."
    exit 1
fi

if ! command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi not found. Please install NVIDIA drivers."
    exit 1
fi

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

echo "Prerequisites check passed"
echo ""

# Confirm start
read -p "This will run the pipeline sequentially (~3-4 hours). Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

START_TIME=$(date +%s)

# Reconstruction
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  STAGE 1/4: Neural Reconstruction & Surface Alignment"
echo "════════════════════════════════════════════════════════════"
docker compose run --rm reconstruction bash /workspace/scripts/reconstruction.sh
echo "Reconstruction complete"

# Segmentation
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  STAGE 2/4: Semantic Grounding & 3D Segmentation"
echo "════════════════════════════════════════════════════════════"
docker compose run --rm segmentation bash /workspace/scripts/segmentation.sh
echo "Segmentation complete"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                 PIPELINE COMPLETE!                         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Total time: ${HOURS}h ${MINUTES}m"
echo ""