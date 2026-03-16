#!/bin/bash
# Quick launcher for Physics: Physics Extraction

set -e

echo "🚀 Starting Physics: Automated Physics Extraction"
echo "=============================================="

# Check Segmentation output
if [ ! -f "/workspace/data/outputs/masks/segmentation_metadata.json" ]; then
    echo "❌ Error: Segmentation output not found. Run Segmentation first!"
    exit 1
fi

# Check GPU (optional for this stage, but helpful for large meshes)
nvidia-smi || echo "⚠️  GPU not found. CPU-only mode."

# Run the pipeline
python /workspace/src/physics.py \
    --config /workspace/configs/pipeline_config.json

echo "✅ Physics complete!"
