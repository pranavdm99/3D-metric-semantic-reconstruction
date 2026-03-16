#!/bin/bash
# Quick launcher for Assembly: MJCF Assembly

set -e

echo "🚀 Starting Assembly: MJCF Assembly & Simulation Bridge"
echo "===================================================="

# Check Physics output
if [ ! -f "/workspace/data/outputs/meshes/physics_metadata.json" ]; then
    echo "❌ Error: Physics output not found. Run Physics first!"
    exit 1
fi

# Run the pipeline
python /workspace/src/assembly.py \
    --config /workspace/configs/pipeline_config.json

echo "✅ Assembly complete!"
echo ""
echo "🎉 All stages complete! Your simulation-ready bundle is ready."
echo "   Check data/outputs/mjcf/assembly_metadata.json for details."
