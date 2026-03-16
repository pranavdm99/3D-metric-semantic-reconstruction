#!/bin/bash
# Master script to run all phases sequentially

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   Physics-Ready Video Pipeline - Full Execution           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check prerequisites
if ! command -v docker &> /dev/null; then
    echo "❌ docker not found. Please install Docker."
    exit 1
fi

if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. Please install NVIDIA drivers."
    exit 1
fi

# Check input video
if [ ! -f "data/raw_video/input.mp4" ]; then
    echo "❌ No input video found at data/raw_video/input.mp4"
    echo ""
    echo "Please place your multi-view video there first:"
    echo "  cp /path/to/your/video.mp4 data/raw_video/input.mp4"
    exit 1
fi

echo "✅ Prerequisites check passed"
echo ""

# Confirm start
read -p "This will run all 4 phases sequentially (~3-4 hours). Continue? (y/N) " -n 1 -r
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
echo "✅ Reconstruction complete"

# Segmentation
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  STAGE 2/4: Semantic Grounding & 3D Segmentation"
echo "════════════════════════════════════════════════════════════"
docker compose run --rm segmentation bash /workspace/scripts/segmentation.sh
echo "✅ Segmentation complete"

# Physics
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  STAGE 3/4: Automated Physics Extraction"
echo "════════════════════════════════════════════════════════════"
docker compose run --rm physics bash /workspace/scripts/physics.sh
echo "✅ Physics complete"

# Assembly
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  STAGE 4/4: MJCF Assembly & Simulation Bridge"
echo "════════════════════════════════════════════════════════════"
docker compose run --rm assembly bash /workspace/scripts/assembly.sh
echo "✅ Assembly complete"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                 🎉 PIPELINE COMPLETE! 🎉                   ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "⏱️  Total time: ${HOURS}h ${MINUTES}m"
echo ""
echo "📦 Your simulation-ready bundle:"
echo "   • MJCF World:    data/outputs/mjcf/world.xml"
echo "   • 3D Splat:      data/outputs/splats/*.ply"
echo "   • Physics Meshes: data/outputs/meshes/"
echo "   • Assembly Info:  data/outputs/mjcf/assembly_metadata.json"
echo ""
echo "🚀 Quick test:"
echo "   python -c \"import mujoco; m=mujoco.MjModel.from_xml_path('data/outputs/mjcf/world.xml'); print('Success!')\""
echo ""
