#!/bin/bash
# Quick launcher for Segmentation: Semantic Segmentation

set -e

cleanup() {
    kill "$OLLAMA_PID" 2>/dev/null || true
}

trap cleanup EXIT

echo "🚀 Starting Segmentation: Semantic Grounding & 3D Segmentation"
echo "========================================================"

# Check Reconstruction output
if [ ! -f "/workspace/data/outputs/splats/reconstruction_metadata.json" ]; then
    echo "❌ Error: Reconstruction output not found. Run Reconstruction first!"
    exit 1
fi

# Start Ollama service in background
echo "🤖 Starting Ollama service..."
ollama serve &
OLLAMA_PID=$!
sleep 5

# Check GPU
nvidia-smi || { echo "❌ Error: NVIDIA GPU not found"; exit 1; }

# Ensure SAM2 checkpoint exists (runtime fetch with fallbacks)
CHECKPOINT_DIR="/workspace/checkpoints"
CHECKPOINT_PATH="$CHECKPOINT_DIR/sam2_hiera_small.pt"

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "⬇️  Downloading SAM2 checkpoint..."
    mkdir -p "$CHECKPOINT_DIR"

    URLS=(
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2_hiera_small.pt"
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt"
        "https://huggingface.co/facebook/sam2-hiera-small/resolve/main/sam2_hiera_small.pt"
    )

    DOWNLOADED=0
    for url in "${URLS[@]}"; do
        echo "  → Trying: $url"
        if curl -fL --retry 3 --retry-delay 2 \
            -A "Mozilla/5.0" \
            -o "$CHECKPOINT_PATH" "$url"; then
            DOWNLOADED=1
            break
        fi
    done

    if [ "$DOWNLOADED" -ne 1 ]; then
        echo "❌ Failed to download SAM2 checkpoint from all known sources."
        echo "   Please manually place it at: $CHECKPOINT_PATH"
        exit 1
    fi

    echo "✅ SAM2 checkpoint ready: $CHECKPOINT_PATH"
fi

# Run the pipeline
python /workspace/src/segmentation.py \
    --config /workspace/configs/pipeline_config.json

echo "✅ Segmentation complete!"
