.PHONY: help build-base build-all run-all clean clean-outputs reconstruction segmentation

help:
	@echo "Semantic Scene Reconstruction Pipeline - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  build-base     - Build base Docker image"
	@echo "  build-reconstruction - Build Neural Reconstruction image"
	@echo "  build-segmentation   - Build Semantic Segmentation image"
	@echo "  reconstruction - Run Neural Reconstruction"
	@echo "  segmentation   - Run Semantic Segmentation"
	@echo "  run-all        - Run all stages sequentially"
	@echo "  clean-outputs  - Remove generated outputs (keeps raw video)"
	@echo "  clean          - Remove all Docker images and outputs"
	@echo "  check-gpu      - Check GPU availability"
	@echo ""

build-base:
	@echo "Building base Docker image..."
	docker compose --profile build build base

build-reconstruction:
	@echo "Building Reconstruction image..."
	docker compose build reconstruction

build-segmentation:
	@echo "Building Segmentation image..."
	docker compose build segmentation

run-all:
	@echo "Running full pipeline..."
	./run_all.sh

reconstruction:
	@echo "Running Neural Reconstruction..."
	docker compose run --rm reconstruction bash /workspace/scripts/reconstruction.sh

segmentation:
	@echo "Running Semantic Segmentation..."
	docker compose run --rm segmentation bash /workspace/scripts/segmentation.sh

check-gpu:
	@echo "Checking GPU availability..."
	@nvidia-smi || echo "❌ NVIDIA GPU not found"

clean-outputs:
	@echo "Cleaning outputs..."
	rm -rf data/outputs/splats/* data/outputs/masks/* data/outputs/colmap/* data/raw_video/frames
	@echo "✅ Outputs cleaned"

clean: clean-outputs
	@echo "Removing Docker images..."
	docker compose down --rmi all
	@echo "✅ Full clean complete"

# Run a bash shell in a Stage container
shell-reconstruction:
	docker compose run --rm reconstruction bash

shell-segmentation:
	docker compose run --rm segmentation bash
