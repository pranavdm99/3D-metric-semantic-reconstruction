.PHONY: help build-base build-all run-all clean clean-outputs test sim reconstruction segmentation physics assembly

help:
	@echo "Physics-Ready Video Pipeline - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  build-base     - Build base Docker image"
	@echo "  build-all      - Build all Stage Docker images"
	@echo "  reconstruction - Run Stage 1: Neural Reconstruction"
	@echo "  segmentation   - Run Stage 2: Semantic Segmentation"
	@echo "  physics        - Run Stage 3: Automated Physics Extraction"
	@echo "  assembly       - Run Stage 4: MJCF Assembly"
	@echo "  run-all        - Run all stages sequentially"
	@echo "  test           - Test MuJoCo output"
	@echo "  sim            - Launch MuJoCo interactive viewer (needs display)"
	@echo "  clean-outputs  - Remove generated outputs (keeps raw video)"
	@echo "  clean          - Remove all Docker images and outputs"
	@echo "  check-gpu      - Check GPU availability"
	@echo ""

build-base:
	@echo "Building base Docker image..."
	docker compose --profile build build base

# Build all Stage images
build-all: build-base
	@echo "Building all Stage images..."
	docker compose build

run-all:
	@echo "Running full pipeline..."
	./run_all.sh

reconstruction:
	@echo "Running Stage 1: Neural Reconstruction..."
	docker compose run --rm reconstruction bash /workspace/scripts/reconstruction.sh

segmentation:
	@echo "Running Stage 2: Semantic Segmentation..."
	docker compose run --rm segmentation bash /workspace/scripts/segmentation.sh

physics:
	@echo "Running Stage 3: Automated Physics Extraction..."
	docker compose run --rm physics bash /workspace/scripts/physics.sh

assembly:
	@echo "Running Stage 4: MJCF Assembly..."
	docker compose run --rm assembly bash /workspace/scripts/assembly.sh

test:
	@echo "Testing MuJoCo world file..."
	@docker compose run --rm assembly bash -c "cd /workspace/data/outputs/mjcf && python -c \"import mujoco; m=mujoco.MjModel.from_xml_path('world.xml'); print('✅ MuJoCo world loads successfully!')\""

sim:
	@echo "Launching MuJoCo viewer (close window to exit)..."
	@docker compose run --rm -e DISPLAY=$${DISPLAY} -e DATA_ROOT=/workspace/data -v /tmp/.X11-unix:/tmp/.X11-unix assembly python /workspace/scripts/view_sim.py

check-gpu:
	@echo "Checking GPU availability..."
	@nvidia-smi || echo "❌ NVIDIA GPU not found"

clean-outputs:
	@echo "Cleaning outputs..."
	rm -rf data/outputs/splats/* data/outputs/masks/* data/outputs/meshes/* data/outputs/mjcf/*
	rm -rf data/raw_video/frames
	@echo "✅ Outputs cleaned (raw video preserved)"

clean: clean-outputs
	@echo "Removing Docker images..."
	docker compose down --rmi all
	@echo "✅ Full clean complete"

# Run a bash shell in a Stage container
shell-reconstruction:
	docker compose run --rm reconstruction bash

shell-segmentation:
	docker compose run --rm segmentation bash

shell-physics:
	docker compose run --rm physics bash

shell-assembly:
	docker compose run --rm assembly bash
