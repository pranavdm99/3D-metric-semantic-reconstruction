# Multi-stage Dockerfile optimized for RTX 4060 (8GB VRAM)
# Base: CUDA 12.1 + PyTorch 2.1 with memory-efficient settings

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS base

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set CUDA memory management for 8GB GPU
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /workspace

# Install PyTorch with CUDA 12.1 support
RUN pip3 install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu121

# ===== RECONSTRUCTION: Neural Reconstruction =====
FROM base AS reconstruction

RUN pip3 install --no-cache-dir \
    opencv-python \
    numpy \
    scipy \
    plyfile \
    tqdm \
    tensorboard \
    ninja \
    einops

# Install nerfstudio (includes 3DGS support)
RUN pip3 install --no-cache-dir nerfstudio

# Clone and install SuGaR (Surface-Aligned Gaussian Splatting)
RUN git clone https://github.com/Anttwo/SuGaR.git /workspace/SuGaR && \
    cd /workspace/SuGaR && \
    pip3 install --no-cache-dir -e .

# Install COLMAP for structure-from-motion
RUN apt-get update && apt-get install -y colmap

# ===== SEGMENTATION: Semantic Segmentation =====
FROM base AS segmentation

RUN pip3 install --no-cache-dir \
    opencv-python \
    numpy \
    pillow \
    transformers \
    segment-anything \
    ollama

# Install SAM 2
RUN pip3 install --no-cache-dir git+https://github.com/facebookresearch/segment-anything-2.git

# ===== PHYSICS: Physics Extraction =====
FROM base AS physics

RUN pip3 install --no-cache-dir \
    trimesh \
    numpy-stl \
    scikit-image \
    open3d

# Install CoACD (Convex Decomposition)
RUN git clone https://github.com/SarahWeiii/CoACD.git /workspace/CoACD && \
    cd /workspace/CoACD && \
    pip3 install --no-cache-dir -e .

# ===== ASSEMBLY: MJCF Assembly =====
FROM base AS assembly

RUN pip3 install --no-cache-dir \
    mujoco \
    dm_control \
    lxml \
    gsplat

# ===== FULL ENVIRONMENT (All phases combined) =====
FROM base AS full

# Install all dependencies
RUN pip3 install --no-cache-dir \
    opencv-python \
    numpy \
    scipy \
    plyfile \
    tqdm \
    tensorboard \
    ninja \
    einops \
    pillow \
    transformers \
    trimesh \
    numpy-stl \
    scikit-image \
    open3d \
    mujoco \
    dm_control \
    lxml \
    gsplat

# Install nerfstudio and SAM 2
RUN pip3 install --no-cache-dir nerfstudio && \
    pip3 install --no-cache-dir git+https://github.com/facebookresearch/segment-anything-2.git

# Install SuGaR
RUN git clone https://github.com/Anttwo/SuGaR.git /workspace/SuGaR && \
    cd /workspace/SuGaR && \
    pip3 install --no-cache-dir -e .

# Install CoACD
RUN git clone https://github.com/SarahWeiii/CoACD.git /workspace/CoACD && \
    cd /workspace/CoACD && \
    pip3 install --no-cache-dir -e .

# Install COLMAP
RUN apt-get update && apt-get install -y colmap && \
    rm -rf /var/lib/apt/lists/*

# Copy project code
COPY . /workspace/project

WORKDIR /workspace/project

CMD ["/bin/bash"]
