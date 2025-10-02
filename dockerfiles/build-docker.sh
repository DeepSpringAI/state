#!/bin/bash
set -euo pipefail

# Docker build script for State ML model
# This script builds a comprehensive Docker image with all dependencies pre-installed

# Configuration
IMAGE_NAME="state-ml"
TAG="${1:-latest}"
FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"

echo "🚀 Building State ML Docker image: ${FULL_IMAGE_NAME}"
echo "📦 This image includes all dependencies to minimize startup time"

# Build the Docker image
docker build \
    --tag "${FULL_IMAGE_NAME}" \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    --progress=plain \
    .

echo "✅ Docker image built successfully: ${FULL_IMAGE_NAME}"
echo ""
echo "🔧 Usage examples:"
echo "  # Run interactive shell:"
echo "  docker run -it --rm ${FULL_IMAGE_NAME} bash"
echo ""
echo "  # Run with GPU support:"
echo "  docker run -it --rm --gpus all ${FULL_IMAGE_NAME} bash"
echo ""
echo "  # Run training:"
echo "  docker run -it --rm --gpus all -v \$(pwd)/data:/data ${FULL_IMAGE_NAME} make train"
echo ""
echo "  # Mount current directory and run:"
echo "  docker run -it --rm -v \$(pwd):/workspace -w /workspace ${FULL_IMAGE_NAME} state --help"

# Optional: Show image size
echo ""
echo "📊 Image information:"
docker images "${FULL_IMAGE_NAME}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"


