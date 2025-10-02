# Docker Setup for State ML Model

This document describes the Docker setup for the State ML model, designed to minimize startup time by pre-installing all necessary dependencies.

## Overview

The Docker setup includes:
- **Dockerfile**: Comprehensive image with all dependencies pre-installed
- **build-docker.sh**: Build script for easy image creation
- **.dockerignore**: Optimized build context
- **Makefile targets**: Convenient Docker commands

## Quick Start

### 1. Build the Docker Image

```bash
# Build with default tag (latest)
make docker-build

# Or build with custom tag
./build-docker.sh v1.0
```

### 2. Run Interactively

```bash
# Run with GPU support
make docker-run

# Or run without make
docker run -it --rm --gpus all -v $(pwd):/workspace -w /workspace state-ml:latest bash
```

### 3. Run Training

```bash
# Run training in Docker
make docker-train

# With custom parameters
make docker-train CONFIG=examples/mixed.toml OUTPUT=./results NAME=my_experiment
```

## Docker Image Features

### Pre-installed Dependencies

The Docker image includes all dependencies from `pyproject.toml`:

**Core ML Libraries:**
- PyTorch 2.7.0+ with CUDA support
- NumPy, SciPy, Pandas
- Scikit-learn

**Single-cell Analysis:**
- AnnData 0.11.4+
- Scanpy 1.11.2+
- Cell-load, Cell-eval

**Deep Learning:**
- Transformers 4.52.3+
- PEFT 0.11.0+
- Geomloss 0.2.6+

**Utilities:**
- Weights & Biases (wandb)
- Hydra-core
- Seaborn, tqdm, PyYAML

**Development Tools:**
- Ruff, Vulture, IPython
- Jupyter kernel support

### System Dependencies

- CUDA 12.1.1 base image
- Python 3.11
- Build tools (gcc, cmake, etc.)
- Scientific computing libraries (BLAS, LAPACK, HDF5)
- UV package manager

## Usage Examples

### Basic Commands

```bash
# Check State CLI
docker run --rm state-ml:latest state --help

# Run preprocessing
docker run --rm -v $(pwd)/data:/data state-ml:latest \
  state tx preprocess_train --adata /data/input.h5ad --output /data/processed.h5ad

# Run inference
docker run --rm --gpus all -v $(pwd):/workspace state-ml:latest \
  state tx infer --model-dir ./model --adata data.h5ad --output predictions.h5ad
```

### Training Workflows

```bash
# Full training pipeline
docker run -it --rm --gpus all \
  -v $(pwd):/workspace \
  -w /workspace \
  state-ml:latest \
  bash -c "
    make setup &&
    make train CONFIG=examples/mixed.toml
  "
```

### Development

```bash
# Mount source code for development
docker run -it --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  state-ml:latest \
  bash

# Inside container:
# - Code changes are reflected immediately
# - All dependencies are pre-installed
# - No need to run 'make setup'
```

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make docker-build` | Build Docker image with all dependencies |
| `make docker-run` | Run interactive container with GPU support |
| `make docker-train` | Run training in Docker container |

## Performance Benefits

### Startup Time Comparison

**Without Docker (cold start):**
```bash
time make setup  # ~5-10 minutes
time make train  # Training time + setup overhead
```

**With Docker (pre-built image):**
```bash
time docker run state-ml:latest make train  # Training time only (~30s startup)
```

### Image Optimization

- **Multi-stage build**: Optimized for size and caching
- **Layer caching**: Dependencies cached separately from source code
- **UV package manager**: Faster than pip for dependency resolution
- **.dockerignore**: Excludes unnecessary files from build context

## Troubleshooting

### GPU Support

Ensure NVIDIA Docker runtime is installed:
```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

### Memory Issues

For large models, increase Docker memory limits:
```bash
# Run with memory limit
docker run --rm --gpus all --memory=32g state-ml:latest make train
```

### Build Issues

If build fails, try:
```bash
# Clean build (no cache)
docker build --no-cache -t state-ml:latest .

# Check disk space
docker system df
docker system prune  # Clean up if needed
```

## Advanced Configuration

### Custom Base Image

To use a different base image, modify the Dockerfile:
```dockerfile
FROM your-custom-cuda-image:tag
```

### Additional Dependencies

Add to the `uv pip install` section in Dockerfile:
```dockerfile
RUN uv pip install --no-cache-dir \
    your-additional-package>=1.0.0
```

### Environment Variables

Set environment variables for training:
```bash
docker run --rm --gpus all \
  -e WANDB_API_KEY=your_key \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  state-ml:latest make train
```

## Integration with Kubernetes

For deployment in Kubernetes (like the current ArgoCD setup):

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: state-training
spec:
  template:
    spec:
      containers:
      - name: state
        image: state-ml:latest
        command: ["make", "train"]
        resources:
          limits:
            nvidia.com/gpu: 1
```

This eliminates the need for the startup installation commands in the current ArgoCD configuration.


