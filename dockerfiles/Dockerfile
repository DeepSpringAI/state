# Multi-stage build for State ML model
# Based on CUDA base image to support GPU training/inference
FROM docker.artifactory.dev.intapp.com/ai/cuda:12.1.1-base-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/root/.local/bin:$PATH"
ENV UV_CACHE_DIR=/tmp/uv-cache

# Install system dependencies in a single layer
RUN apt-get update && apt-get install -y \
    # Essential build tools
    curl \
    wget \
    git \
    build-essential \
    cmake \
    pkg-config \
    # Python development
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    # Scientific computing dependencies
    libopenblas-dev \
    liblapack-dev \
    libhdf5-dev \
    libffi-dev \
    libssl-dev \
    # Graphics and visualization
    libcairo2-dev \
    libgirepository1.0-dev \
    # Additional utilities
    vim \
    htop \
    tmux \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create symlinks for python3.11
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# Install uv (fast Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Set working directory
WORKDIR /home/agent/workspace/state

# Copy dependency files first for better caching
COPY pyproject.toml ./
COPY README.md ./
COPY run.sh ./
COPY Makefile ./

# Make run.sh executable
RUN chmod +x run.sh

# Pre-install Python dependencies using uv
# This creates a virtual environment and installs all dependencies
RUN uv venv .venv && \
    uv pip install --no-cache-dir \
    # Core ML/Scientific libraries (install first for better caching)
    torch>=2.7.0 \
    numpy>=2.2.6 \
    scipy>=1.15.0 \
    pandas>=2.2.3 \
    scikit-learn>=1.6.1 \
    # Single-cell analysis libraries
    anndata>=0.11.4 \
    scanpy>=1.11.2 \
    # Deep learning and transformers
    transformers>=4.52.3 \
    peft>=0.11.0 \
    geomloss>=0.2.6 \
    # Visualization and utilities
    seaborn>=0.13.2 \
    tqdm>=4.67.1 \
    pyyaml>=6.0.2 \
    # ML experiment tracking
    wandb>=0.19.11 \
    hydra-core>=1.3.2 \
    # Custom packages
    cell-load>=0.8.3 \
    cell-eval>=0.5.22 \
    # Development tools
    ipykernel>=6.30.1 \
    ruff>=0.11.11 \
    vulture>=2.14 \
    ipython>=8.37.0 \
    # Optional dependencies
    lancedb>=0.24.0

# Activate virtual environment by default
ENV VIRTUAL_ENV=/home/agent/workspace/state/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy the rest of the source code
COPY . .

# Install the state package in development mode
RUN uv pip install -e .

# Clean up uv cache to reduce image size
RUN rm -rf /tmp/uv-cache

# Create necessary directories
RUN mkdir -p examples mixed_for_competition

# Set proper permissions
RUN chmod -R 755 .

# Expose any ports if needed (adjust as necessary)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import state; print('State package loaded successfully')" || exit 1

# Default command - can be overridden
CMD ["bash"]

# Labels for metadata
LABEL maintainer="AI Team"
LABEL description="State ML model for cellular perturbation prediction with pre-installed dependencies"
LABEL version="1.0"
LABEL python.version="3.11"
LABEL cuda.version="12.1.1"
