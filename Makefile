CONFIG        ?= $(CURDIR)/examples/mixed.toml
OUTPUT        ?= $(CURDIR)/mixed_for_competition
NAME          ?= unified_model_mixed_for_the_meeting
BATCH         ?= 64
STEPS         ?= 5000
LR            ?= 1e-4
NUM_WORKERS   ?= 32
EMBED_KEY     ?= X_hvg
BATCH_COL     ?= batch_var
PERT_COL      ?= target_gene
CELL_TYPE_KEY ?= cell_type
CONTROL_PERT  ?= TARGET1
MODEL         ?= state

.PHONY: help train setup setup-cpu clean docker-build docker-run docker-train
.DEFAULT_GOAL := help

help:
	@echo "Targets:"
	@echo "  make train        - run training"
	@echo "  make setup        - prepare environment with CUDA support"
	@echo "  make setup-cpu    - prepare environment with CPU-only PyTorch"
	@echo "  make clean        - remove local venv and cache"
	@echo "  make docker-build - build Docker image with all dependencies"
	@echo "  make docker-run   - run Docker container interactively"
	@echo "  make docker-train - run training in Docker container"

train:
	./run.sh tx train \
	  data.kwargs.toml_config_path="$(CONFIG)" \
	  data.kwargs.embed_key=$(EMBED_KEY) \
	  data.kwargs.num_workers=$(NUM_WORKERS) \
	  data.kwargs.batch_col=$(BATCH_COL) \
	  data.kwargs.pert_col=$(PERT_COL) \
	  data.kwargs.cell_type_key=$(CELL_TYPE_KEY) \
	  data.kwargs.control_pert=$(CONTROL_PERT) \
	  training.max_steps=$(STEPS) \
	  training.batch_size=$(BATCH) \
	  training.lr=$(LR) \
	  model=$(MODEL) \
	  output_dir="$(OUTPUT)" \
	  name="$(NAME)"

setup:
	@if ! command -v uv >/dev/null 2>&1; then \
	  echo "[INFO] uv not found, installing via pip"; \
	  pip install uv; \
	else \
	  echo "[INFO] uv is already installed"; \
	fi
	@echo "[INFO] Creating virtual environment and installing dependencies"
	uv venv --clear
	@echo "[INFO] Installing this repo in development mode"
	uv pip install -e . --index-strategy unsafe-best-match
	@echo "[INFO] Installing CUDA-compatible PyTorch"
	uv pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121 --index-strategy unsafe-best-match

# setup-cpu:
# 	@if ! command -v uv >/dev/null 2>&1; then \
# 	  echo "[INFO] uv not found, installing via pip"; \
# 	  pip install uv; \
# 	else \
# 	  echo "[INFO] uv is already installed"; \
# 	fi
# 	@echo "[INFO] Creating virtual environment and installing dependencies"
# 	uv venv --clear
# 	@echo "[INFO] Installing this repo in development mode"
# 	uv pip install -e . --index-strategy unsafe-best-match
# 	@echo "[INFO] Installing CPU-only PyTorch"
# 	uv pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
# =======
# 	@echo "[INFO] Installing tqdm via uv"
# 	uv pip install tqdm

# >>>>>>> 10cba40 (data_update)

clean:
	rm -rf .venv __pycache__ .pytest_cache .ruff_cache

docker-build:
	@echo "[INFO] Building Docker image with all dependencies pre-installed"
	./build-docker.sh

docker-run:
	@echo "[INFO] Running Docker container interactively"
	docker run -it --rm --gpus all \
	  -v $(CURDIR):/workspace \
	  -w /workspace \
	  state-ml:latest bash

docker-train:
	@echo "[INFO] Running training in Docker container"
	docker run -it --rm --gpus all \
	  -v $(CURDIR):/workspace \
	  -w /workspace \
	  state-ml:latest \
	  make train CONFIG="$(CONFIG)" OUTPUT="$(OUTPUT)" NAME="$(NAME)"