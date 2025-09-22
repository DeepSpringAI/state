.PHONY: help setup clean install dev test lint format train-tx predict-tx infer-tx train-emb transform-emb preprocess-train preprocess-infer

# Default target
help:
	@echo "Available targets:"
	@echo "  setup           - Set up environment using uv (following project recommendations)"
	@echo "  install         - Install dependencies in editable mode using uv tool install"
	@echo "  dev             - Install development dependencies"
	@echo "  install-vectordb - Install with vector database support"
	@echo "  test            - Run tests"
	@echo "  lint            - Run linting with ruff"
	@echo "  format          - Format code with ruff"
	@echo "  clean           - Clean up cache files"
	@echo ""
	@echo "Training & Inference:"
	@echo "  train-tx        - Train State Transition model (customize variables)"
	@echo "  predict-tx      - Predict using trained ST model (customize variables)"
	@echo "  infer-tx        - Inference with ST model on new data (customize variables)"
	@echo "  train-emb       - Train State Embedding model (customize CONFIG variable)"
	@echo "  transform-emb   - Transform data using SE model (customize variables)"
	@echo "  preprocess-train - Preprocess training data (customize variables)"
	@echo "  preprocess-infer - Preprocess inference data (customize variables)"
	@echo ""
	@echo "Usage examples:"
	@echo "  make install    # Install for development"
	@echo "  uv run state    # Run state CLI"
	@echo "  make train-tx   # Train ST model with default example config"

# Set up environment using uv (following project recommendations)
setup:
	@echo "Setting up environment with uv..."
	@if command -v uv >/dev/null 2>&1; then \
		echo "uv is available. Environment setup complete!"; \
		echo "To install dependencies: make install"; \
		echo "To run state: uv run state"; \
	else \
		echo "Error: uv is not installed. Please install uv first:"; \
		echo "curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		exit 1; \
	fi

# Install dependencies from pyproject.toml (editable install for development)
install: setup
	@echo "Installing dependencies in editable mode..."
	@uv tool install -e .

# Install development dependencies
dev: install
	@echo "Installing development dependencies..."
	@uv tool install -e ".[dev]"

# Install with vector database support
install-vectordb: install
	@echo "Installing with vector database support..."
	@uv tool install -e ".[vectordb]"

# Run tests
test:
	@echo "Running tests..."
	@python -m pytest tests/ -v

# Run linting
lint:
	@echo "Running linting..."
	@ruff check src/ tests/

# Format code
format:
	@echo "Formatting code..."
	@ruff format src/ tests/

# Clean up cache files
clean:
	@echo "Cleaning up cache files..."
	@rm -rf __pycache__
	@rm -rf .pytest_cache
	@rm -rf .ruff_cache
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Note: uv manages its own environment, no .venv to clean"

# Training and Inference Tasks
# Customize these variables as needed for your specific use case

# State Transition Model Training
# Example: make train-tx TOML_CONFIG=examples/fewshot.toml OUTPUT_DIR=$HOME/state NAME=my_experiment
train-tx:
	@echo "Training State Transition model..."
	@if [ -z "$(TOML_CONFIG)" ]; then \
		echo "Using default config: examples/fewshot.toml"; \
		TOML_CONFIG="examples/fewshot.toml"; \
	fi; \
	if [ -z "$(OUTPUT_DIR)" ]; then \
		echo "Using default output dir: $HOME/state"; \
		OUTPUT_DIR="$HOME/state"; \
	fi; \
	if [ -z "$(NAME)" ]; then \
		echo "Using default name: test"; \
		NAME="test"; \
	fi; \
	uv run state tx train \
		data.kwargs.toml_config_path="$$TOML_CONFIG" \
		data.kwargs.embed_key=X_hvg \
		data.kwargs.num_workers=12 \
		data.kwargs.batch_col=batch_var \
		data.kwargs.pert_col=target_gene \
		data.kwargs.cell_type_key=cell_type \
		data.kwargs.control_pert=TARGET1 \
		training.max_steps=40000 \
		training.val_freq=100 \
		training.ckpt_every_n_steps=100 \
		training.batch_size=8 \
		training.lr=1e-4 \
		model.kwargs.cell_set_len=64 \
		model.kwargs.hidden_dim=328 \
		model=pertsets \
		wandb.tags="[test]" \
		output_dir="$$OUTPUT_DIR" \
		name="$$NAME"

# State Transition Model Prediction
# Example: make predict-tx OUTPUT_DIR=$HOME/state/test CHECKPOINT=final.ckpt
predict-tx:
	@echo "Running prediction with trained ST model..."
	@if [ -z "$(OUTPUT_DIR)" ]; then \
		echo "Error: OUTPUT_DIR is required. Example: make predict-tx OUTPUT_DIR=$HOME/state/test"; \
		exit 1; \
	fi; \
	if [ -z "$(CHECKPOINT)" ]; then \
		echo "Using default checkpoint: final.ckpt"; \
		CHECKPOINT="final.ckpt"; \
	fi; \
	uv run state tx predict --output_dir $(OUTPUT_DIR) --checkpoint $(CHECKPOINT)

# State Transition Model Inference on New Data
# Example: make infer-tx OUTPUT=$HOME/state/test MODEL_DIR=/path/to/model CHECKPOINT=/path/to/model/final.ckpt ADATA=/path/to/data.h5ad
infer-tx:
	@echo "Running inference with ST model on new data..."
	@if [ -z "$(OUTPUT)" ] || [ -z "$(MODEL_DIR)" ] || [ -z "$(CHECKPOINT)" ] || [ -z "$(ADATA)" ]; then \
		echo "Error: OUTPUT, MODEL_DIR, CHECKPOINT, and ADATA are required."; \
		echo "Example: make infer-tx OUTPUT=$HOME/state/test MODEL_DIR=/path/to/model CHECKPOINT=/path/to/model/final.ckpt ADATA=/path/to/data.h5ad"; \
		exit 1; \
	fi; \
	uv run state tx infer \
		--output $(OUTPUT) \
		--output_dir $(MODEL_DIR) \
		--checkpoint $(CHECKPOINT) \
		--adata $(ADATA) \
		--pert_col gene \
		--embed_key X_hvg

# State Embedding Model Training
# Example: make train-emb CONFIG=my_config.yaml
train-emb:
	@echo "Training State Embedding model..."
	@if [ -z "$(CONFIG)" ]; then \
		echo "Error: CONFIG is required. Example: make train-emb CONFIG=my_config.yaml"; \
		exit 1; \
	fi; \
	uv run state emb fit --conf $(CONFIG)

# State Embedding Model Transform
# Example: make transform-emb MODEL_FOLDER=/path/to/model CHECKPOINT=/path/to/model/se600m_epoch15.ckpt INPUT=/path/to/input.h5ad OUTPUT=/path/to/output.h5ad
transform-emb:
	@echo "Transforming data using State Embedding model..."
	@if [ -z "$(MODEL_FOLDER)" ] || [ -z "$(CHECKPOINT)" ] || [ -z "$(INPUT)" ] || [ -z "$(OUTPUT)" ]; then \
		echo "Error: MODEL_FOLDER, CHECKPOINT, INPUT, and OUTPUT are required."; \
		echo "Example: make transform-emb MODEL_FOLDER=/path/to/model CHECKPOINT=/path/to/model/se600m_epoch15.ckpt INPUT=/path/to/input.h5ad OUTPUT=/path/to/output.h5ad"; \
		exit 1; \
	fi; \
	uv run state emb transform \
		--model-folder $(MODEL_FOLDER) \
		--checkpoint $(CHECKPOINT) \
		--input $(INPUT) \
		--output $(OUTPUT)

# Preprocess Training Data
# Example: make preprocess-train ADATA=/path/to/raw_data.h5ad OUTPUT=/path/to/preprocessed.h5ad NUM_HVGS=2000
preprocess-train:
	@echo "Preprocessing training data..."
	@if [ -z "$(ADATA)" ] || [ -z "$(OUTPUT)" ]; then \
		echo "Error: ADATA and OUTPUT are required."; \
		echo "Example: make preprocess-train ADATA=/path/to/raw_data.h5ad OUTPUT=/path/to/preprocessed.h5ad"; \
		exit 1; \
	fi; \
	if [ -z "$(NUM_HVGS)" ]; then \
		echo "Using default num_hvgs: 2000"; \
		NUM_HVGS="2000"; \
	fi; \
	uv run state tx preprocess_train \
		--adata $(ADATA) \
		--output $(OUTPUT) \
		--num_hvgs $(NUM_HVGS)

# Preprocess Inference Data
# Example: make preprocess-infer ADATA=/path/to/real_data.h5ad OUTPUT=/path/to/control_template.h5ad CONTROL_CONDITION=DMSO PERT_COL=treatment
preprocess-infer:
	@echo "Preprocessing inference data..."
	@if [ -z "$(ADATA)" ] || [ -z "$(OUTPUT)" ] || [ -z "$(CONTROL_CONDITION)" ] || [ -z "$(PERT_COL)" ]; then \
		echo "Error: ADATA, OUTPUT, CONTROL_CONDITION, and PERT_COL are required."; \
		echo "Example: make preprocess-infer ADATA=/path/to/real_data.h5ad OUTPUT=/path/to/control_template.h5ad CONTROL_CONDITION=DMSO PERT_COL=treatment"; \
		exit 1; \
	fi; \
	if [ -z "$(SEED)" ]; then \
		echo "Using default seed: 42"; \
		SEED="42"; \
	fi; \
	uv run state tx preprocess_infer \
		--adata $(ADATA) \
		--output $(OUTPUT) \
		--control_condition $(CONTROL_CONDITION) \
		--pert_col $(PERT_COL) \
		--seed $(SEED)
