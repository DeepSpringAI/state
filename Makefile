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

.PHONY: help train setup clean
.DEFAULT_GOAL := help

help:
	@echo "Targets:"
	@echo "  make train    - run training"
	@echo "  make setup    - prepare environment"
	@echo "  make clean    - remove local venv and cache"

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
	@if command -v uv >/dev/null 2>&1; then \
	  echo "[INFO] uv detected; syncing environment"; \
	  uv sync; \
	else \
	  echo "[INFO] uv not found; creating .venv and installing requirements.txt"; \
	  if [ ! -d ".venv" ]; then python -m venv .venv; fi; \
	  . .venv/bin/activate && pip install uv; \
	fi

clean:
	rm -rf .venv __pycache__ .pytest_cache .ruff_cache