# Predicting cellular responses to perturbation across diverse contexts with State

> Train State transition models or pretrain State embedding models. See the State [paper](https://www.biorxiv.org/content/10.1101/2025.06.26.661135v2).
> 
> See the [Google Colab](https://colab.research.google.com/drive/1QKOtYP7bMpdgDJEipDxaJqOchv7oQ-_l) to train STATE for the [Virtual Cell Challenge](https://virtualcellchallenge.org/).

## Associated repositories

- Model evaluation framework: [cell-eval](https://github.com/ArcInstitute/cell-eval)
- Dataloaders and preprocessing: [cell-load](https://github.com/ArcInstitute/cell-load)

## Getting started

- Train an ST model for genetic perturbation prediction using the Replogle-Nadig dataset: [Colab](https://colab.research.google.com/drive/1Ih-KtTEsPqDQnjTh6etVv_f-gRAA86ZN)
- Perform inference using an ST model trained on Tahoe-100M: [Colab](https://colab.research.google.com/drive/1bq5v7hixnM-tZHwNdgPiuuDo6kuiwLKJ)
- Embed and annotate a new dataset using SE: [Colab](https://colab.research.google.com/drive/1uJinTJLSesJeot0mP254fQpSxGuDEsZt)
- Train STATE for the Virtual Cell Challenge: [Colab](https://colab.research.google.com/drive/1QKOtYP7bMpdgDJEipDxaJqOchv7oQ-_l)

## Installation

### Installation from PyPI

This package is distributed via [`uv`](https://docs.astral.sh/uv).

```bash
uv tool install arc-state
```

### Installation

```bash
git clone --branch experimental_setup git@github.com:DeepSpringAI/state.git
cd state
uv tool install -e .
```

## CLI Usage


```state --help```

## set the dataset path
set the path to the *examples* directory if different:
```
export DATASET_PATH='examples'

```

## State Transition Model (ST)

To train with a mixed experiment (including both zeroshot and fewshot)

state tx train \
  data.kwargs.toml_config_path="$(pwd)/examples/mixed.toml" \
  data.kwargs.embed_key=X_hvg \
  data.kwargs.num_workers=32 \
  data.kwargs.batch_col=batch_var \
  data.kwargs.pert_col=target_gene \
  data.kwargs.cell_type_key=cell_type \
  data.kwargs.control_pert=TARGET1 \
  training.max_steps=5000 \
  training.batch_size=64 \
  training.lr=1e-4 \
  model=state \
  output_dir="./mixed_for_competition" \
  name="unified_model_mixed_for_the_meeting"

The cell lines and perturbations specified in the TOML should match the values appearing in the
`data.kwargs.cell_type_key` and `data.kwargs.pert_col` used above. To evaluate STATE on the specified task,
you can use the `tx predict` command:


```
 state tx predict \                                              
  --output-dir ./mixed_for_competition/unified_model_mixed_for_the_meeting/ \
  --checkpoint final.ckpt
```


It will look in the `output_dir` above, for a `checkpoints` folder.

If you instead want to use a trained checkpoint for inference (e.g. on data not specified)
in the TOML file:


```bash
state tx infer \                                                                                          
  --model-dir ./mixed_for_competition/unified_model_mixed_for_the_meeting/ \
  --adata competition_support_set/competition_val_template.h5ad \
  --output competition/prediction_new.h5ad \
  --pert-col target_gene
```

### Data Preprocessing

State provides two preprocessing commands to prepare data for training and inference:

#### Training Data Preprocessing

Use `preprocess_train` to normalize, log-transform, and select highly variable genes from your training data:

```bash
state tx preprocess_train \
  --adata /path/to/raw_data.h5ad \
  --output /path/to/preprocessed_training_data.h5ad \
  --num_hvgs 2000
```

This command:
- Normalizes total counts per cell (`sc.pp.normalize_total`)
- Applies log1p transformation (`sc.pp.log1p`) 
- Identifies highly variable genes (`sc.pp.highly_variable_genes`)
- Stores the HVG expression matrix in `.obsm['X_hvg']`

#### Inference Data Preprocessing

Use `preprocess_infer` to create a "control template" for model inference:

```bash
state tx preprocess_infer \
  --adata /path/to/real_data.h5ad \
  --output /path/to/control_template.h5ad \
  --control_condition "DMSO" \
  --pert_col "treatment" \
  --seed 42
```


#### converting to the competition template

run h5da_convertor.py to get the **.vcc** file:

```
python h5da_convertor.py

```



This command replaces all perturbed cells with control cell expression while preserving perturbation annotations. The resulting dataset serves as a baseline where `state_transition(control_template) ≈ original_data`, allowing you to evaluate how well the model reconstructs perturbation effects from control states.

## TOML Configuration Files

State experiments are configured using TOML files that define datasets, training splits, and evaluation scenarios. The configuration system supports both **zeroshot** (unseen cell types) and **fewshot** (limited perturbation examples) evaluation paradigms.

### Configuration Structure

#### Required Sections

**`[datasets]`** - Maps dataset names to their file system paths
```toml
[datasets]
replogle = "/path/to/replogle/dataset/"
# YOU CAN ADD MORE
```

**`[training]`** - Specifies which datasets participate in training
```toml
[training]
replogle = "train"  # Include all replogle data in training (unless overridden below)
```

#### Optional Evaluation Sections

**`[zeroshot]`** - Reserves entire cell types for validation/testing
```toml
[zeroshot]
"replogle.jurkat" = "test"     # All jurkat cells go to test set
"replogle.k562" = "val"        # All k562 cells go to validation set
```

**`[fewshot]`** - Specifies perturbation-level splits within cell types
```toml
[fewshot]
[fewshot."replogle.rpe1"]      # Configure splits for rpe1 cell type
val = ["AARS", "TUFM"]         # These perturbations go to validation
test = ["NUP107", "RPUSD4"]    # These perturbations go to test
# Note: All other perturbations in rpe1 automatically go to training

```

### Configuration Examples

#### Example 1: Pure Zeroshot Evaluation
```toml
# Evaluate generalization to completely unseen cell types
[datasets]
replogle = "/data/replogle/"

[training]
replogle = "train"

[zeroshot]
"replogle.jurkat" = "test"     # Hold out entire jurkat cell line
"replogle.rpe1" = "val"        # Hold out entire rpe1 cell line

[fewshot]
# Empty - no perturbation-level splits
```

#### Example 2: Pure Fewshot Evaluation
```toml
# Evaluate with limited examples of specific perturbations
[datasets]
replogle = "/data/replogle/"

[training]
replogle = "train"

[zeroshot]
# Empty - all cell types participate in training

[fewshot]
[fewshot."replogle.k562"]
val = ["AARS"]                 # Limited AARS examples for validation
test = ["NUP107", "RPUSD4"]    # Limited examples of these genes for testing

[fewshot."replogle.jurkat"]
val = ["TUFM"]
test = ["MYC", "TP53"]
```

#### Example 3: Mixed Evaluation Strategy
```toml
# Combine both zeroshot and fewshot evaluation
[datasets]
replogle = "/data/replogle/"

[training]
replogle = "train"

[zeroshot]
"replogle.jurkat" = "test"        # Zeroshot: unseen cell type

[fewshot]
[fewshot."replogle.k562"]      # Fewshot: limited perturbation examples
val = ["STAT1"]
test = ["MYC", "TP53"]
```

### Important Notes

- **Automatic training assignment**: Any cell type not mentioned in `[zeroshot]` automatically participates in training, with perturbations not listed in `[fewshot]` going to the training set
- **Overlapping splits**: Perturbations can appear in both validation and test sets within fewshot configurations
- **Dataset naming**: Use the format `"dataset_name.cell_type"` when specifying cell types in zeroshot and fewshot sections
- **Path requirements**: Dataset paths should point to directories containing h5ad files
- **Control perturbations**: Ensure your control condition (specified via `control_pert` parameter) is available across all splits

### Validation

The configuration system will validate that:
- All referenced datasets exist at the specified paths
- Cell types mentioned in zeroshot/fewshot sections exist in the datasets
- Perturbations listed in fewshot sections are present in the corresponding cell types
- No conflicts exist between zeroshot and fewshot assignments for the same cell type


## State Embedding Model (SE)

After following the same installation commands above:

```bash
state emb fit --conf ${CONFIG}
```

To run inference with a trained State checkpoint, e.g., the State trained to 16 epochs:

```bash
state emb transform \
  --model-folder /large_storage/ctc/userspace/aadduri/SE-600M \
  --checkpoint /large_storage/ctc/userspace/aadduri/SE-600M/se600m_epoch15.ckpt \
  --input /large_storage/ctc/datasets/replogle/rpe1_raw_singlecell_01.h5ad \
  --output /home/aadduri/vci_pretrain/test_output.h5ad
```

Notes on the h5ad file format:
 - CSR matrix format is required
 - `gene_name` is required in the `var` dataframe

### Vector Database

Install the optional dependencies:

```bash
uv tool install ".[vectordb]"
```

If working off a previous installation, you may need to run:

```bash
uv sync --extra vectordb
```

#### Build the vector database

```bash
state emb transform \
  --model-folder /large_storage/ctc/userspace/aadduri/SE-600M \
  --input /large_storage/ctc/public/scBasecamp/GeneFull_Ex50pAS/GeneFull_Ex50pAS/Homo_sapiens/SRX27532045.h5ad \
  --lancedb tmp/state_embeddings.lancedb \
  --gene-column gene_symbols
```

Running this command multiple times with the same lancedb appends the new data to the provided database.

#### Query the database

Obtain the embeddings:

```bash
state emb transform \
  --model-folder /large_storage/ctc/userspace/aadduri/SE-600M \
  --input /large_storage/ctc/public/scBasecamp/GeneFull_Ex50pAS/GeneFull_Ex50pAS/Homo_sapiens/SRX27532046.h5ad \
  --output tmp/SRX27532046.h5ad \
  --gene-column gene_symbols
```

Query the database with the embeddings:

```bash
state emb query \
  --lancedb tmp/state_embeddings.lancedb \
  --input tmp/SRX27532046.h5ad \
  --output tmp/similar_cells.csv \
  --k 3

# Singularity

Containerization for STATE is available via the `singularity.def` file.

Build the container:

```bash
singularity build state.sif singularity.def
```

Run the container:

```bash
singularity run state.sif --help
```

Example run of `state emb transform`:

```bash
singularity run --nv -B /large_storage:/large_storage \
  state.sif emb transform \
    --model-folder /large_storage/ctc/userspace/aadduri/SE-600M \
    --checkpoint /large_storage/ctc/userspace/aadduri/SE-600M/se600m_epoch15.ckpt \
    --input /large_storage/ctc/datasets/replogle/rpe1_raw_singlecell_01.h5ad \
    --output test_output.h5ad
```



## Licenses
State code is [licensed](LICENSE) under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0).

The model weights and output are licensed under the [Arc Research Institute State Model Non-Commercial License](MODEL_LICENSE.md) and subject to the [Arc Research Institute State Model Acceptable Use Policy](MODEL_ACCEPTABLE_USE_POLICY.md).

Any publication that uses this source code or model parameters should cite the State [paper](https://arcinstitute.org/manuscripts/State).
