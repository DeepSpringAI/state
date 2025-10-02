

## Getting started
Predicting cellular responses to perturbation across diverse contexts with State

## Running on the server
if you are running on the server, simply follow below:

```bash
git clone --branch experimental_setup git@github.com:DeepSpringAI/state.git
cd state
```
### Installation 
```
chmod +x run.sh
make setup 
```
### training
```
make train
```



### Installation (not on the server)
### if *uv* is not already installed, use the command below to install it:

```
pip install uv
```

### clone the repo and installing the package 

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
```
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
```
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


## Licenses
State code is [licensed](LICENSE) under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0).

The model weights and output are licensed under the [Arc Research Institute State Model Non-Commercial License](MODEL_LICENSE.md) and subject to the [Arc Research Institute State Model Acceptable Use Policy](MODEL_ACCEPTABLE_USE_POLICY.md).

Any publication that uses this source code or model parameters should cite the State [paper](https://arcinstitute.org/manuscripts/State).
