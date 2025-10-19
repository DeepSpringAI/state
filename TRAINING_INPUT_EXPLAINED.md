# Training Input File Explained

## 🎯 **The Training Input File**

**File:** `examples/random.h5ad`

This is your **single training input file** that contains all the data!

---

## 📂 **File Structure**

```
/home/agent/workspace/state_modelling/state/
├── Makefile                    ← Training commands
├── examples/
│   ├── mixed.toml             ← Configuration (points to data)
│   └── random.h5ad            ← 🔥 YOUR TRAINING INPUT FILE 🔥
└── ...
```

---

## 📊 **What's Inside `random.h5ad`?**

### **File Format:** AnnData (H5AD)
- Standard format for single-cell data
- Stores cells × genes matrix + metadata

### **Contents:**

```
examples/random.h5ad
├── 10,000 cells (rows)
├── 50 genes total (columns)
├── adata.X                    - Raw gene expression
├── adata.obsm['X_hvg']        - Highly variable genes (11 genes) ← USED FOR TRAINING
└── adata.obs (metadata)       - Cell information
    ├── target_gene            - Perturbation label (TARGET1-5)
    ├── cell_type              - Cell type (CT1-5)
    └── batch_var              - Batch ID (1)
```

### **Data Breakdown:**

```
Perturbations in random.h5ad:
├── TARGET1 (Control):   2,077 cells  ← Baseline/unperturbed
├── TARGET2 (Perturbed): 1,951 cells  ← Intervention #1
├── TARGET3 (Perturbed): 2,017 cells  ← Intervention #2
├── TARGET4 (Perturbed): 1,923 cells  ← Intervention #3
└── TARGET5 (Perturbed): 2,032 cells  ← Intervention #4

Cell Types:
├── CT1: 1,955 cells
├── CT2: 2,010 cells
├── CT3: 2,015 cells
├── CT4: 1,997 cells
└── CT5: 2,023 cells
```

---

## 🔧 **How Training Uses This File**

### **Step 1: Configuration (`examples/mixed.toml`)**

```toml
[datasets]
example = "./examples"  ← Points to examples/ directory

[training]
example = "train"  ← Use "example" dataset for training

[zeroshot]
"example.CT3" = "test"  ← Hold out CT3 for testing

[fewshot."example.CT4"]
val = ["TARGET3"]                ← Validate on TARGET3 in CT4
test = ["TARGET4", "TARGET5"]    ← Test on TARGET4/5 in CT4
```

### **Step 2: Data Loading**

The data loader automatically:
1. **Scans** `./examples/` directory
2. **Finds** `random.h5ad` file
3. **Loads** all cells and metadata
4. **Splits** data based on configuration:
   - Training: Most cells
   - Validation: CT4 with TARGET3
   - Test: CT3 (all perturbations) + CT4 with TARGET4/5

### **Step 3: Training Process**

```python
For each training batch:
    1. Sample perturbed cells from random.h5ad
    2. For each perturbed cell:
       - Get its gene expression from adata.obsm['X_hvg']
       - Get its perturbation label from adata.obs['target_gene']
       - Sample a control cell (TARGET1) with matching cell_type
       - Get control's gene expression
    3. Create batch:
       - ctrl_cell_emb:  [batch_size, 11 genes]
       - pert_cell_emb:  [batch_size, 11 genes] (target)
       - pert_emb:       [batch_size, pert_dim] (one-hot)
    4. Feed to model
    5. Compare prediction with pert_cell_emb
    6. Update weights
```

---

## 🖥️ **Training Command**

### **Makefile Command:**
```bash
make train
```

### **What It Does:**
```bash
./run.sh tx train \
  data.kwargs.toml_config_path="./examples/mixed.toml" \  ← Config file
  data.kwargs.embed_key=X_hvg \                           ← Use X_hvg data
  data.kwargs.pert_col=target_gene \                      ← Perturbation column
  data.kwargs.cell_type_key=cell_type \                   ← Cell type column
  data.kwargs.batch_col=batch_var \                       ← Batch column
  data.kwargs.control_pert=TARGET1 \                      ← Control label
  training.max_steps=5000 \                               ← Train for 5000 steps
  training.batch_size=64 \                                ← 64 cells per batch
  ...
```

### **Key Parameters:**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `toml_config_path` | `./examples/mixed.toml` | Points to config that references `random.h5ad` |
| `embed_key` | `X_hvg` | Use `adata.obsm['X_hvg']` (11 genes) |
| `pert_col` | `target_gene` | Perturbation labels are in `adata.obs['target_gene']` |
| `cell_type_key` | `cell_type` | Cell types are in `adata.obs['cell_type']` |
| `batch_col` | `batch_var` | Batch IDs are in `adata.obs['batch_var']` |
| `control_pert` | `TARGET1` | Use TARGET1 as control/baseline |

---

## 📝 **Complete Data Flow**

```
┌──────────────────────────────────────────────────────┐
│ 1. INPUT FILE: examples/random.h5ad                  │
│    • 10,000 cells                                    │
│    • 11 genes (X_hvg)                                │
│    • 5 perturbations (TARGET1-5)                     │
│    • 5 cell types (CT1-5)                            │
└──────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────┐
│ 2. CONFIGURATION: examples/mixed.toml                │
│    • Specifies data directory: ./examples            │
│    • Defines train/val/test splits                   │
│    • Identifies control: TARGET1                     │
└──────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────┐
│ 3. DATA LOADER                                       │
│    • Loads random.h5ad                               │
│    • Samples cells for training                      │
│    • Creates control-perturbed pairs                 │
└──────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────┐
│ 4. TRAINING BATCH                                    │
│    • ctrl_cell_emb: [64, 11] from TARGET1            │
│    • pert_cell_emb: [64, 11] from TARGET2-5          │
│    • pert_emb: [64, pert_dim] one-hot labels         │
└──────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────┐
│ 5. MODEL TRAINING                                    │
│    • Encode control cells                            │
│    • Encode perturbations                            │
│    • Combine and transform                           │
│    • Predict perturbed state                         │
│    • Compare with pert_cell_emb                      │
│    • Update weights                                  │
└──────────────────────────────────────────────────────┘
```

---

## 🔍 **Verify Your Input File**

```bash
# Check if file exists
ls -lh examples/random.h5ad

# Check file size
# Should be a few MB

# Inspect with Python
python -c "
import scanpy as sc
adata = sc.read_h5ad('examples/random.h5ad')
print(f'Cells: {adata.n_obs}')
print(f'Genes: {adata.n_vars}')
print(f'Perturbations: {adata.obs[\"target_gene\"].unique()}')
print(f'Cell types: {adata.obs[\"cell_type\"].unique()}')
"
```

**Expected Output:**
```
Cells: 10000
Genes: 50
Perturbations: ['TARGET1' 'TARGET2' 'TARGET3' 'TARGET4' 'TARGET5']
Cell types: ['CT1' 'CT2' 'CT3' 'CT4' 'CT5']
```

---

## 💡 **Key Points**

### ✅ **Single File Contains Everything**
- You don't need separate files for control vs perturbed
- Everything is in `random.h5ad`
- Different perturbations are distinguished by the `target_gene` column

### ✅ **Control is Identified by Label**
- `TARGET1` cells are controls (unperturbed)
- Specified in Makefile: `CONTROL_PERT ?= TARGET1`
- Data loader automatically separates them

### ✅ **Gene Expression Data**
- Uses `adata.obsm['X_hvg']` (11 highly variable genes)
- Not the full 50 genes in `adata.X`
- Specified by: `EMBED_KEY ?= X_hvg`

### ✅ **Metadata is Critical**
- `target_gene` column: Which perturbation (TARGET1-5)
- `cell_type` column: Which cell type (CT1-5)
- `batch_var` column: Which batch (used for batch correction)

---

## 🎯 **Summary**

**Question:** Which one is the training input file?

**Answer:** 
```
FILE: examples/random.h5ad
```

This single file contains:
- ✓ All 10,000 cells
- ✓ Control cells (TARGET1)
- ✓ Perturbed cells (TARGET2-5)
- ✓ Gene expression data
- ✓ All metadata (perturbations, cell types, batches)

**How to use:**
1. File is already in place: `examples/random.h5ad`
2. Configuration points to it: `examples/mixed.toml`
3. Just run: `make train`
4. The system handles everything!

**No separate files needed** - it's all in one H5AD file! 🎉


