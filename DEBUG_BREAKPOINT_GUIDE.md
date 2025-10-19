# Debug Breakpoint Guide: Where Data is Batched

## 🎯 Key Locations for Debugging Data

Here are the **exact locations** to set breakpoints to inspect data at each stage of batching:

---

## 📍 **BREAKPOINT 1: DataModule Setup**

### **File:** `src/state/_cli/_tx/_train.py`
### **Line:** 122-123

```python
data_module.setup(stage="fit")  # ← BREAKPOINT HERE
dl = data_module.train_dataloader()  # ← OR HERE
```

**What you'll see:**
- Data module initializes datasets
- Splits data into train/val/test
- Creates perturbation/cell_type/batch mappings

**Variables to inspect:**
```python
data_module.train_dataset    # Training dataset
data_module.val_dataset      # Validation dataset
data_module.pert_onehot_map  # Perturbation mappings
data_module.cell_type_onehot_map  # Cell type mappings
```

---

## 📍 **BREAKPOINT 2: Single Sample Creation (Before Batching)**

### **File:** `src/state/tx/data/dataset/scgpt_perturbation_dataset.py`
### **Line:** 125

```python
def __getitem__(self, idx: int):
    """
    Returns a dictionary with:
        - 'X': the (possibly transformed) expression of the perturbed cell
        - 'basal': the control cell's expression as chosen by the mapping strategy
        - 'pert': the one-hot encoding (or other featurization) for the perturbation
        - 'pert_name': the perturbation name
        - 'cell_type': the cell type (from the full array)
        - 'gem_group': the batch (as an int or string)

    The index `idx` here is into the filtered set of cells.
    """
    # Map idx to the underlying file index
    underlying_idx = int(self.all_indices[idx])  # ← BREAKPOINT HERE
    split = self._find_split_for_idx(underlying_idx)

    # Get expression from the h5 file
    pert_expr, ctrl_expr, ctrl_idx = self.mapping_strategy.get_mapped_expressions(
        self, split, underlying_idx
    )  # ← OR BREAKPOINT HERE
    
    # ... rest of the function
```

**What you'll see:**
- ONE sample being created
- Control cell paired with perturbed cell
- Perturbation label assignment

**Variables to inspect:**
```python
idx                # Sample index
underlying_idx     # Actual index in h5 file
pert_expr         # Perturbed cell expression [11 genes]
ctrl_expr         # Control cell expression [11 genes]
pert_name         # e.g., "TARGET2"
cell_type         # e.g., "CT1"
```

**Full sample at the end (line 198):**
```python
return sample  # ← BREAKPOINT HERE to see complete sample dict
```

---

## 📍 **BREAKPOINT 3: Batch Collation (WHERE BATCHING HAPPENS!)**

### **File:** `src/state/tx/data/dataset/scgpt_perturbation_dataset.py`
### **Line:** 232

```python
@staticmethod
def collate_fn(batch, transform=None, pert_col="drug", int_counts=False):
    """
    Custom collate that reshapes data into sequences.
    Safely handles normalization when vectors sum to zero.
    """
    # First do normal collation  ← BREAKPOINT HERE
    batch_dict = {
        "pert_cell_emb": torch.stack([item["pert_cell_emb"] for item in batch]),
        "ctrl_cell_emb": torch.stack([item["ctrl_cell_emb"] for item in batch]),
        "pert_emb": torch.stack([item["pert_emb"] for item in batch]),
        "pert_name": [item["pert_name"] for item in batch],
        "cell_type": [item["cell_type"] for item in batch],
        "cell_type_onehot": torch.stack([item["cell_type_onehot"] for item in batch]),
        "batch": torch.stack([item["batch"] for item in batch]),
        "batch_name": [item["batch_name"] for item in batch],
        "gene_ids": torch.stack([item["gene_ids"] for item in batch]),
    }  # ← BREAKPOINT HERE to see batched dict
    
    # ... transformations ...
    
    return batch_dict  # ← BREAKPOINT HERE to see final batch
```

**What you'll see:**
- List of individual samples → stacked tensors
- Multiple cells combined into batch
- **THIS IS WHERE DATA IS FIRST BATCHED!**

**Variables to inspect:**
```python
batch              # List of individual samples (e.g., 64 samples)
len(batch)         # Batch size (e.g., 64)
batch[0]          # First sample (dict with keys like 'ctrl_cell_emb', 'pert_emb')
batch_dict        # Final batched dictionary with stacked tensors
batch_dict["ctrl_cell_emb"].shape   # e.g., [64, 11]
batch_dict["pert_cell_emb"].shape   # e.g., [64, 11]
batch_dict["pert_emb"].shape        # e.g., [64, 64]
```

---

## 📍 **BREAKPOINT 4: Model Receives Batch**

### **File:** `src/state/tx/models/state_transition.py`
### **Line:** 443

```python
def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, padded=True) -> torch.Tensor:
    """Training step logic for both main model and decoder."""
    # ← BREAKPOINT HERE to see batch at model entry
    
    # Get model predictions (in latent space)
    confidence_pred = None
    if self.confidence_token is not None:
        pred, confidence_pred = self.forward(batch, padded=padded)
    else:
        pred = self.forward(batch, padded=padded)

    target = batch["pert_cell_emb"]  # ← BREAKPOINT HERE to see target
    
    # ... rest of training step
```

**What you'll see:**
- Complete batch as received by model
- Batch after any preprocessing/reshaping
- Targets for training

**Variables to inspect:**
```python
batch              # Complete batch dict
batch.keys()       # All keys in batch
batch["ctrl_cell_emb"]     # Control cells
batch["pert_cell_emb"]     # Perturbed cells (targets)
batch["pert_emb"]          # Perturbation labels
batch["cell_type"]         # Cell type names
batch_idx          # Which batch in epoch (0, 1, 2, ...)
```

---

## 📍 **BREAKPOINT 5: Model Forward Pass**

### **File:** `src/state/tx/models/state_transition.py`
### **Line:** 354-367

```python
def forward(self, batch: dict, padded=True) -> torch.Tensor:
    # ← BREAKPOINT HERE to see batch at forward entry
    
    if padded:
        pert = batch["pert_emb"].reshape(-1, self.cell_sentence_len, self.pert_dim)
        basal = batch["ctrl_cell_emb"].reshape(-1, self.cell_sentence_len, self.input_dim)
        # ← BREAKPOINT HERE to see reshaped inputs
    
    # Shape: [B, S, input_dim]
    pert_embedding = self.encode_perturbation(pert)
    control_cells = self.encode_basal_expression(basal)
    # ← BREAKPOINT HERE to see encoded inputs
    
    # Add encodings in input_dim space, then project to hidden_dim
    combined_input = pert_embedding + control_cells
    # ← BREAKPOINT HERE to see combined inputs
```

---

## 🎯 **RECOMMENDED DEBUGGING SEQUENCE**

### **For First-Time Debugging:**

```python
# 1. START HERE - See how data is loaded
File: src/state/_cli/_tx/_train.py
Line: 122
Breakpoint: data_module.setup(stage="fit")

# 2. NEXT - See individual samples
File: src/state/tx/data/dataset/scgpt_perturbation_dataset.py
Line: 145
Breakpoint: pert_expr, ctrl_expr, ctrl_idx = self.mapping_strategy.get_mapped_expressions(...)

# 3. CRITICAL - See batching happen ⭐
File: src/state/tx/data/dataset/scgpt_perturbation_dataset.py
Line: 238
Breakpoint: batch_dict = { ... }

# 4. VERIFY - See batch at model
File: src/state/tx/models/state_transition.py
Line: 443
Breakpoint: def training_step(self, batch, batch_idx):
```

---

## 💻 **How to Set Breakpoints**

### **Using pdb (Python Debugger):**

Add this line at the breakpoint location:
```python
import pdb; pdb.set_trace()
```

Example:
```python
# In collate_fn
def collate_fn(batch, transform=None, pert_col="drug", int_counts=False):
    import pdb; pdb.set_trace()  # ← Add this line
    batch_dict = {
        "pert_cell_emb": torch.stack([item["pert_cell_emb"] for item in batch]),
        ...
    }
```

### **Using VS Code Debugger:**

1. Open file in VS Code
2. Click left of line number to add breakpoint (red dot)
3. Run with F5 or Debug > Start Debugging

### **Using PyCharm:**

1. Click left of line number to add breakpoint
2. Click Debug button or Shift+F9

---

## 🔍 **What to Inspect at Each Breakpoint**

### **At Breakpoint 3 (collate_fn) - MOST IMPORTANT:**

```python
# When breakpoint hits, in debugger console type:

# See batch size
len(batch)

# See first sample structure
batch[0].keys()

# See first sample data
batch[0]["ctrl_cell_emb"]     # Control cell
batch[0]["pert_cell_emb"]     # Perturbed cell
batch[0]["pert_name"]         # Perturbation name
batch[0]["cell_type"]         # Cell type

# After batching
batch_dict["ctrl_cell_emb"].shape    # Should be [batch_size, n_genes]
batch_dict["pert_cell_emb"].shape    # Should be [batch_size, n_genes]
batch_dict["pert_emb"].shape         # Should be [batch_size, pert_dim]

# Check values
batch_dict["ctrl_cell_emb"][0]       # First control cell
batch_dict["pert_cell_emb"][0]       # First perturbed cell
batch_dict["pert_name"][:5]          # First 5 perturbation names
batch_dict["cell_type"][:5]          # First 5 cell types
```

---

## 📋 **Quick Reference: File Locations**

| Location | File | Line | Purpose |
|----------|------|------|---------|
| **Setup** | `src/state/_cli/_tx/_train.py` | 122 | Data module setup |
| **DataLoader** | `src/state/_cli/_tx/_train.py` | 123 | Get dataloader |
| **Single Sample** | `src/state/tx/data/dataset/scgpt_perturbation_dataset.py` | 125 | `__getitem__` |
| **⭐ Batching** | `src/state/tx/data/dataset/scgpt_perturbation_dataset.py` | 232 | `collate_fn` ⭐ |
| **Model Entry** | `src/state/tx/models/state_transition.py` | 443 | `training_step` |
| **Forward Pass** | `src/state/tx/models/state_transition.py` | 354 | `forward` |

---

## 🎯 **Minimal Debug Script**

Create `debug_data.py` in the project root:

```python
"""
Minimal script to debug data loading
"""
import torch
from omegaconf import OmegaConf
from cell_load.utils.modules import get_datamodule

# Load config
cfg = OmegaConf.load("examples/mixed.toml")

# Create data module
data_module = get_datamodule(
    "perturbation",
    {
        "toml_config_path": "examples/mixed.toml",
        "embed_key": "X_hvg",
        "control_pert": "TARGET1",
        "pert_col": "target_gene",
        "cell_type_key": "cell_type",
        "batch_col": "batch_var",
    },
    batch_size=4,  # Small batch for debugging
    cell_sentence_len=4,
)

# Setup
data_module.setup(stage="fit")

# Get dataloader
train_loader = data_module.train_dataloader()

# Get ONE batch
print("Getting first batch...")
for batch_idx, batch in enumerate(train_loader):
    print(f"\n=== BATCH {batch_idx} ===")
    print(f"Keys: {batch.keys()}")
    print(f"ctrl_cell_emb shape: {batch['ctrl_cell_emb'].shape}")
    print(f"pert_cell_emb shape: {batch['pert_cell_emb'].shape}")
    print(f"pert_emb shape: {batch['pert_emb'].shape}")
    print(f"pert_names: {batch['pert_name']}")
    print(f"cell_types: {batch['cell_type']}")
    
    print(f"\nFirst control cell: {batch['ctrl_cell_emb'][0]}")
    print(f"First perturbed cell: {batch['pert_cell_emb'][0]}")
    
    # ← PUT BREAKPOINT HERE
    import pdb; pdb.set_trace()
    
    break  # Only process first batch
```

Run with:
```bash
python debug_data.py
```

---

## ✅ **Summary**

**Question:** Where is the first place data is batched?

**Answer:** 
```
File: src/state/tx/data/dataset/scgpt_perturbation_dataset.py
Line: 232
Function: collate_fn()

This is where individual samples are stacked into batches!
```

**Set breakpoint at:**
```python
def collate_fn(batch, transform=None, pert_col="drug", int_counts=False):
    import pdb; pdb.set_trace()  # ← Add this
    batch_dict = {
        "pert_cell_emb": torch.stack([item["pert_cell_emb"] for item in batch]),
        ...
    }
```

This is the **FIRST and PRIMARY location** where individual samples become a batch! 🎯

