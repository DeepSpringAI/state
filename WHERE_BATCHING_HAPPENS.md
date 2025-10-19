# ✅ THE COMPLETE ANSWER: Where Data is Batched

## 🎯 **The Batching Flow (Step by Step)**

### **Step 1: DataLoader is Created** 
📍 **Location:** `.venv/lib/python3.11/site-packages/cell_load/data_modules/perturbation_dataloader.py`  
📍 **Line:** 364-395

```python
def _create_dataloader(self, datasets, test=False, batch_size=None):
    """Create a DataLoader with appropriate configuration."""
    
    # ⭐ THIS is where collate_fn is specified!
    collate_fn = partial(PerturbationDataset.collate_fn, int_counts=use_int_counts)
    # Line 376 ↑
    
    ds = MetadataConcatDataset(datasets)
    
    # Return DataLoader with collate_fn attached
    return DataLoader(
        ds,
        batch_sampler=sampler,
        num_workers=self.num_workers,
        collate_fn=collate_fn,  # ← collate_fn is passed here! (Line 391)
        pin_memory=True,
        prefetch_factor=4 if not test and self.num_workers > 0 else None,
    )
```

**What happens:** The DataLoader is created with `collate_fn` parameter pointing to `PerturbationDataset.collate_fn`

---

### **Step 2: DataLoader Iterates (During Training)**
📍 **Location:** PyTorch DataLoader (internal)  
📍 **When:** Every time you do `for batch in train_loader:`

```python
# In training code (e.g., src/state/_cli/_tx/_train.py line 123)
train_loader = data_module.train_dataloader()

# When PyTorch Lightning calls trainer.fit(), it does:
for batch_idx, batch in enumerate(train_loader):  # ← Iteration happens here
    # PyTorch DataLoader:
    # 1. Samples batch_size indices
    # 2. Calls dataset.__getitem__(idx) for each index → gets list of samples
    # 3. ⭐ Calls collate_fn(list_of_samples) ← BATCHING HAPPENS HERE!
    # 4. Returns the batched result
    pass
```

---

### **Step 3: collate_fn is Called (BATCHING ACTUALLY HAPPENS HERE!)**
📍 **Location:** `src/state/tx/data/dataset/scgpt_perturbation_dataset.py`  
📍 **Line:** 232  
📍 **Function:** `collate_fn()`

```python
@staticmethod
def collate_fn(batch, transform=None, pert_col="drug", int_counts=False):
    """
    Custom collate that reshapes data into sequences.
    Safely handles normalization when vectors sum to zero.
    """
    # ⭐⭐⭐ THIS IS WHERE BATCHING HAPPENS! ⭐⭐⭐
    # Input: batch = list of 64 individual samples
    # Output: batch_dict = dictionary with stacked tensors
    
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
    }
    
    # ... additional transformations ...
    
    return batch_dict  # ← Returns batched data
```

**What happens:** Individual samples (list) are stacked into tensors using `torch.stack()`

---

## 📊 **Complete Visual Flow**

```
┌─────────────────────────────────────────────────────────────────┐
│                     COMPLETE BATCHING FLOW                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. SETUP (One-time, line 122 in _train.py)                    │
│     data_module.setup(stage="fit")                             │
│     train_loader = data_module.train_dataloader()             │
│                         ↓                                       │
│     Calls: _create_dataloader()                                │
│            ↓                                                    │
│     Creates: DataLoader with collate_fn attached              │
│                                                                  │
│  ─────────────────────────────────────────────────────────────  │
│                                                                  │
│  2. TRAINING LOOP (Every iteration)                            │
│     for batch in train_loader:  # ← Iteration starts          │
│                         ↓                                       │
│     PyTorch DataLoader internally:                             │
│        a. Samples 64 indices from dataset                      │
│        b. Calls dataset.__getitem__(idx) 64 times             │
│           → Returns list of 64 individual samples              │
│        c. ⭐ Calls collate_fn(list_of_64_samples) ⭐          │
│           → Returns batched dictionary                         │
│                                                                  │
│  ─────────────────────────────────────────────────────────────  │
│                                                                  │
│  3. COLLATE_FN (Where batching actually happens!)              │
│     File: scgpt_perturbation_dataset.py                       │
│     Line: 232                                                  │
│                         ↓                                       │
│     Input: [sample_0, sample_1, ..., sample_63]               │
│            Each sample = {                                     │
│                "ctrl_cell_emb": [11 genes],                   │
│                "pert_cell_emb": [11 genes],                   │
│                "pert_emb": [pert_dim],                        │
│                ...                                             │
│            }                                                    │
│                         ↓                                       │
│     torch.stack() combines them:                              │
│     batch_dict = {                                             │
│         "ctrl_cell_emb": torch.stack([...]),  # [64, 11]     │
│         "pert_cell_emb": torch.stack([...]),  # [64, 11]     │
│         "pert_emb": torch.stack([...]),       # [64, pert]   │
│         ...                                                    │
│     }                                                          │
│                         ↓                                       │
│     Output: batch_dict (batched tensors!)                     │
│                                                                  │
│  ─────────────────────────────────────────────────────────────  │
│                                                                  │
│  4. MODEL RECEIVES BATCH                                        │
│     def training_step(self, batch, batch_idx):                │
│         # batch is the output from collate_fn                 │
│         # batch["ctrl_cell_emb"] is [64, 11]                  │
│         ...                                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 **WHERE TO PUT YOUR BREAKPOINT**

### **Option 1: See batching happen (RECOMMENDED)**
```python
File: src/state/tx/data/dataset/scgpt_perturbation_dataset.py
Line: 232

@staticmethod
def collate_fn(batch, transform=None, pert_col="drug", int_counts=False):
    import pdb; pdb.set_trace()  # ← PUT BREAKPOINT HERE
    
    batch_dict = {
        "pert_cell_emb": torch.stack([item["pert_cell_emb"] for item in batch]),
        ...
    }
```

**What you'll see:**
- `batch` = list of individual samples (e.g., length 64)
- `batch[0]` = first sample (dict with keys like 'ctrl_cell_emb')
- After `torch.stack()`, you see batched tensors

### **Option 2: See DataLoader creation**
```python
File: .venv/lib/python3.11/site-packages/cell_load/data_modules/perturbation_dataloader.py
Line: 376

def _create_dataloader(self, datasets, test=False, batch_size=None):
    collate_fn = partial(PerturbationDataset.collate_fn, int_counts=use_int_counts)
    import pdb; pdb.set_trace()  # ← PUT BREAKPOINT HERE
    
    return DataLoader(..., collate_fn=collate_fn, ...)
```

**What you'll see:**
- How collate_fn is configured
- DataLoader parameters

### **Option 3: See batch after batching**
```python
File: src/state/tx/models/state_transition.py
Line: 443

def training_step(self, batch, batch_idx):
    import pdb; pdb.set_trace()  # ← PUT BREAKPOINT HERE
    # batch is already batched here
```

**What you'll see:**
- Fully batched data as received by model
- All tensors already stacked

---

## ⚡ **Why It's Hard to Find**

The batching happens **inside PyTorch's DataLoader**, which:
1. Is part of PyTorch (not your code)
2. Runs in a background thread (if `num_workers > 0`)
3. Calls `collate_fn` automatically during iteration

**The flow is:**
```python
# You don't see this in your code, but PyTorch does it:
samples = [dataset[i] for i in indices]  # Get individual samples
batch = collate_fn(samples)               # ← Batch them!
return batch                              # Yield to your code
```

---

## 📝 **Quick Test Script**

Create `test_batching.py`:

```python
"""Test where batching happens"""
import sys
sys.path.insert(0, 'src')

# Monkey-patch to see when collate_fn is called
from state.tx.data.dataset.scgpt_perturbation_dataset import scGPTPerturbationDataset

original_collate = scGPTPerturbationDataset.collate_fn

@staticmethod
def instrumented_collate(batch, *args, **kwargs):
    print(f"\n⭐ collate_fn CALLED!")
    print(f"   Input: list of {len(batch)} samples")
    print(f"   Sample 0 keys: {batch[0].keys()}")
    print(f"   Sample 0 ctrl shape: {batch[0]['ctrl_cell_emb'].shape}")
    
    result = original_collate(batch, *args, **kwargs)
    
    print(f"   Output: batched dict")
    print(f"   Output ctrl shape: {result['ctrl_cell_emb'].shape}")
    print(f"   ⭐ BATCHING COMPLETED!\n")
    
    return result

scGPTPerturbationDataset.collate_fn = instrumented_collate

# Now run training and you'll see when batching happens!
from state._cli._tx._train import run_tx_train
from omegaconf import OmegaConf

cfg = OmegaConf.load("path/to/config.yaml")
run_tx_train(cfg)
```

---

## ✅ **FINAL ANSWER**

**Question:** Where is the first place data is batched?

**Answer:** 

### **File:** `src/state/tx/data/dataset/scgpt_perturbation_dataset.py`  
### **Line:** 232  
### **Function:** `collate_fn()`

**This function is called by PyTorch's DataLoader during iteration.**

The DataLoader is set up at line 388 in:
`cell_load/data_modules/perturbation_dataloader.py`

But the **actual batching** (stacking individual samples) happens in `collate_fn()` when:
```python
torch.stack([item["ctrl_cell_emb"] for item in batch])
```

**Put your breakpoint at line 232 in `scgpt_perturbation_dataset.py`!** 🎯

