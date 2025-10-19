# 🎯 Quick Guide: Where to Put Debug Breakpoint for Data Batching

## ⭐ **THE ANSWER:**

### **File:** `src/state/tx/data/dataset/scgpt_perturbation_dataset.py`
### **Line:** 232
### **Function:** `collate_fn()`

This is **THE FIRST PLACE** where data is batched!

---

## 🔧 **How to Set Breakpoint:**

### **Option 1: Add pdb (Python Debugger)**

Edit the file and add this line:

```python
# File: src/state/tx/data/dataset/scgpt_perturbation_dataset.py
# Line: 232

@staticmethod
def collate_fn(batch, transform=None, pert_col="drug", int_counts=False):
    """
    Custom collate that reshapes data into sequences.
    Safely handles normalization when vectors sum to zero.
    """
    import pdb; pdb.set_trace()  # ← ADD THIS LINE
    
    # First do normal collation
    batch_dict = {
        "pert_cell_emb": torch.stack([item["pert_cell_emb"] for item in batch]),
        "ctrl_cell_emb": torch.stack([item["ctrl_cell_emb"] for item in batch]),
        ...
    }
```

Then run training normally:
```bash
make train
```

When the breakpoint hits, you can inspect:
```python
# In pdb console:
len(batch)                 # Batch size (e.g., 64)
batch[0].keys()           # Keys in first sample
batch[0]["ctrl_cell_emb"] # First control cell
batch[0]["pert_name"]     # First perturbation name
```

### **Option 2: Use VS Code Debugger**

1. Open: `src/state/tx/data/dataset/scgpt_perturbation_dataset.py`
2. Go to line 232
3. Click left of line number (red dot appears)
4. Run with debugger (F5)

### **Option 3: Use PyCharm**

1. Open file
2. Click left margin at line 232
3. Click Debug button

---

## 📊 **What You'll See at the Breakpoint:**

### **Input (before batching):**
```python
batch = [
    {  # Sample 0
        "ctrl_cell_emb": tensor([0.00, 0.74, ...]),  # [11 genes]
        "pert_cell_emb": tensor([0.67, 1.85, ...]),  # [11 genes]
        "pert_emb": tensor([0, 1, 0, 0, 0]),         # [pert_dim]
        "pert_name": "TARGET2",
        "cell_type": "CT1",
        ...
    },
    {  # Sample 1
        "ctrl_cell_emb": tensor([0.00, 0.94, ...]),
        ...
    },
    ...  # 64 total samples
]
```

### **Output (after batching):**
```python
batch_dict = {
    "ctrl_cell_emb": tensor([[...], [...],...]),  # [64, 11]
    "pert_cell_emb": tensor([[...], [...],...]),  # [64, 11]
    "pert_emb": tensor([[...], [...],...]),       # [64, pert_dim]
    "pert_name": ["TARGET2", "TARGET3", ...],     # List of 64
    "cell_type": ["CT1", "CT2", ...],             # List of 64
    ...
}
```

---

## 📍 **Alternative Breakpoint Locations:**

If you want to debug **before** batching:

### **Single Sample Creation:**
```python
File: src/state/tx/data/dataset/scgpt_perturbation_dataset.py
Line: 145

def __getitem__(self, idx: int):
    ...
    pert_expr, ctrl_expr, ctrl_idx = self.mapping_strategy.get_mapped_expressions(...)
    import pdb; pdb.set_trace()  # ← Breakpoint here
```

### **Model Receives Batch:**
```python
File: src/state/tx/models/state_transition.py
Line: 443

def training_step(self, batch, batch_idx):
    import pdb; pdb.set_trace()  # ← Breakpoint here
    ...
```

---

## ✅ **Summary:**

**Question:** Where is the first place data is batched?

**Answer:**
```
File: src/state/tx/data/dataset/scgpt_perturbation_dataset.py
Line: 232
Function: collate_fn()

Add: import pdb; pdb.set_trace()
```

This function takes a **list of individual samples** and combines them into a **single batched dictionary** using `torch.stack()`.

**That's where batching happens!** 🎯

