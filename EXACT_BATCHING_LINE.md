# 🎯 EXACT LOCATION WHERE DATA IS BATCHED

## ✅ **THE ANSWER:**

### **File:** 
```
src/state/tx/data/dataset/scgpt_perturbation_dataset.py
```

### **Lines 232-244:**
```python
230|    ##############################
231|    # Static methods
232|    ##############################
233|    @staticmethod
234|    def collate_fn(batch, transform=None, pert_col="drug", int_counts=False):
235|        """
236|        Custom collate that reshapes data into sequences.
237|        Safely handles normalization when vectors sum to zero.
238|        """
239|        # First do normal collation
240|        batch_dict = {
241|            "pert_cell_emb": torch.stack([item["pert_cell_emb"] for item in batch]),
242|            "ctrl_cell_emb": torch.stack([item["ctrl_cell_emb"] for item in batch]),
243|            "pert_emb": torch.stack([item["pert_emb"] for item in batch]),
244|            "pert_name": [item["pert_name"] for item in batch],
245|            "cell_type": [item["cell_type"] for item in batch],
246|            "cell_type_onehot": torch.stack([item["cell_type_onehot"] for item in batch]),
247|            "batch": torch.stack([item["batch"] for item in batch]),
```

---

## 🔍 **EXACTLY WHERE BATCHING HAPPENS:**

### **Lines 241-243** are the key batching lines:

```python
"pert_cell_emb": torch.stack([item["pert_cell_emb"] for item in batch]),
"ctrl_cell_emb": torch.stack([item["ctrl_cell_emb"] for item in batch]),  
"pert_emb": torch.stack([item["pert_emb"] for item in batch]),
```

**What `torch.stack()` does:**
```python
# Input (batch is a list):
batch = [
    {"ctrl_cell_emb": tensor([0.1, 0.2, ...])},  # Sample 0: [11]
    {"ctrl_cell_emb": tensor([0.3, 0.4, ...])},  # Sample 1: [11]
    ...  # 64 total samples
]

# torch.stack([item["ctrl_cell_emb"] for item in batch])
# Output:
tensor([[0.1, 0.2, ...],
        [0.3, 0.4, ...],
        ...])  # [64, 11]
```

---

## 🔧 **HOW TO SET BREAKPOINT:**

### **Method 1: pdb (Easiest)**

Add this at line 234 (right after `@staticmethod`):

```python
@staticmethod
def collate_fn(batch, transform=None, pert_col="drug", int_counts=False):
    import pdb; pdb.set_trace()  # ← ADD THIS LINE
    """
    Custom collate that reshapes data into sequences.
    """
```

### **Method 2: Print statements**

Add prints to see batching happen:

```python
@staticmethod
def collate_fn(batch, transform=None, pert_col="drug", int_counts=False):
    print(f"\n{'='*60}")
    print(f"BATCHING HAPPENING NOW!")
    print(f"Input: list of {len(batch)} samples")
    print(f"Sample 0 type: {type(batch[0])}")
    print(f"Sample 0 keys: {batch[0].keys()}")
    print(f"Sample 0 ctrl_cell shape: {batch[0]['ctrl_cell_emb'].shape}")
    print(f"{'='*60}\n")
    
    # First do normal collation
    batch_dict = {
        "pert_cell_emb": torch.stack([item["pert_cell_emb"] for item in batch]),
        "ctrl_cell_emb": torch.stack([item["ctrl_cell_emb"] for item in batch]),
        ...
    }
    
    print(f"\n{'='*60}")
    print(f"BATCHING COMPLETE!")
    print(f"Output ctrl_cell shape: {batch_dict['ctrl_cell_emb'].shape}")
    print(f"Output pert_cell shape: {batch_dict['pert_cell_emb'].shape}")
    print(f"{'='*60}\n")
    
    return batch_dict
```

Then run: `make train`

---

## 📊 **WHY YOU COULDN'T FIND IT:**

The batching is **hidden** because:

1. **PyTorch calls it automatically:**
   - You write: `for batch in train_loader:`
   - PyTorch internally calls: `collate_fn(samples)`
   - You never explicitly see the call!

2. **It's registered as a callback:**
   - Set up in: `cell_load/data_modules/perturbation_dataloader.py` line 376
   - Used by: PyTorch DataLoader line 388
   - Called during: iteration (invisible to you)

3. **The function name doesn't say "batch":**
   - It's called `collate_fn` (standard PyTorch terminology)
   - "Collate" means "combine" or "merge"
   - This is PyTorch's standard way of batching

---

## 🎯 **TO SUMMARIZE:**

**You asked:** "Where is the first place data is batched?"

**Answer:** 

```
File: src/state/tx/data/dataset/scgpt_perturbation_dataset.py
Line: 241-243 (the torch.stack() calls)
Function: collate_fn() (starts at line 234)
```

**Put breakpoint at line 234 right after `@staticmethod`**

This is called automatically by PyTorch DataLoader during training iteration.

**This is THE ONLY place where individual samples become batches!** 🎯

---

## ✅ **VERIFICATION:**

To verify this is correct, run:

```bash
cd src/state/tx/data/dataset/
grep -n "torch.stack" scgpt_perturbation_dataset.py
```

You'll see torch.stack() calls at lines 241-247 - **these are the batching operations!**

