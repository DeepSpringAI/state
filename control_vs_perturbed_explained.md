# Control vs Perturbed Cells: Different Lengths Explained

## 🎯 Your Observation is CORRECT!

In your dataset:
```
Control (TARGET1):     2,077 cells  ← More control cells
Perturbed (TARGET5):   2,032 cells  ← Fewer perturbed cells
```

**Question**: How does the model handle different numbers?

---

## 📊 Three Levels of Understanding

### Level 1: Dataset Level (Storage)

```
┌─────────────────────────────────────────────┐
│         Your H5AD File                      │
├─────────────────────────────────────────────┤
│                                             │
│  TARGET1 (Control):     2,077 cells ████   │
│  TARGET2 (Perturbed):   1,951 cells ███    │
│  TARGET3 (Perturbed):   2,017 cells ████   │
│  TARGET4 (Perturbed):   1,923 cells ███    │
│  TARGET5 (Perturbed):   2,032 cells ████   │
│                                             │
│  Total: 10,000 cells                        │
└─────────────────────────────────────────────┘
```

**At this level:** Different lengths exist naturally!

---

### Level 2: Sampling Level (Data Loader)

When creating training batches, the **mapping strategy** pairs cells:

```
For each perturbed cell, sample control cell(s):

Perturbed Cell #1 (TARGET2) ──→ Sample 1 control from TARGET1
Perturbed Cell #2 (TARGET2) ──→ Sample 1 control from TARGET1
Perturbed Cell #3 (TARGET3) ──→ Sample 1 control from TARGET1
...
Perturbed Cell #N (TARGET5) ──→ Sample 1 control from TARGET1
```

**Key Point**: 
- You have **2,077 control cells** available
- You have **1,951 + 2,017 + 1,923 + 2,032 = 7,923 perturbed cells**
- Each perturbed cell gets paired with a **random control cell**
- Control cells can be **reused** (sampled with replacement)

---

### Level 3: Model Input Level (After Collation)

After the collator creates batches:

```
Batch (64 samples):
┌────────────────────────────────────────┐
│  ctrl_cell_emb:  [64, 11 genes]        │  ← 64 control cells
│  pert_cell_emb:  [64, 11 genes]        │  ← 64 perturbed cells (targets)
│  pert_emb:       [64, pert_dim]        │  ← 64 perturbation labels
└────────────────────────────────────────┘
```

Then reshaped for model:

```
Model Forward (with cell_sentence_len = 64):
┌────────────────────────────────────────┐
│  ctrl_cell_emb:  [1, 64, 11]           │  ← All 64 controls in one sentence
│  pert_emb:       [1, 64, pert_dim]     │  ← All 64 pert labels
│                                        │
│  (B=1 batch, S=64 cells per sentence) │
└────────────────────────────────────────┘
```

**At this level:** Same shape! [B, S, dim]

---

## 🔄 Complete Flow with Your Data

### Example Training Iteration:

```
Step 1: DataLoader samples 64 perturbed cells
   - 15 cells with TARGET2
   - 18 cells with TARGET3
   - 14 cells with TARGET4
   - 17 cells with TARGET5

Step 2: For EACH perturbed cell, sample a control
   - Cell 1 (TARGET2) → Control cell #452 from TARGET1
   - Cell 2 (TARGET2) → Control cell #1203 from TARGET1
   - Cell 3 (TARGET3) → Control cell #89 from TARGET1
   - Cell 4 (TARGET3) → Control cell #1804 from TARGET1
   - ... (64 total pairs)

Step 3: Collate into batch
   ctrl_cell_emb:  [64, 11]  ← 64 sampled controls
   pert_cell_emb:  [64, 11]  ← 64 perturbed cells (ground truth)
   pert_emb:       [64, 64]  ← 64 perturbation one-hots

Step 4: Reshape for model
   ctrl_cell_emb:  [1, 64, 11]   ← Reshaped to [B, S, genes]
   pert_emb:       [1, 64, 64]   ← Reshaped to [B, S, pert_dim]

Step 5: Model processes
   Encode control:    [1, 64, 128]
   Encode pert:       [1, 64, 128]
   Combined (ADD):    [1, 64, 128]
   Transformer:       [1, 64, 128]
   Output projection: [1, 64, 11]  ← Predictions

Step 6: Compare with target
   Predicted:  [1, 64, 11]
   Target:     [1, 64, 11]  (the pert_cell_emb)
   Loss:       MSE(predicted, target)
```

---

## 💡 Key Insights

### 1. **Control cells are REUSED**
```
You have 2,077 control cells
You have 7,923 perturbed cells

During training:
- Each epoch, perturbed cells are sampled
- For each, a random control is sampled
- Same control cell might be used multiple times
- This is INTENTIONAL! (data augmentation)
```

### 2. **Perturbation is a LABEL, not cell data**
```
pert_emb = [0, 1, 0, 0, 0]  ← One-hot for "TARGET2"
         This is just an ID!

ctrl_cell_emb = [0.74, 0.71, ...]  ← Actual gene expression
pert_cell_emb = [1.85, 0.00, ...]  ← Actual gene expression
```

The model learns:
- `ctrl_cell_emb` = "Where you start"
- `pert_emb` = "What intervention to apply"
- `pert_cell_emb` = "Where you should end up" (training target)

### 3. **All shapes match in the model**
```
Dataset Level:
✗ Different numbers: 2,077 control vs 1,951 perturbed

Batch Level:
✓ Same shape: [64, 11] control vs [64, 11] perturbed

Model Level:
✓ Same shape: [1, 64, 11] control vs [1, 64, 11] perturbed
```

---

## 🔍 Mapping Strategies

How control cells are selected:

### 1. **Random Strategy** (most common)
```python
For each perturbed cell:
    - Get its cell_type (e.g., "CT2")
    - Randomly sample a control cell from TARGET1 with same cell_type
    - Return that control's expression
```

### 2. **Batch Strategy**
```python
For each perturbed cell:
    - Get its batch_id (e.g., "batch_3")
    - Randomly sample a control from TARGET1 in same batch
    - Return that control's expression
```

### 3. **Nearest Neighbor Strategy**
```python
For each perturbed cell:
    - Find K nearest control cells (by gene expression)
    - Sample one of those K
    - Return that control's expression
```

All strategies ensure: **1 perturbed cell → 1 control cell** in each sample

---

## 📝 Code Evidence

From `state_transition.py`:

```python
# Line 356: Both are reshaped to same shape
basal = batch["ctrl_cell_emb"].reshape(-1, self.cell_sentence_len, self.input_dim)
#              └─────────────┘                  └──────────────┘   └────────┘
#              Control cells                    Number of cells    Gene dim

pert = batch["pert_emb"].reshape(-1, self.cell_sentence_len, self.pert_dim)
#      └────────────┘                └──────────────┘   └────────┘
#      Perturbation IDs              Number of cells    Pert dim

# Both have shape: [B, S, dim] where S = cell_sentence_len
```

From `scgpt_perturbation_dataset.py`:

```python
# Line 145: Each sample gets ONE control and ONE perturbed cell
pert_expr, ctrl_expr, ctrl_idx = self.mapping_strategy.get_mapped_expressions(
    self, split, underlying_idx
)
#  └──────┘ └──────┘
#  One      One control
#  perturbed (sampled from available controls)
```

---

## ✅ Summary

**Your observation**: Different numbers at dataset level  
**Answer**: Correct! But they're paired 1-to-1 during training

```
Dataset:    2,077 controls vs 1,951 perturbed  ← Different!
            ↓ Sampling pairs them
Batch:      64 controls vs 64 perturbed        ← Same!
            ↓ Reshape
Model:      [1, 64, 11] vs [1, 64, 11]         ← Same shape!
```

**Why this works:**
1. Control cells are sampled **with replacement** (reused)
2. Each perturbed cell gets paired with a control during loading
3. By the time data reaches the model, shapes match perfectly!

**Benefit:**
- Having more controls (2,077) is GOOD!
- Gives more diversity when sampling
- Better representation of baseline state
- Reduces overfitting to specific control cells


