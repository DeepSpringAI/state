# ST Model Training Flow: Complete Data Combination Process

This document explains **exactly** how the ST model combines data during training, step by step.

---

## 🎯 Overview: From H5AD File to Model Training

```
random.h5ad → DataLoader → Batch → Model Forward → Loss → Update Weights
```

---

## 📊 Step 1: Data in File (random.h5ad)

### **File Structure:**
```
examples/random.h5ad
├── 10,000 cells
├── adata.obsm['X_hvg']: [10000, 11] gene expression
└── adata.obs metadata:
    ├── target_gene: TARGET1 (control), TARGET2-5 (perturbed)
    ├── cell_type:   CT1-CT5
    └── batch_var:   Batch IDs
```

### **Cell Distribution:**
```
TARGET1 (control):   2,077 cells  ← Unperturbed baseline
TARGET2 (perturbed): 1,951 cells  ← Intervention #1
TARGET3 (perturbed): 2,017 cells  ← Intervention #2
TARGET4 (perturbed): 1,923 cells  ← Intervention #3
TARGET5 (perturbed): 2,032 cells  ← Intervention #4
```

---

## 📦 Step 2: Dataset.__getitem__() - Creating Single Samples

### **Code:** `scgpt_perturbation_dataset.py` lines 125-198

```python
def __getitem__(self, idx: int):
    # 1. Get a perturbed cell index (idx points to a perturbed cell)
    underlying_idx = int(self.all_indices[idx])
    
    # 2. Sample ONE control cell that matches this perturbed cell
    pert_expr, ctrl_expr, ctrl_idx = self.mapping_strategy.get_mapped_expressions(
        self, split, underlying_idx
    )
    #   pert_expr: Expression of the perturbed cell
    #   ctrl_expr: Expression of a MATCHED control cell (same cell_type)
    
    # 3. Get perturbation label
    pert_name = self.metadata_cache.pert_categories[pert_code]  # e.g., "TARGET2"
    pert_onehot = self.pert_onehot_map[pert_name]  # Convert to one-hot
    
    # 4. Get metadata
    cell_type = self.metadata_cache.cell_type_categories[cell_type_code]
    cell_type_onehot = self.cell_type_onehot_map[cell_type]
    batch = self.batch_onehot_map[batch_name]
    
    # 5. Return ONE training sample
    return {
        "pert_cell_emb": pert_expr,    # [11 genes] Perturbed cell (TARGET OUTPUT)
        "ctrl_cell_emb": ctrl_expr,    # [11 genes] Control cell (MODEL INPUT)
        "pert_emb":      pert_onehot,  # [pert_dim] Perturbation one-hot (MODEL INPUT)
        "cell_type":     cell_type,
        "cell_type_onehot": cell_type_onehot,
        "batch":         batch,
        "pert_name":     pert_name,
    }
```

### **Key Points:**
- **ONE sample** = ONE perturbed cell + ONE matched control cell
- Control cell is sampled **on-the-fly** using mapping_strategy
- Mapping strategy ensures control has **same cell_type**
- Control cells can be **reused** (sampled with replacement)

---

## 🔄 Step 3: collate_fn() - Creating Batches

### **Code:** `scgpt_perturbation_dataset.py` lines 232-248

```python
@staticmethod
def collate_fn(batch, ...):
    """
    Takes multiple samples and stacks them into a batch
    
    Input: batch = list of 64 samples (from __getitem__)
    Output: batch_dict with stacked tensors
    """
    
    batch_dict = {
        "pert_cell_emb":    torch.stack([item["pert_cell_emb"] for item in batch]),
        "ctrl_cell_emb":    torch.stack([item["ctrl_cell_emb"] for item in batch]),
        "pert_emb":         torch.stack([item["pert_emb"] for item in batch]),
        "pert_name":        [item["pert_name"] for item in batch],
        "cell_type":        [item["cell_type"] for item in batch],
        "cell_type_onehot": torch.stack([item["cell_type_onehot"] for item in batch]),
        "batch":            torch.stack([item["batch"] for item in batch]),
        "batch_name":       [item["batch_name"] for item in batch],
        "gene_ids":         torch.stack([item["gene_ids"] for item in batch]),
    }
    
    return batch_dict
```

### **Example Batch (batch_size=64):**
```
batch_dict = {
    "pert_cell_emb":    [64, 11]      ← 64 perturbed cells' gene expression (TARGET)
    "ctrl_cell_emb":    [64, 11]      ← 64 control cells' gene expression (INPUT)
    "pert_emb":         [64, 64]      ← 64 perturbation one-hots (INPUT)
    "cell_type":        [64 strings]  ← Cell type names
    "cell_type_onehot": [64, 5]       ← Cell type one-hots
    "batch":            [64, 1]       ← Batch IDs
    "pert_name":        [64 strings]  ← Perturbation names
}
```

### **What This Means:**
```
Sample 0: ctrl_cell[452] from TARGET1 → MODEL → predict TARGET2 effect → compare with pert_cell[1001]
Sample 1: ctrl_cell[1203] from TARGET1 → MODEL → predict TARGET3 effect → compare with pert_cell[2453]
Sample 2: ctrl_cell[89] from TARGET1 → MODEL → predict TARGET2 effect → compare with pert_cell[1034]
...
Sample 63: ctrl_cell[1804] from TARGET1 → MODEL → predict TARGET5 effect → compare with pert_cell[8932]
```

---

## 🧠 Step 4: Model Forward Pass - Combining Data

### **Code:** `state_transition.py` lines 341-441

```python
def forward(self, batch: dict, padded=True) -> torch.Tensor:
    """
    Main forward pass - THIS IS WHERE DATA IS COMBINED!
    """
    
    # ========================================================================
    # STEP 4.1: Extract and Reshape Inputs
    # ========================================================================
    if padded:
        pert = batch["pert_emb"].reshape(-1, self.cell_sentence_len, self.pert_dim)
        # Shape: [B, S, pert_dim] where B=batch, S=sentence_len
        # Example: [1, 64, 64]
        
        basal = batch["ctrl_cell_emb"].reshape(-1, self.cell_sentence_len, self.input_dim)
        # Shape: [B, S, input_dim]
        # Example: [1, 64, 11]
    
    # ========================================================================
    # STEP 4.2: Encode Control Cells (Basal Encoder)
    # ========================================================================
    control_cells = self.encode_basal_expression(basal)
    # Input:  [1, 64, 11]    ← Control cell gene expression
    # Output: [1, 64, 128]   ← Control encoded in hidden space
    
    # This is a simple MLP:
    # control_cells = LayerNorm(GELU(Linear(basal)))
    
    # ========================================================================
    # STEP 4.3: Encode Perturbation (Pert Encoder)
    # ========================================================================
    pert_embedding = self.encode_perturbation(pert)
    # Input:  [1, 64, 64]    ← Perturbation one-hot
    # Output: [1, 64, 128]   ← Perturbation encoded in hidden space
    
    # This is a simple MLP:
    # pert_embedding = LayerNorm(GELU(Linear(pert)))
    
    # ========================================================================
    # STEP 4.4: COMBINE via Addition ⭐ KEY STEP!
    # ========================================================================
    combined_input = pert_embedding + control_cells
    # Shape: [1, 64, 128]
    # 
    # This is element-wise addition in hidden space:
    # combined[i] = control_cells[i] + pert_embedding[i]
    #             = "Where cell is" + "Direction to move"
    #             = "Modified cell state"
    
    seq_input = combined_input  # Shape: [1, 64, 128]
    
    # ========================================================================
    # STEP 4.5: Optional Batch Correction (if enabled)
    # ========================================================================
    if self.batch_encoder is not None:
        batch_embeddings = self.batch_encoder(batch_indices.long())
        seq_input = seq_input + batch_embeddings
        # seq_input = control + perturbation + batch_correction
    
    # ========================================================================
    # STEP 4.6: Transformer Processing
    # ========================================================================
    transformer_out = self.transformer(seq_input)
    # Input:  [1, 64, 128]
    # Output: [1, 64, 128]
    #
    # Transformer lets cells "attend" to each other
    # Each cell's representation is refined based on other cells in the batch
    
    # ========================================================================
    # STEP 4.7: Add Residual Connection
    # ========================================================================
    if self.predict_residual:
        # Add control cells back to transformer output
        transformer_out = transformer_out + control_cells
        # This means: predict CHANGE from control, not absolute state
    
    # ========================================================================
    # STEP 4.8: Output Projection (Decoder)
    # ========================================================================
    output = self.project_out(transformer_out)
    # Input:  [1, 64, 128]  ← Hidden space
    # Output: [1, 64, 11]   ← Gene expression space
    #
    # This is an MLP that translates from hidden space back to gene space
    # output = ReLU(Linear(GELU(Linear(transformer_out))))
    
    return output  # Shape: [1, 64, 11]
```

### **Visual Flow:**

```
Control cells [1, 64, 11]                Perturbation [1, 64, 64]
      ↓                                          ↓
Basal Encoder (MLP)                      Pert Encoder (MLP)
      ↓                                          ↓
control_encoded [1, 64, 128]            pert_encoded [1, 64, 128]
      └─────────────────┬──────────────────────┘
                        ↓
              ADDITION (element-wise)
                        ↓
              combined [1, 64, 128]
                        ↓
            (optional: + batch_embeddings)
                        ↓
            Transformer (self-attention)
                        ↓
              transformer_out [1, 64, 128]
                        ↓
          (if predict_residual: + control_encoded)
                        ↓
              Output Projection (MLP)
                        ↓
              prediction [1, 64, 11]
```

---

## 🎯 Step 5: Training Step - Computing Loss

### **Code:** `state_transition.py` lines 443-462

```python
def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, padded=True):
    """
    One training iteration
    """
    
    # ========================================================================
    # STEP 5.1: Forward Pass (get predictions)
    # ========================================================================
    pred = self.forward(batch, padded=padded)
    # Shape: [1, 64, 11]  ← Predicted perturbed cell gene expression
    
    # ========================================================================
    # STEP 5.2: Get Target (ground truth)
    # ========================================================================
    target = batch["pert_cell_emb"]
    # Shape: [1, 64, 11]  ← Actual perturbed cell gene expression
    
    # Reshape for loss computation
    pred = pred.reshape(-1, self.cell_sentence_len, self.output_dim)    # [1, 64, 11]
    target = target.reshape(-1, self.cell_sentence_len, self.output_dim) # [1, 64, 11]
    
    # ========================================================================
    # STEP 5.3: Compute Loss
    # ========================================================================
    main_loss = self.loss_fn(pred, target).nanmean()
    # Compare predicted vs actual perturbed cells
    # Loss functions: Energy distance, Sinkhorn, or Mixture-Kernel MMD
    # These are distributional losses (compare populations, not individual cells)
    
    # ========================================================================
    # STEP 5.4: Log Loss
    # ========================================================================
    self.log("train_loss", main_loss)
    
    # ========================================================================
    # STEP 5.5: Return Loss (PyTorch Lightning handles backward pass)
    # ========================================================================
    return main_loss  # Gradients computed automatically
```

---

## 🔁 Step 6: Complete Training Loop

```python
# Pseudo-code for training loop (handled by PyTorch Lightning)

for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(train_dataloader):
        
        # ====================================================================
        # 1. DATA LOADING
        # ====================================================================
        # DataLoader samples:
        #   - 64 perturbed cells (random from TARGET2-5)
        #   - For each, sample 1 control cell (from TARGET1, same cell_type)
        # Result: batch with matched pairs
        
        batch = {
            "ctrl_cell_emb":  [64, 11],   # Control cells (INPUT)
            "pert_cell_emb":  [64, 11],   # Perturbed cells (TARGET)
            "pert_emb":       [64, 64],   # Perturbation labels (INPUT)
            ...
        }
        
        # ====================================================================
        # 2. FORWARD PASS
        # ====================================================================
        # Encode control cells:       [64, 11] → [64, 128]
        # Encode perturbations:       [64, 64] → [64, 128]
        # Combine (addition):         [64, 128] + [64, 128] → [64, 128]
        # Transform (attention):      [64, 128] → [64, 128]
        # Add residual:               [64, 128] + [64, 128] → [64, 128]
        # Project to gene space:      [64, 128] → [64, 11]
        
        predictions = model.forward(batch)  # [64, 11]
        
        # ====================================================================
        # 3. LOSS COMPUTATION
        # ====================================================================
        targets = batch["pert_cell_emb"]    # [64, 11]
        loss = loss_fn(predictions, targets)
        
        # ====================================================================
        # 4. BACKPROPAGATION
        # ====================================================================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # ====================================================================
        # 5. LOGGING
        # ====================================================================
        print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")
```

---

## 💡 Key Insights: How Data is Combined

### **1. Control + Perturbation Pairing (Data Loading)**
```
random.h5ad → DataLoader samples 64 perturbed cells
            → For each, sample 1 control cell (same cell_type)
            → Result: 64 matched pairs
```

### **2. Encoding (Model Input Processing)**
```
Control cells [64, 11]     → Basal Encoder  → [64, 128] hidden
Perturbation [64, 64]      → Pert Encoder   → [64, 128] hidden
```

### **3. Combination (Addition in Hidden Space)**
```
combined = control_encoded + pert_encoded
         = [64, 128] + [64, 128]
         = [64, 128]

Interpretation:
  control_encoded[i]  = "Current state of cell i"
  pert_encoded[i]     = "Effect of perturbation"
  combined[i]         = "Modified state after perturbation"
```

### **4. Set-to-Set Learning (Transformer)**
```
Before: [cell_0, cell_1, ..., cell_63]  ← Independent
After:  [cell_0', cell_1', ..., cell_63'] ← Influenced by each other

Transformer allows cells to "talk" to each other:
- Some cells respond strongly to perturbation
- Some cells respond weakly
- Attention mechanism learns these relationships
```

### **5. Residual Prediction**
```
If predict_residual=True:
  output = transformer_output + control_encoded
         = "Change" + "Starting point"
         = "Final state"

This is easier to learn than predicting absolute values!
```

### **6. Output Projection**
```
Hidden space [64, 128] → Output Projection → Gene space [64, 11]
  Abstract features → Biological gene expression
```

---

## 📊 Summary Table: Data Shapes Through Training

| Step | Data | Shape | Description |
|------|------|-------|-------------|
| **File** | adata.obsm['X_hvg'] | [10000, 11] | All cells in file |
| **Sample** | One control+perturbed pair | 2 × [11] | One training sample |
| **Batch** | ctrl_cell_emb | [64, 11] | Batch of control cells |
| | pert_cell_emb | [64, 11] | Batch of perturbed cells (targets) |
| | pert_emb | [64, 64] | Batch of perturbation labels |
| **Reshape** | ctrl_cell_emb | [1, 64, 11] | [B, S, genes] |
| | pert_emb | [1, 64, 64] | [B, S, pert_dim] |
| **Encode** | control_encoded | [1, 64, 128] | Hidden representation |
| | pert_encoded | [1, 64, 128] | Hidden representation |
| **Combine** | combined | [1, 64, 128] | Addition |
| **Transform** | transformer_out | [1, 64, 128] | After attention |
| **Residual** | residual | [1, 64, 128] | + control_encoded |
| **Output** | prediction | [1, 64, 11] | Final prediction |
| **Loss** | Compare | [1, 64, 11] vs [1, 64, 11] | pred vs target |

---

## 🎯 Bottom Line

**How does ST model combine data for training?**

1. **Pairing**: Control + Perturbed cells paired by cell_type
2. **Encoding**: Both encoded to hidden space (128 dims)
3. **Combination**: Simple addition in hidden space
4. **Transformation**: Transformer refines with attention
5. **Residual**: Add control back (predict change)
6. **Projection**: Back to gene space (11 genes)
7. **Loss**: Compare with actual perturbed cells

**Key Innovation**: Set-to-set learning with optimal transport losses!

---

## 📁 Files to Read

- `src/state/tx/data/dataset/scgpt_perturbation_dataset.py` - Data loading
- `src/state/tx/models/state_transition.py` - Model forward & training
- `examples/mixed.toml` - Configuration
- `examples/random.h5ad` - Input data

You now understand the complete training flow! 🎉

