# LatentToGeneDecoder Architecture Explained

## 🎯 Purpose

The **LatentToGeneDecoder** is an optional component that transforms latent embeddings (from the ST model) back to **full gene expression space**.

---

## 🔍 Why Do We Need It?

### **Problem:**
The ST model typically works with:
- **Input**: Highly Variable Genes (HVG) - e.g., 11 genes or 2000 genes
- **Output**: Same HVG space - e.g., 11 genes or 2000 genes

But sometimes you want to predict **ALL genes** (e.g., 5000+ genes), not just HVGs!

### **Solution:**
The LatentToGeneDecoder translates from the HVG latent space to the full gene space.

```
ST Model prediction → [batch, 2000 HVGs] (latent/compressed space)
         ↓
LatentToGeneDecoder
         ↓
Full gene prediction → [batch, 5000+ genes] (complete gene space)
```

---

## 🏗️ Architecture

### **Two Variants:**

#### **1. Standard Decoder (residual_decoder=False)**
```
Input: [batch_size, latent_dim]
   ↓
Linear(latent_dim → hidden_dim1) → LayerNorm → GELU → Dropout
   ↓
Linear(hidden_dim1 → hidden_dim2) → LayerNorm → GELU → Dropout
   ↓
...more layers...
   ↓
Linear(last_hidden → gene_dim) → ReLU
   ↓
Output: [batch_size, gene_dim]
```

#### **2. Residual Decoder (residual_decoder=True)**
```
Input: [batch_size, latent_dim]
   ↓
Block 0: Linear → LayerNorm → GELU → Dropout
   ↓
Block 1: Linear → LayerNorm → GELU → Dropout + (Block 0 output)  ← Residual
   ↓
Block 2: Linear → LayerNorm → GELU → Dropout
   ↓
Block 3: Linear → LayerNorm → GELU → Dropout + (Block 2 output)  ← Residual
   ↓
...pattern continues...
   ↓
Linear(last_hidden → gene_dim) → ReLU
   ↓
Output: [batch_size, gene_dim]
```

**Pattern**: Odd blocks (1, 3, 5...) get residual connections from previous even blocks (0, 2, 4...)

---

## 📊 Input and Output

### **Input:**
```python
x: torch.Tensor
   Shape: [batch_size, latent_dim]
   Example: [64, 2000]  # 64 cells, 2000 HVGs
   
   Description:
   - Latent embeddings from the ST model
   - These are the model's predictions in HVG space
   - Or could be state embeddings from SE model
```

### **Output:**
```python
gene_predictions: torch.Tensor
   Shape: [batch_size, gene_dim]
   Example: [64, 5000]  # 64 cells, 5000 total genes
   
   Description:
   - Full gene expression predictions
   - Non-negative values (ReLU at the end)
   - Represents all genes, not just HVGs
```

---

## 🔧 Configuration Parameters

### **Constructor Arguments:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `latent_dim` | int | Required | Input dimension (HVG space or latent space) |
| `gene_dim` | int | Required | Output dimension (number of total genes) |
| `hidden_dims` | List[int] | [512, 1024] | Hidden layer dimensions |
| `dropout` | float | 0.1 | Dropout rate for regularization |
| `residual_decoder` | bool | False | Whether to use residual connections |

### **Example Configurations:**

```python
# Small dataset (< 10K genes)
decoder = LatentToGeneDecoder(
    latent_dim=2000,      # HVG dimension
    gene_dim=5000,        # Total genes
    hidden_dims=[1024, 1024, 512],
    dropout=0.1,
    residual_decoder=False
)

# Large dataset (> 10K genes)
decoder = LatentToGeneDecoder(
    latent_dim=2000,
    gene_dim=20000,
    hidden_dims=[1024, 512, 256],
    dropout=0.1,
    residual_decoder=False
)

# With residual connections (better for deep networks)
decoder = LatentToGeneDecoder(
    latent_dim=2000,
    gene_dim=5000,
    hidden_dims=[2048, 2048, 2048, 2048, 2048],  # 5 layers
    dropout=0.1,
    residual_decoder=True
)
```

---

## 🔄 How It's Used in Training

### **Complete Flow:**

```
STEP 1: ST Model Forward Pass
───────────────────────────────
Input: control cells [64, 2000 HVGs]
       + perturbation labels
       ↓
ST Model processes
       ↓
Output: latent predictions [64, 2000]

STEP 2: LatentToGeneDecoder (Optional)
───────────────────────────────────────
IF decoder is configured:
    Input: latent predictions [64, 2000]
           ↓
    LatentToGeneDecoder
           ↓
    Output: full gene predictions [64, 5000]

STEP 3: Loss Computation
─────────────────────────
Main Loss:
    Compare latent predictions [64, 2000] 
    vs HVG targets [64, 2000]

Decoder Loss (if decoder exists):
    Compare gene predictions [64, 5000]
    vs full gene targets [64, 5000]

Total Loss = Main Loss + decoder_loss_weight * Decoder Loss
```

### **Code From `state_transition.py` (lines 475-504):**

```python
def training_step(self, batch, batch_idx, padded=True):
    # Main model prediction
    pred = self.forward(batch, padded=padded)  # [batch, HVG_dim]
    target = batch["pert_cell_emb"]            # [batch, HVG_dim]
    
    # Main loss (on HVG space)
    main_loss = self.loss_fn(pred, target).nanmean()
    total_loss = main_loss
    
    # Optional decoder loss (on full gene space)
    if self.gene_decoder is not None and "pert_cell_counts" in batch:
        gene_targets = batch["pert_cell_counts"]  # [batch, gene_dim]
        
        # Decoder forward pass
        if self.detach_decoder:
            latent_preds = pred.detach()  # Don't backprop through main model
        else:
            latent_preds = pred           # Backprop through main model
        
        # Decode to full gene space
        gene_predictions = self.gene_decoder(latent_preds)  # [batch, gene_dim]
        
        # Compute decoder loss
        decoder_loss = self.loss_fn(gene_predictions, gene_targets).mean()
        
        # Add to total loss
        total_loss = total_loss + self.decoder_loss_weight * decoder_loss
    
    return total_loss
```

---

## 📐 Architecture Details

### **Layer-by-Layer (Standard Decoder):**

```
Example: latent_dim=2000, gene_dim=5000, hidden_dims=[1024, 1024, 512]

Layer 1:
  Linear(2000 → 1024)
  LayerNorm(1024)
  GELU()
  Dropout(0.1)
  
Layer 2:
  Linear(1024 → 1024)
  LayerNorm(1024)
  GELU()
  Dropout(0.1)
  
Layer 3:
  Linear(1024 → 512)
  LayerNorm(512)
  GELU()
  Dropout(0.1)
  
Output Layer:
  Linear(512 → 5000)
  ReLU()  ← Ensures non-negative gene expression

Total Parameters:
  Layer 1: 2000 × 1024 = 2,048,000
  Layer 2: 1024 × 1024 = 1,048,576
  Layer 3: 1024 × 512  = 524,288
  Output:  512 × 5000  = 2,560,000
  ──────────────────────────────────
  Total:                ~6.2M parameters
```

---

## 💡 Key Design Choices

### **1. Non-negative Output (ReLU)**
```python
layers.append(nn.ReLU())
```
- Gene expression values are always ≥ 0
- ReLU ensures predictions are non-negative

### **2. Layer Normalization**
```python
layers.append(nn.LayerNorm(hidden_dim))
```
- Stabilizes training
- Prevents gradient explosion/vanishing
- Normalizes across features for each sample

### **3. GELU Activation**
```python
layers.append(nn.GELU())
```
- Smooth, differentiable activation
- Better than ReLU for deep networks
- Popular in transformers and modern architectures

### **4. Dropout Regularization**
```python
layers.append(nn.Dropout(dropout))
```
- Prevents overfitting
- Randomly drops neurons during training
- Default: 10% dropout

### **5. Residual Connections (Optional)**
```python
if i >= 1 and i % 2 == 1:
    output = output + block_outputs[residual_idx]
```
- Helps gradient flow in deep networks
- Pattern: odd blocks get residual from previous even block
- Useful for very deep decoders (5+ layers)

---

## 🎓 When to Use the Decoder

### **Use LatentToGeneDecoder When:**

✅ You're using **State Embeddings (SE)** as input (e.g., X_state with 1536 dims)
   - ST model predicts in embedding space
   - Decoder converts to interpretable gene space

✅ You want predictions for **ALL genes**, not just HVGs
   - HVGs are typically 2000 genes
   - Full transcriptome might be 5000-20000 genes

✅ You need **biologically interpretable outputs**
   - Full gene predictions for downstream analysis
   - Gene set enrichment analysis (GSEA)
   - Pathway analysis

### **Don't Need Decoder When:**

❌ Already working in HVG space (X_hvg)
   - Model input and output are both HVGs
   - No need for additional decoding

❌ Only care about **relative changes**, not absolute values
   - Comparing perturbations
   - Ranking responses

❌ Computational constraints
   - Decoder adds parameters and compute time
   - May not be necessary for all applications

---

## 📊 Practical Example

### **Scenario: Using State Embeddings**

```python
# Your data
adata.obsm['X_state']  # [10000, 1536] - State embeddings from SE model
adata.X                # [10000, 5000] - Full gene expression (target)

# Configuration
latent_dim = 1536      # State embedding dimension
gene_dim = 5000        # Total genes

# Create decoder
decoder = LatentToGeneDecoder(
    latent_dim=1536,
    gene_dim=5000,
    hidden_dims=[1024, 1024, 512],
    dropout=0.1,
    residual_decoder=False
)

# Training forward pass
# Step 1: ST model predicts in latent space
latent_pred = st_model.forward(batch)  # [64, 1536]

# Step 2: Decoder translates to gene space
gene_pred = decoder(latent_pred)        # [64, 5000]

# Step 3: Compare with full gene expression
gene_target = batch["pert_cell_counts"] # [64, 5000]
decoder_loss = mse_loss(gene_pred, gene_target)
```

---

## 🔍 Comparison: With vs Without Decoder

### **Without Decoder:**
```
Control HVGs [64, 2000]
      ↓
 ST Model
      ↓
Predicted HVGs [64, 2000]
      ↓
Compare with Target HVGs [64, 2000]
      ↓
Loss & Train

✓ Simpler
✓ Faster
✓ Works in HVG space
✗ Can't predict full transcriptome
```

### **With Decoder:**
```
Control State Embeddings [64, 1536]
      ↓
 ST Model
      ↓
Predicted State Embeddings [64, 1536]
      ↓
LatentToGeneDecoder
      ↓
Predicted Full Genes [64, 5000]
      ↓
Compare with Target Full Genes [64, 5000]
      ↓
Loss & Train

✓ Full gene predictions
✓ Biologically interpretable
✓ Better for downstream analysis
✗ More parameters
✗ Slower training
```

---

## 🎯 Summary

### **What is LatentToGeneDecoder?**
A neural network that translates from latent/HVG space to full gene expression space.

### **Input:**
- Shape: `[batch_size, latent_dim]`
- Content: Latent embeddings or HVG predictions from ST model
- Example: `[64, 2000]` or `[64, 1536]`

### **Output:**
- Shape: `[batch_size, gene_dim]`
- Content: Full gene expression predictions (non-negative)
- Example: `[64, 5000]`

### **Architecture:**
- Multi-layer MLP with LayerNorm, GELU, and Dropout
- Optional residual connections for deep networks
- Final ReLU to ensure non-negative outputs

### **Use Cases:**
- Decoding state embeddings to genes
- Predicting full transcriptome (not just HVGs)
- Biologically interpretable predictions

**It's optional but powerful when you need full gene-level predictions!** 🎉

