# Model Sequence Order: Complete Pipeline

## 🎯 The Correct Order

There are **TWO main workflows** depending on whether you use the SE (State Embedding) model:

---

## 📊 **WORKFLOW 1: Without SE Model (Direct HVG)**

### **Most Common Setup**

```
┌─────────────────────────────────────────────────────────────┐
│                    WORKFLOW 1: HVG → ST                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. RAW DATA (random.h5ad)                                  │
│     adata.obsm['X_hvg']                                     │
│     Control cells: [batch, 11 genes]                        │
│     Perturbation:  [batch, pert_dim]                        │
│                                                              │
│  2. ST MODEL (StateTransitionPerturbationModel)             │
│     ├─ Basal Encoder (encode control cells)                 │
│     │    [batch, 11] → [batch, 128]                         │
│     │                                                        │
│     ├─ Pert Encoder (encode perturbation)                   │
│     │    [batch, 64] → [batch, 128]                         │
│     │                                                        │
│     ├─ COMBINE (addition in hidden space)                   │
│     │    [batch, 128] + [batch, 128] → [batch, 128]        │
│     │                                                        │
│     ├─ TRANSFORMER (inside ST model!)                       │
│     │    [batch, 128] → [batch, 128]                        │
│     │    Cells attend to each other                         │
│     │                                                        │
│     ├─ Add Residual                                         │
│     │    [batch, 128] + control_encoded → [batch, 128]     │
│     │                                                        │
│     └─ Output Projection                                    │
│          [batch, 128] → [batch, 11 genes]                   │
│                                                              │
│  3. OUTPUT                                                   │
│     Predicted perturbed cells: [batch, 11 genes]            │
│     (Same space as input!)                                  │
│                                                              │
│  4. OPTIONAL: Gene Decoder (LatentToGeneDecoder)            │
│     [batch, 11] → [batch, 5000 all genes]                  │
│     Only if you want full transcriptome                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### **Key Points:**
- ✅ Input and output in **same HVG space**
- ✅ **Transformer is INSIDE the ST model** (not separate!)
- ✅ Gene decoder is **optional** (only if you want all genes)

---

## 📊 **WORKFLOW 2: With SE Model (Embeddings → ST)**

### **Using Pre-trained State Embeddings**

```
┌─────────────────────────────────────────────────────────────┐
│                 WORKFLOW 2: SE → ST → Decoder                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. RAW DATA (random.h5ad)                                  │
│     adata.X or counts                                       │
│     Raw gene expression: [batch, 5000+ genes]               │
│                                                              │
│  2. SE MODEL (State Embedding Model) - FIRST!               │
│     Raw expression → State embeddings                       │
│     [batch, 5000] → [batch, 1536]                          │
│                                                              │
│     This is a SEPARATE pre-trained model that:             │
│     • Compresses gene expression to fixed-size embeddings  │
│     • Trained with self-supervised learning                │
│     • Creates universal cell representations               │
│                                                              │
│     Result stored in: adata.obsm['X_state']                │
│                                                              │
│  3. ST MODEL (StateTransitionPerturbationModel)             │
│     Uses embeddings as input (not raw genes!)              │
│                                                              │
│     ├─ Basal Encoder                                        │
│     │    [batch, 1536] → [batch, 128]                       │
│     │                                                        │
│     ├─ Pert Encoder                                         │
│     │    [batch, 64] → [batch, 128]                         │
│     │                                                        │
│     ├─ COMBINE (addition)                                   │
│     │    [batch, 128] + [batch, 128] → [batch, 128]        │
│     │                                                        │
│     ├─ TRANSFORMER (inside ST model!)                       │
│     │    [batch, 128] → [batch, 128]                        │
│     │                                                        │
│     ├─ Add Residual                                         │
│     │    [batch, 128] + control_encoded → [batch, 128]     │
│     │                                                        │
│     └─ Output Projection                                    │
│          [batch, 128] → [batch, 1536 embeddings]            │
│                                                              │
│  4. GENE DECODER (LatentToGeneDecoder) - REQUIRED!          │
│     Translate embeddings back to genes                      │
│     [batch, 1536] → [batch, 5000 genes]                    │
│                                                              │
│  5. OUTPUT                                                   │
│     Predicted perturbed cells: [batch, 5000 genes]          │
│     Full gene expression!                                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### **Key Points:**
- ✅ SE model runs **FIRST** (preprocessing step)
- ✅ ST model works in **embedding space**, not gene space
- ✅ Gene decoder is **REQUIRED** to get back to genes
- ✅ Allows working with full transcriptome

---

## 🔍 **Important Clarification: Where is the Transformer?**

### **The Transformer is INSIDE the ST Model!**

```
ST Model Architecture:
┌──────────────────────────────────────────────┐
│        StateTransitionPerturbationModel      │
├──────────────────────────────────────────────┤
│                                              │
│  [Input] Control + Perturbation             │
│     ↓                                        │
│  Basal Encoder  +  Pert Encoder             │
│     ↓                                        │
│  Combined (addition)                         │
│     ↓                                        │
│  ┌────────────────────────────┐             │
│  │   TRANSFORMER (GPT-2 like) │  ← HERE!    │
│  │   • Multi-head attention   │             │
│  │   • Self-attention         │             │
│  │   • Cells attend to cells  │             │
│  └────────────────────────────┘             │
│     ↓                                        │
│  Add Residual                                │
│     ↓                                        │
│  Output Projection                           │
│     ↓                                        │
│  [Output] Predicted cells                   │
│                                              │
└──────────────────────────────────────────────┘
```

**NOT like this (common misconception):**
```
❌ WRONG:
   Data → SE Model → ST Model → Transformer → Decoder
                                    ↑
                          (Transformer is separate)

✅ CORRECT:
   Data → SE Model → ST Model (transformer inside) → Decoder
                         ↑
               (Transformer is inside ST model)
```

---

## 📋 **Complete Sequence Summary**

### **Workflow 1 (No SE):**
```
1. Load HVG data from random.h5ad
2. ST Model (contains transformer inside):
   a. Encode control cells
   b. Encode perturbation
   c. Combine (addition)
   d. Transformer attention
   e. Add residual
   f. Output projection
3. Output: predictions in HVG space
4. (Optional) Gene decoder: HVG → All genes
```

### **Workflow 2 (With SE):**
```
1. Load raw data from random.h5ad
2. SE Model: Raw genes → State embeddings (preprocessing)
3. ST Model (contains transformer inside):
   a. Encode control embeddings
   b. Encode perturbation
   c. Combine (addition)
   d. Transformer attention
   e. Add residual
   f. Output projection
4. Gene Decoder: Embeddings → All genes (required!)
5. Output: predictions in full gene space
```

---

## 🔄 **Data Flow Visualization**

### **With Your Data (random.h5ad) - Workflow 1:**

```
random.h5ad
    │
    ├─ Control cells: CT1 + TARGET1
    │  adata.obsm['X_hvg']: [400, 11]
    │
    ├─ Perturbed cells: CT1 + TARGET2
    │  adata.obsm['X_hvg']: [356, 11]
    │
    ↓
Sample batch (64 pairs)
    │
    ├─ ctrl_cell_emb: [64, 11]
    ├─ pert_emb: [64, 64]
    │
    ↓
ST MODEL
    │
    ├─ Encode both
    │  ctrl: [64, 11] → [64, 128]
    │  pert: [64, 64] → [64, 128]
    │
    ├─ Combine
    │  [64, 128] + [64, 128] = [64, 128]
    │
    ├─ TRANSFORMER (inside!)
    │  [64, 128] → [64, 128]
    │
    ├─ Output Projection
    │  [64, 128] → [64, 11]
    │
    ↓
Prediction: [64, 11]
    │
    ↓
Compare with target: [64, 11]
    │
    ↓
Loss & Train
```

---

## 🎓 **Common Misconceptions**

### ❌ **WRONG Order:**
```
SE → Gene Decoder → ST → Transformer
(Decoder before ST? No!)
(Transformer separate from ST? No!)
```

### ❌ **WRONG Order:**
```
Data → Transformer → ST Model → Decoder
(Transformer first? No! It's inside ST!)
```

### ✅ **CORRECT Order:**
```
WORKFLOW 1: Data → ST Model (has transformer inside) → [Optional: Decoder]
WORKFLOW 2: Data → SE Model → ST Model (has transformer inside) → Decoder (required)
```

---

## 🔍 **Code Evidence**

### **From `state_transition.py`:**

```python
class StateTransitionPerturbationModel(PerturbationModel):
    def __init__(...):
        # Encoders
        self.basal_encoder = build_mlp(...)
        self.pert_encoder = build_mlp(...)
        
        # TRANSFORMER IS HERE (inside ST model!)
        self.transformer = build_transformer(...)
        
        # Output projection
        self.project_out = build_mlp(...)
    
    def forward(self, batch):
        # 1. Encode
        control_cells = self.encode_basal_expression(basal)
        pert_embedding = self.encode_perturbation(pert)
        
        # 2. Combine
        combined_input = pert_embedding + control_cells
        
        # 3. TRANSFORMER (inside forward pass!)
        transformer_out = self.transformer(combined_input)
        
        # 4. Residual
        if self.predict_residual:
            transformer_out = transformer_out + control_cells
        
        # 5. Output projection
        output = self.project_out(transformer_out)
        
        return output
```

**See?** The transformer is called **INSIDE** the ST model's forward pass!

---

## 📊 **Timeline: When Each Component Runs**

### **Training Time:**

```
Time Step 1: (Before training, one-time)
    IF using SE embeddings:
        Run SE model on all data
        Save embeddings to adata.obsm['X_state']
    
Time Step 2: (Every training iteration)
    Load batch of data
    ↓
    ST Model forward pass:
        1. Encode inputs (basal & pert)
        2. Combine
        3. Transformer  ← runs here, inside ST
        4. Project output
    ↓
    IF decoder exists:
        Decoder forward pass:
            Latent → Genes
    ↓
    Compute loss
    ↓
    Backpropagate
    ↓
    Update weights
```

### **Inference Time:**

```
1. Load test data
2. IF using SE: apply SE model to get embeddings
3. ST Model forward (transformer inside)
4. IF decoder: apply decoder to get full genes
5. Return predictions
```

---

## ✅ **Final Answer to Your Question**

### **Question:** "Do embeddings go directly from SE model to gene decoder and then ST model and then transformer?"

### **Answer:**

**NO!** The correct order is:

1. **SE Model** (if used) → Creates embeddings
2. **ST Model** (contains transformer inside) → Processes embeddings
   - Transformer is **INSIDE** the ST model, not separate!
3. **Gene Decoder** (if needed) → Translates to full genes

**The transformer does NOT come after the ST model - it IS INSIDE the ST model!**

```
Correct Flow:
SE Model → ST Model → Gene Decoder
              ↑
        (Transformer is here, inside ST Model)

NOT:
SE Model → Gene Decoder → ST Model → Transformer
```

---

## 🎯 **Summary Table**

| Component | When It Runs | Input | Output | Location |
|-----------|--------------|-------|--------|----------|
| **SE Model** | Before training (preprocessing) | Raw genes [N, 5000] | Embeddings [N, 1536] | Separate model |
| **Basal Encoder** | During ST forward | Control cells [B, input_dim] | Encoded [B, 128] | Inside ST model |
| **Pert Encoder** | During ST forward | Pert labels [B, 64] | Encoded [B, 128] | Inside ST model |
| **Combine** | During ST forward | Two [B, 128] | Combined [B, 128] | Inside ST model |
| **Transformer** | During ST forward | Combined [B, 128] | Processed [B, 128] | **Inside ST model** |
| **Output Proj** | During ST forward | Processed [B, 128] | Predictions [B, out_dim] | Inside ST model |
| **Gene Decoder** | After ST forward | Predictions [B, latent] | Full genes [B, 5000] | Separate module |

---

**The key insight: The transformer is a COMPONENT INSIDE the ST model, not a separate step in the pipeline!** 🎯

