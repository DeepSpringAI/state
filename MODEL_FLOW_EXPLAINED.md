# STATE Model Flow: Step-by-Step Explanation

This document explains what happens at each step when data flows through the STATE model, based on **actual data** from your dataset.

## 📊 Your Dataset Overview

From `examples/random.h5ad`:
- **Total cells**: 10,000
- **Gene dimension**: 11 (highly variable genes)
- **Perturbations**: 5 (TARGET1 through TARGET5)
- **Cell types**: 5 (CT1 through CT5)

**Key insight**: TARGET1 = Control (2,077 cells), others = Perturbed

---

## 🔄 Complete Data Flow

### Step 1: Raw Input Data

```
🔵 CONTROL CELLS (INPUT):
Shape: [4 cells, 11 genes]
Sample cell 0: [0.00, 0.74, 0.71, 0.84, 0.00, 0.50, 0.18, 0.49, 0.61, 0.00, ...]
Statistics: Mean=0.384, Std=0.502, Min=0.0, Max=2.064

🔴 PERTURBED CELLS (TARGET):
Shape: [4 cells, 11 genes]
Sample cell 0: [0.67, 1.85, 0.00, 0.00, 0.00, 0.00, 0.00, 1.32, 0.74, 0.00, ...]
Statistics: Mean=0.443, Std=0.572, Min=0.0, Max=1.846

📊 DIFFERENCE:
[+0.67, +1.10, -0.71, -0.84, 0.00, -0.50, -0.18, +0.83, +0.13, 0.00, ...]
Changed genes: 6/10 (with threshold > 0.5)
```

**What this shows:**
- Control and perturbed cells have DIFFERENT gene expression
- Some genes go up (↑), some go down (↓), some unchanged
- This is what the model needs to learn to predict

---

### Step 2: Encode Control Cells (Basal Encoder)

```
INPUT:  [4, 11] genes
        ↓ Linear(11 → 128) + LayerNorm + GELU
OUTPUT: [4, 128] hidden dimensions

Sample encoded: [-0.17, -0.13, 0.82, 0.61, -0.02, 0.47, 1.21, ...]
Statistics: Mean=0.294, Std=0.564
```

**What happened:**
- 11 gene expression values → compressed to 128-dim hidden space
- Each dimension captures abstract features (not individual genes)
- This representation is easier for the model to process

**Why:**
- Hidden space is more flexible for learning
- 128 dims can capture complex patterns in 11 genes
- Allows model to learn relationships between genes

---

### Step 3: Encode Perturbation (Pert Encoder)

```
INPUT:  [4, 64] one-hot encoding of perturbation
        First 10 dims: [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        ↓ Linear(64 → 128) + LayerNorm + GELU
OUTPUT: [4, 128] hidden dimensions

Sample encoded: [-0.16, -0.17, -0.09, -0.14, -0.17, 1.78, 0.98, ...]
Statistics: Mean=0.296, Std=0.558
```

**What this is:**
- Perturbation = One-hot encoding of "TARGET5" label
- **NOT** the perturbed cells themselves!
- Just an ID: "Which intervention to apply?"

**Why encode it:**
- Model learns: "What does TARGET5 do to cells?"
- Each perturbation gets its own learned representation
- After training, this vector captures the effect of TARGET5

---

### Step 4: COMBINE (Addition) ⭐ CRITICAL STEP

```
Control encoded:  [-0.17, -0.13,  0.82,  0.61, -0.02,  0.47,  1.21, ...]
Pert encoded:     [-0.16, -0.17, -0.09, -0.14, -0.17,  1.78,  0.98, ...]
                   ────────────────────────────────────────────────────
Combined (ADD):   [-0.33, -0.30,  0.74,  0.47, -0.19,  2.25,  2.19, ...]

Statistics: Mean=0.590, Std=0.815
```

**Why Addition (not concatenation)?**

```
Conceptually:
Control encoded     = "Current cell state"
Perturbation encoded = "Direction to move"
Combined            = "Modified cell state"

Like vectors:
Position[x, y, z] + Movement[Δx, Δy, Δz] = New_Position[x', y', z']
```

**Example interpretation:**
```
Control says: "Cell is in proliferative state [0.82, 0.61, ...]"
Perturbation says: "Drug reduces proliferation [-0.09, -0.14, ...]"
Combined: "Cell with reduced proliferation [0.74, 0.47, ...]"
```

---

### Step 5: Transformer Processing

```
INPUT:  [4, 128] combined representation
        ↓ TransformerEncoder (1 layer, 4 attention heads)
OUTPUT: [4, 128] refined representation

Sample output: [-1.15, -0.87, -0.45, 0.13, -0.95, 1.68, 2.14, ...]
Statistics: Mean=0.000, Std=1.001 (normalized by LayerNorm)
```

**What the Transformer does:**

```
BEFORE Transformer:
Cell 1: [0.74, 0.47, -0.19, ...]  → Isolated
Cell 2: [0.68, 0.52, -0.15, ...]  → Isolated
Cell 3: [0.71, 0.49, -0.18, ...]  → Isolated
Cell 4: [0.73, 0.48, -0.17, ...]  → Isolated

AFTER Transformer (with attention):
Cell 1: [-1.15, -0.87, -0.45, ...] → Influenced by all cells
Cell 2: [-1.08, -0.82, -0.42, ...] → Influenced by all cells
Cell 3: [-1.12, -0.85, -0.44, ...] → Influenced by all cells
Cell 4: [-1.14, -0.86, -0.45, ...] → Influenced by all cells
```

**Why this matters:**
- Cells don't respond independently to drugs
- Some cells respond strongly, some weakly
- Transformer learns: "If most cells show effect X, probably this cell too"
- This is **set-to-set learning** - the key innovation!

---

### Step 6: Add Residual Connection

```
Transformer output: [-1.15, -0.87, -0.45,  0.13, -0.95, ...]
Control encoded:    [-0.17, -0.13,  0.82,  0.61, -0.02, ...]
                     ─────────────────────────────────────────
Residual (ADD):     [-1.32, -1.00,  0.38,  0.74, -0.97, ...]
```

**Why add control back?**

```
Transformer output = "Change to apply"
Control encoded    = "Starting point"
Residual          = "Starting point + Change"

This implements: predict_residual = True
The model predicts CHANGE, not absolute state
```

**Benefit:**
- Easier to learn (changes are smaller than absolute values)
- More stable training
- Better generalization

---

### Step 7: Output Projection (Decoder)

```
INPUT:  [4, 128] hidden space
        ↓ Linear(128 → 128) + GELU + Linear(128 → 11) + ReLU
OUTPUT: [4, 11] gene space

Sample prediction: [0.33, 0.20, 0.28, 0.46, 0.00, 0.22, 0.20, ...]
Statistics: Mean=0.191, Std=0.189, Min=0.0, Max=0.63
```

**What this does:**
- Translates from "model language" (128 hidden dims)
- Back to "biology language" (11 gene expression values)
- ReLU ensures non-negative (gene expression ≥ 0)

**This is the FINAL PREDICTION!**

---

### Step 8: Compare with Target

```
Control (INPUT):      [0.00, 0.74, 0.71, 0.84, 0.00, 0.50, 0.18, 0.49, 0.61, 0.00]
Predicted:            [0.33, 0.20, 0.28, 0.46, 0.00, 0.22, 0.20, 0.05, 0.02, 0.00]
Target (GROUND TRUTH):[0.67, 1.85, 0.00, 0.00, 0.00, 0.00, 0.00, 1.32, 0.74, 0.00]

Error:
MSE: 0.446
MAE: 0.468
```

**Why prediction is bad:**
- This is a **RANDOM untrained model**!
- Just showing you the architecture
- After training on your data:
  - MSE should be < 0.05
  - Predictions should closely match targets

---

## 🎯 Key Answers to Your Questions

### 1. How and Why Combine?

**How:** Simple element-wise addition
```python
combined = control_encoded + pert_encoded
```

**Why:** Represents "current state + perturbation effect"
- Like: position + movement = new position
- Control is where you are, perturbation is where to go
- Combined is where you'll end up

### 2. What is Perturbation [B, S, pert]?

**It's the INTERVENTION (drug/gene), NOT the perturbed cells!**

```
perturbation = [0, 1, 0, 0, 0]  ← One-hot encoding of "TARGET2"
              This is an ID/label, not gene expression!

perturbed_cells = [1.8, 6.2, 0.5, ...]  ← Actual gene expression
                 These are the target outputs (separate!)
```

The perturbation tensor is just a **label** that the model learns to associate with specific effects.

### 3. What is Output Projection?

**Decoder that translates hidden space → gene space**

```
Hidden space (128 dims): Abstract features model understands
                ↓ Output Projection (MLP)
Gene space (11 dims):    Biological gene expression values
```

It's like translating from "AI language" to "biology language"

---

## 📈 Shape Transformations Summary

```
Control cells:        [4, 11]    Raw gene expression
  ↓ Basal Encoder
Control encoded:      [4, 128]   Hidden representation
  ↓
  + (Addition)
  ↓
Perturbation one-hot: [4, 64]    Perturbation ID
  ↓ Pert Encoder
Pert encoded:         [4, 128]   Hidden representation
  ↓
Combined:             [4, 128]   Control + Pert in hidden space
  ↓ Transformer
Transformed:          [4, 128]   Refined by attention
  ↓ Add residual
Residual:             [4, 128]   + Control back
  ↓ Output Projection
Predicted:            [4, 11]    Final gene expression prediction

Compare with:
Target:               [4, 11]    Ground truth perturbed cells
```

---

## 💡 The Big Picture

**What the model learns:**

1. **Encoders learn**: How to represent cells and perturbations in hidden space
2. **Combination learns**: How perturbations modify cell states
3. **Transformer learns**: How cells influence each other's responses
4. **Decoder learns**: How to translate back to gene expression

**After training:**
- Given: Any control cell + Any perturbation
- Predict: What the perturbed cell will look like
- Even for NEW perturbations not seen during training! (generalization)

**Your data has:**
- 2,077 control cells (TARGET1)
- ~2,000 cells for each perturbation (TARGET2-5)
- The model learns patterns from these examples
- Then predicts responses to new perturbations

---

## 🔬 Next Steps

To see how training improves predictions:

1. **Run training**:
   ```bash
   make train STEPS=5000
   ```

2. **Check results**:
   ```bash
   state tx predict --output-dir ./mixed_for_competition/unified_model/
   ```

3. **Re-run inspection** with trained model to see better predictions!

The script `inspect_model_flow.py` is saved for your reference. You can modify it to inspect trained models too!


