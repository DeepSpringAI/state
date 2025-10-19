# STATE Codebase Overview

## What Does This Code Do?

**STATE** (State Transition Artificial intelligence for Estimating cellular responses) is a machine learning framework for **predicting how cells respond to perturbations** (genetic modifications or drug treatments) across diverse biological contexts.

### Main Purpose
Given:
- A set of **control (unperturbed) cells**
- A **perturbation** (gene knockout, drug treatment, etc.)
- A **cell type** context

The model predicts:
- The **gene expression changes** that would occur if those cells were exposed to that perturbation
- How cellular state transitions from control → perturbed

This is valuable for:
- 🧬 **Drug Discovery**: Predict drug effects without experiments
- 🔬 **Gene Function**: Understand what genes do when disrupted
- 🏥 **Precision Medicine**: Predict patient-specific responses
- 🧪 **Virtual Cell Challenge**: Compete on predictive accuracy

---

## Architecture Overview

STATE consists of **two main models**:

### 1. State Embedding (SE) Model
**Purpose:** Creates rich, meaningful embeddings of single cells

- **Input:** Raw gene expression data (single cells × genes)
- **Architecture:** Transformer-based encoder with protein embeddings
- **Output:** High-dimensional cell embeddings (normalized)
- **Use Cases:** Cell similarity search, cell type annotation, cross-dataset comparison

### 2. State Transition (ST) Model ⭐ (Primary Focus)
**Purpose:** Predicts cellular responses to perturbations

- **Input:** Control cells + perturbation information
- **Architecture:** Neural optimal transport with transformer backbone
- **Output:** Predicted perturbed cell gene expression
- **Use Cases:** Drug effect prediction, gene knockout simulation, virtual experiments

---

## State Embedding (SE) Model: Deep Dive

### What Does the SE Model Do?

The SE model creates **universal cell embeddings** - like creating a "passport photo" for each cell that captures its identity and state in a fixed-size vector.

#### Purpose
**Compress a cell's entire biological state into a fixed-size vector**

```python
# Input: A cell with 19,000 gene expression values
cell_input = [0.0, 2.3, 0.1, 5.7, ..., 1.2]  # 19,000 genes

# SE Model processes it
↓

# Output: A rich 1536-dimensional embedding
cell_embedding = [0.15, -0.82, 0.31, ..., 0.67]  # 1536 values

# This embedding captures:
# - Cell type (e.g., "This is a K562 cell")
# - Cell state (e.g., "This cell is actively dividing")
# - Biological features (e.g., "High glycolysis, low oxidative phosphorylation")
```

#### Key Characteristics

1. **Universal**: Works across different datasets, experiments, labs
2. **Normalized**: Cells from different sources are comparable
3. **Searchable**: Can find similar cells using vector similarity
4. **Transfer Learning**: Pre-trained on millions of cells

---

### SE Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              State Embedding (SE) Model                      │
│                                                               │
│  Gene Expression                                             │
│  [19,000 genes]                                              │
│         ↓                                                    │
│  ┌──────────────────────────┐                               │
│  │ 1. Gene → Token          │                               │
│  │    Conversion            │                               │
│  │ + ESM2 Protein Embeddings│                               │
│  └──────────┬───────────────┘                               │
│             ↓                                                │
│  ┌──────────────────────────┐                               │
│  │ 2. Token Encoder         │                               │
│  │    (Linear + LayerNorm)  │                               │
│  └──────────┬───────────────┘                               │
│             ↓                                                │
│  ┌──────────────────────────┐                               │
│  │ 3. [CLS] Token           │                               │
│  │    (like BERT)           │                               │
│  └──────────┬───────────────┘                               │
│             ↓                                                │
│  ┌──────────────────────────┐                               │
│  │ 4. Transformer Encoder   │                               │
│  │    (Multiple layers)     │                               │
│  │    Self-attention between│                               │
│  │    gene tokens           │                               │
│  └──────────┬───────────────┘                               │
│             ↓                                                │
│  ┌──────────────────────────┐                               │
│  │ 5. Extract CLS embedding │                               │
│  └──────────┬───────────────┘                               │
│             ↓                                                │
│  ┌──────────────────────────┐                               │
│  │ 6. Decoder               │                               │
│  │    (Project to output_dim)│                              │
│  └──────────┬───────────────┘                               │
│             ↓                                                │
│  ┌──────────────────────────┐                               │
│  │ 7. L2 Normalization      │                               │
│  └──────────┬───────────────┘                               │
│             ↓                                                │
│  Cell Embedding                                              │
│  [1536 dimensions]                                           │
│  (Normalized vector)                                         │
└─────────────────────────────────────────────────────────────┘
```

#### Architecture Components

##### 1. Protein Embeddings (ESM2)

The SE model uses **pre-trained protein language models** (ESM2) to represent genes:

```python
# ESM2 is like BERT for proteins
# Each gene → represented by its protein embedding

gene_BRCA1 → ESM2_embedding[5120 dims]
gene_TP53  → ESM2_embedding[5120 dims]
...

# These capture protein structure, function, evolutionary relationships
```

**Why ESM2?** Genes with similar functions have similar protein embeddings, even if their sequences differ. This provides biological priors to the model.

##### 2. Token Encoding

```python
# Each gene becomes a "token" (like a word in a sentence)
self.encoder = nn.Sequential(
    nn.Linear(5120, d_model),      # ESM2 embedding size → transformer size
    nn.LayerNorm(d_model),
    nn.SiLU(),
)

# Process:
for each gene in cell:
    expression_value = X[gene_idx]              # e.g., 2.3
    protein_embedding = ESM2[gene_idx]          # [5120 dims]
    token = expression_value * protein_embedding # Weighted by expression
    encoded_token = encoder(token)              # [d_model dims]
```

This creates a **"sentence"** where each gene is a "word" represented by its protein embedding weighted by expression level.

##### 3. CLS Token (Classification Token)

```python
# Like BERT's [CLS] token - aggregates information
self.cls_token = nn.Parameter(torch.randn(1, token_dim))

# Prepended to the sequence:
sequence = [CLS_token, gene1_token, gene2_token, ..., gene_N_token]
```

The CLS token **gathers information** from all genes into a single cell-level representation.

##### 4. Transformer Encoder

```python
# Multiple layers of self-attention
self.transformer_encoder = FlashTransformerEncoder(layers)

# What happens:
# - Each gene token attends to all other gene tokens
# - Captures gene-gene interactions and co-expression patterns
# - CLS token aggregates information from all genes
# - Output: contextualized representations
```

##### 5. Decoder & Normalization

```python
self.decoder = nn.Sequential(
    SkipBlock(d_model),                 # Residual connection
    nn.Linear(d_model, output_dim),     # Project to final embedding
)

# Then L2 normalize for similarity search:
embedding = F.normalize(embedding, p=2, dim=-1)
```

---

### How is SE Model Trained?

#### Training Objective: Self-Supervised Learning

**Key Insight**: Unlike ST model (which needs perturbation labels), SE model is trained **WITHOUT manual labels** using self-supervision!

#### Training Process

##### 1. Data Preparation

```python
# Training data: Millions of cells from many datasets
training_data = {
    'Human Cell Atlas':  [millions of cells],
    'Tabula Sapiens':    [millions of cells],
    'Allen Brain Atlas': [millions of cells],
    'CellXGene':        [millions of cells],
    'Perturbation datasets': [thousands of cells],
    ...
}

# Each cell: [19,000 genes] expression values
```

##### 2. Loss Functions (Multiple Objectives)

**a. Gene Expression Reconstruction**
```python
criterion = BCEWithLogitsLoss()  # Binary cross-entropy

# Task: Given a cell embedding, predict gene expression pattern
# Forces embedding to capture gene expression information
```

**b. Maximum Mean Discrepancy (MMD)**
```python
criterion = MMDLoss(kernel="energy")

# Task: Make cell embeddings match the distribution of a reference
# Forces embeddings to have good statistical properties
# Ensures smooth, well-distributed embedding space
```

**c. Contrastive Learning**
```python
# Implicit in the architecture:
# - Similar cells should have similar embeddings (close in space)
# - Different cells should have different embeddings (far apart)

# Achieved by:
# 1. Augmenting cells (dropout some genes, add noise)
# 2. Ensure augmented versions are close
# 3. Ensure different cells are far apart
```

**d. Dataset Correction (Optional)**
```python
self.dataset_loss = nn.CrossEntropyLoss()

# Task: Predict which dataset a cell comes from
# But train adversarially - embedding should NOT reveal dataset
# Forces cross-dataset consistency (dataset-invariant features)
```

##### 3. Training Loop

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # 1. Get gene expression
        gene_expression = batch['X']  # [B, 19000]
        
        # 2. Convert to tokens using ESM2
        tokens = gene_expression * ESM2_embeddings  # [B, 19000, 5120]
        
        # 3. Add CLS token
        tokens = torch.cat([CLS_token, tokens], dim=1)
        
        # 4. Transformer encoding
        hidden_states = transformer_encoder(tokens)  # [B, seq_len, d_model]
        
        # 5. Extract CLS embedding
        cls_output = hidden_states[:, 0, :]  # [B, d_model]
        
        # 6. Decode
        cell_embedding = decoder(cls_output)  # [B, output_dim]
        
        # 7. Normalize
        cell_embedding = F.normalize(cell_embedding, p=2, dim=-1)
        
        # 8. Compute loss (e.g., reconstruction)
        predicted_expression = binary_decoder(cell_embedding)
        loss = BCEWithLogitsLoss(predicted_expression, gene_expression)
        
        # 9. Backprop
        loss.backward()
        optimizer.step()
```

##### 4. Key Training Features

- **Self-supervised**: No manual labels needed!
- **Large-scale**: Trained on millions of cells
- **Multi-dataset**: Learns cross-dataset representations
- **Protein-informed**: Uses ESM2 biological priors
- **Contrastive**: Similar cells cluster together

#### Training Data Sources

```python
training_corpora = [
    "Human Cell Atlas",          # Millions of human cells across tissues
    "Tabula Sapiens",           # Cross-tissue human atlas
    "Allen Brain Atlas",        # Brain cells
    "CellXGene",               # Public cell databases
    "Perturbation datasets",   # Replogle, Norman, etc.
]

# Result: A universal cell embedding model that works across:
# - Different tissues
# - Different cell types  
# - Different experimental conditions
# - Different labs/protocols
```

---

### What the Model Learns

#### Embedding Space Properties

```python
# After training, cells cluster by biology, not by dataset:

Embedding Space (1536 dimensions, visualized in 2D):

    T-cells ●●●●
            ●●●●
                    B-cells ●●●
                            ●●●
                            
    Neurons ■■■■
            ■■■■
            
                    Hepatocytes ▲▲▲
                                ▲▲▲
                                
# Properties:
# 1. Same cell type from different datasets cluster together
# 2. Similar cell types are nearby (e.g., CD4+ and CD8+ T-cells)
# 3. Developmental trajectories are smooth paths
# 4. Can do math: embedding(activated_T) - embedding(resting_T) = "activation vector"
```

#### Vector Operations

```python
# The embeddings support semantic operations:

# 1. Cell similarity
similarity = cosine_similarity(cell1_emb, cell2_emb)

# 2. Cell type arithmetic  
T_cell_emb ≈ immune_emb + lymphoid_emb - myeloid_emb

# 3. State transitions
activated_emb - resting_emb = "activation_vector"

# 4. Query similar cells
query = "K562 cell in G2/M phase"
results = vector_database.search(query_embedding, top_k=100)
```

---

### SE Model Usage

#### Training (Usually done by Arc Institute)

```bash
# Training from scratch
state emb fit \
  --conf embedding_config.yaml \
  --data-dir /path/to/millions/of/cells
  
# Takes days/weeks on multiple GPUs
# Produces: SE-600M model (600 million parameters)
```

#### Using Pre-trained SE Model (What you would do)

```bash
# Transform your data using pre-trained model
state emb transform \
  --model-folder SE-600M \
  --input your_data.h5ad \
  --output your_data_embedded.h5ad
  
# Takes minutes, produces embeddings in adata.obsm['X_state']
```

#### Query Similar Cells

```bash
# Build vector database
state emb build-db \
  --input atlas_embedded.h5ad \
  --output atlas.lancedb

# Query for similar cells
state emb query \
  --lancedb atlas.lancedb \
  --input query_cells.h5ad \
  --output similar_cells.csv \
  --top-k 100
```

---

### SE Model vs ST Model Comparison

| Aspect | State Embedding (SE) | State Transition (ST) |
|--------|---------------------|---------------------|
| **Purpose** | Universal cell representation | Perturbation effect prediction |
| **Training Data** | Just cells (no labels) | Cells + perturbations (labeled) |
| **Learning Type** | Self-supervised | Supervised |
| **Training Scale** | Millions of cells | 10k-100k cell pairs |
| **Objective** | Encode cell identity | Predict response to perturbation |
| **Loss Function** | Reconstruction, MMD, Contrastive | Optimal transport (Energy/Sinkhorn) |
| **Output** | Fixed cell embedding (1536 dims) | Variable (depends on perturbation) |
| **Use Case** | Cell similarity, clustering, annotation | Drug prediction, gene knockout simulation |
| **Pre-training** | Yes (like BERT) | Optional (can use SE embeddings) |
| **Inference Speed** | Fast (~1000 cells/sec) | Moderate (depends on cell_set_len) |

---

### Why Two Models?

#### The Two-Stage Philosophy

**SE Model (Stage 1): Learn Universal Cell Biology**
```python
# Trained once on massive data
# Learns: "What makes a cell a cell?"
# Captures: Cell types, states, developmental trajectories
# Output: Universal cell representations
```

**ST Model (Stage 2): Learn Perturbation Effects**
```python
# Trained on task-specific data
# Learns: "How do cells change under perturbations?"
# Input: SE embeddings (optional) or raw genes
# Output: Predicted perturbed cells
```

#### Analogy

- **SE Model** = Learning English vocabulary and grammar (foundation)  
- **ST Model** = Learning to write scientific papers (specific task)

You **can** learn to write papers without grammar (direct approach), but it's **easier** with language foundation (transfer learning)!

---

### Two Workflows: With and Without SE

#### Workflow 1: Direct (ST Model Alone) - **Current Standard**

```python
# Use raw gene expression directly
state tx train \
  data.kwargs.embed_key=X_hvg    # Read HVGs directly

# Flow:
Raw Genes [2000] → Basal Encoder → Hidden [696] → Transformer → Output
```

**Pros:**
- ✅ Simpler pipeline
- ✅ Task-specific optimization
- ✅ No dependency on SE model

**Cons:**
- ❌ Basal Encoder learns from scratch
- ❌ Needs more training data
- ❌ Less cross-dataset generalization

#### Workflow 2: Transfer Learning (SE → ST)

```python
# Step 1: Create SE embeddings
state emb transform \
  --model-folder SE-600M \
  --input data.h5ad \
  --output data_embedded.h5ad

# Step 2: Train ST on embeddings
state tx train \
  data.kwargs.embed_key=X_state    # Use SE embeddings

# Flow:
Raw Genes → SE Model → Embeddings [1536] → Basal Encoder → Hidden [696] → Output
```

**Pros:**
- ✅ Better with limited data
- ✅ Better cross-dataset generalization
- ✅ Faster ST training
- ✅ Leverages biological priors

**Cons:**
- ❌ Additional preprocessing step
- ❌ Dependency on SE model
- ❌ Slightly more complex

---

### Key Relationship: Control Cells and Embeddings

**Critical Understanding:**

```python
# SE embeddings REPLACE raw genes as input to ST model

# Without SE:
Basal Encoder receives: raw gene expression [2000 genes]
Input values: [2.3, 5.7, 1.2, ..., 3.4]  # Expression values

# With SE:
Basal Encoder receives: SE embeddings [1536 dims]
Input values: [0.15, -0.82, 0.31, ..., 0.67]  # Rich representations

# The embed_key parameter controls which gets used:
embed_key = "X_hvg"    → Uses raw genes (adata.obsm['X_hvg'])
embed_key = "X_state"  → Uses SE embeddings (adata.obsm['X_state'])
```

**Data Flow with SE:**

```
SE Model Output (X_state embeddings)
         ↓
    [Stored in adata.obsm['X_state']]
         ↓
    ST Data Loader reads it (based on embed_key)
         ↓
    batch["ctrl_cell_emb"] = X_state
         ↓
    ST Basal Encoder processes it
         ↓
    Rest of ST model continues...
```

---

## Understanding Data Formats: `obsm['X_hvg']` vs `obsm['X_state']`

This is a critical distinction that often causes confusion. Let me explain the key differences between these two data representations.

### Quick Answer

```python
adata.obsm['X_hvg']    # Raw/processed gene expression (biology data)
adata.obsm['X_state']  # SE model embeddings (learned representation)
```

### Detailed Comparison

#### `adata.obsm['X_hvg']` - Highly Variable Genes

**What it is:** Filtered and normalized gene expression values

```python
# Example values:
adata.obsm['X_hvg'][0, :]  # Cell #0
# [2.31, 0.05, 5.72, 0.00, 1.23, 3.45, ..., 0.89]
#  ↑     ↑     ↑     ↑     ↑     ↑          ↑
# Gene1 Gene2 Gene3 Gene4 Gene5 Gene6  ... Gene2000

# Each value = expression level of a specific gene
# High value = gene is highly expressed
# Low/zero value = gene is lowly/not expressed
```

**Key Characteristics:**

| Property | Details |
|----------|---------|
| **Dimensions** | [N cells × ~2000 genes] |
| **Values** | Real numbers (log-normalized counts) |
| **Range** | Typically 0-10 (after log1p transform) |
| **Meaning** | Each column = specific gene's expression |
| **Interpretable** | ✅ Yes! Each dimension has biological meaning |
| **Type** | Raw biological measurement |
| **Creation** | From wet-lab sequencing + preprocessing |

**How It's Created:**

```python
# Preprocessing pipeline:
import scanpy as sc

# 1. Start with raw counts
adata = sc.read_h5ad("raw_data.h5ad")
# adata.X: [10000 cells × 19000 genes] - all genes

# 2. Normalize (counts per cell)
sc.pp.normalize_total(adata, target_sum=1e4)

# 3. Log transform
sc.pp.log1p(adata)

# 4. Select highly variable genes (most informative)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)

# 5. Store HVGs
adata.obsm['X_hvg'] = adata[:, adata.var['highly_variable']].X
# Result: [10000 cells × 2000 HVGs]
```

**What It Represents:**

```python
# Biological interpretation:
cell_42_hvg = adata.obsm['X_hvg'][42, :]

# You can say:
"Cell 42 has:"
- High expression of BRCA1 (value: 5.2)
- Medium expression of TP53 (value: 3.1)
- Low expression of MYC (value: 0.8)
- No expression of CD4 (value: 0.0)

# Direct biological meaning! ✅
```

---

#### `adata.obsm['X_state']` - SE Model Embeddings

**What it is:** Learned representation from the State Embedding model

```python
# Example values:
adata.obsm['X_state'][0, :]  # Cell #0
# [0.15, -0.82, 0.31, -0.45, 0.67, -0.23, ..., 0.91]
#  ↑      ↑      ↑      ↑      ↑      ↑          ↑
# Dim1  Dim2   Dim3   Dim4   Dim5   Dim6   ... Dim1536

# Each value = learned feature (NOT a specific gene!)
# High value = that abstract feature is present
# Negative value = feature is absent/opposite
```

**Key Characteristics:**

| Property | Details |
|----------|---------|
| **Dimensions** | [N cells × 1536 dimensions] |
| **Values** | Real numbers (usually -1 to +1 after normalization) |
| **Range** | Typically -2 to +2 (L2 normalized) |
| **Meaning** | Each dimension = learned abstract feature |
| **Interpretable** | ❌ No! Dimensions have no direct biological meaning |
| **Type** | Learned representation (like word2vec) |
| **Creation** | From SE model (deep learning) |

**How It's Created:**

```python
# SE model processing:
state emb transform \
  --model-folder SE-600M \
  --input data.h5ad \
  --output data_embedded.h5ad

# What happens internally:
# 1. SE model reads: adata.X [N × 19000]
# 2. Processes through:
#    - Token encoding
#    - Transformer layers
#    - Attention mechanisms
#    - Learned projections
# 3. Outputs: adata.obsm['X_state'] [N × 1536]
```

**What It Represents:**

```python
# Abstract interpretation:
cell_42_state = adata.obsm['X_state'][42, :]

# You CANNOT say:
"Dimension 5 = TP53 expression" ❌ Wrong!

# You CAN say:
"This 1536-d vector captures:"
- Overall cell identity (T-cell vs neuron)
- Cell state (activated vs resting)
- Biological processes (glycolysis level, cell cycle phase)
- Similarity to other cells

# But individual dimensions are NOT interpretable! ⚠️
```

---

### Side-by-Side Example

Let's look at the same cell in both representations:

#### Cell #42 (K562 cell, untreated)

**`adata.obsm['X_hvg']` - Gene Expression**

```python
# [2000 values, one per gene]
[
    2.31,  # Gene 0: ACTB (high - housekeeping gene)
    0.05,  # Gene 1: CD3D (low - not a T-cell)
    5.72,  # Gene 2: MYC (high - cancer-related)
    0.00,  # Gene 3: INS (none - not pancreatic)
    1.23,  # Gene 4: GAPDH (medium - housekeeping)
    ...
    3.45   # Gene 1999: BCL2 (medium - cancer-related)
]

# Interpretation:
# - This is a cancer cell (high MYC, BCL2)
# - Not a T-cell (no CD3D)
# - Metabolically active (high ACTB, GAPDH)
```

**`adata.obsm['X_state']` - SE Embeddings**

```python
# [1536 values, learned features]
[
    0.15,   # Dim 0: ??? (learned feature A)
   -0.82,   # Dim 1: ??? (learned feature B)
    0.31,   # Dim 2: ??? (learned feature C)
   -0.45,   # Dim 3: ??? (learned feature D)
    0.67,   # Dim 4: ??? (learned feature E)
    ...
    0.91    # Dim 1535: ??? (learned feature ZZZ)
]

# Interpretation:
# - Individual dimensions are NOT interpretable
# - But as a whole vector, it captures:
#   * Cell type (K562)
#   * Cell state (untreated)
#   * Biological similarity to other cells
#
# Use for: similarity search, clustering, etc.
```

---

### Visualization of Differences

#### Dimension Meanings

```
X_hvg (Gene Expression):
┌─────────┬─────────┬─────────┬─────────┐
│ ACTB    │  MYC    │  TP53   │  BCL2   │ ... (2000 genes)
│ (2.31)  │ (5.72)  │ (3.10)  │ (3.45)  │
└─────────┴─────────┴─────────┴─────────┘
    ↑         ↑         ↑         ↑
  Known   Known     Known     Known
  Gene    Gene      Gene      Gene

Each dimension = Specific gene with known function


X_state (SE Embeddings):
┌─────────┬─────────┬─────────┬─────────┐
│  Dim0   │  Dim1   │  Dim2   │  Dim3   │ ... (1536 dims)
│ (0.15)  │(-0.82)  │ (0.31)  │(-0.45)  │
└─────────┴─────────┴─────────┴─────────┘
    ↑         ↑         ↑         ↑
 Unknown  Unknown   Unknown   Unknown
 Feature  Feature   Feature   Feature

Each dimension = Abstract learned feature (not interpretable)
```

#### Usage Patterns

```
X_hvg → Direct Analysis:
┌────────────────────────────────────┐
│ • Differential expression          │
│ • Gene set enrichment              │
│ • Biological interpretation        │
│ • Validate specific hypotheses     │
│ • Generate hypotheses              │
└────────────────────────────────────┘

X_state → Machine Learning:
┌────────────────────────────────────┐
│ • Similarity search                │
│ • Clustering                       │
│ • Classification                   │
│ • Transfer learning (to ST model)  │
│ • Cross-dataset integration        │
└────────────────────────────────────┘
```

---

### Key Differences Summary

| Aspect | `obsm['X_hvg']` | `obsm['X_state']` |
|--------|-----------------|-------------------|
| **What** | Gene expression values | Learned embeddings |
| **Dimensions** | 2000 (genes) | 1536 (abstract features) |
| **Interpretable?** | ✅ Yes (each gene has meaning) | ❌ No (learned features) |
| **Created by** | Wet-lab + preprocessing | SE model (deep learning) |
| **Values mean** | Expression level of specific gene | Strength of abstract feature |
| **Used for** | Biology analysis, ST model input | Similarity search, ST model input |
| **Dimension names** | Gene names (ACTB, TP53, etc.) | Just numbers (0-1535) |
| **Cross-dataset** | ❌ Batch effects present | ✅ Normalized across datasets |
| **File size** | Smaller (2000 dims) | Larger (1536 dims) |
| **Always present?** | ✅ Usually yes | ❌ Only if SE model run |

---

### When to Use Which?

#### Use `X_hvg` (Gene Expression) When:

```python
✅ You want to:
- Understand which genes are up/down-regulated
- Do differential expression analysis
- Validate biological hypotheses
- Interpret results biologically
- Train ST model (standard approach)
- Don't have access to SE model

Example:
"Which genes change when we knock out TP53?"
→ Compare X_hvg between control and TP53-knockout cells
```

#### Use `X_state` (SE Embeddings) When:

```python
✅ You want to:
- Find similar cells across different datasets
- Cluster cells with better cross-dataset generalization
- Train ST model with transfer learning
- Search for similar cells in a large atlas
- Integrate data from different labs/protocols

Example:
"Find cells similar to this rare cell type across all datasets"
→ Search X_state embeddings in vector database
```

---

### How They Relate

```python
# Both can be present in the same file:
adata = sc.read_h5ad("data_with_both.h5ad")

print(adata.obsm['X_hvg'].shape)    # (10000, 2000) - genes
print(adata.obsm['X_state'].shape)  # (10000, 1536) - embeddings

# Different views of the same cells:
# X_hvg  → Biological view (what genes are expressed?)
# X_state → Abstract view (what is this cell similar to?)

# Both capture cell information, but differently!
```

#### Relationship:

```
Raw Genes [19000]
     ↓ (preprocessing)
X_hvg [2000] ──────────────→ Used for biology analysis
     ↓                       Used as ST model input
     ↓ (SE Model)
     ↓
X_state [1536] ─────────────→ Used for similarity search
                               Used as ST model input (optional)
```

---

### Practical Example

```python
import scanpy as sc
import numpy as np

# Load data
adata = sc.read_h5ad("examples/random.h5ad")

# Check what you have:
print("Available obsm keys:", adata.obsm.keys())
# Output: ['X_hvg']  # Only gene expression

# Look at gene expression
cell_0_genes = adata.obsm['X_hvg'][0, :]
print(f"X_hvg shape: {cell_0_genes.shape}")  # (2000,)
print(f"X_hvg values: {cell_0_genes[:5]}")   # [2.31, 0.05, 5.72, 0.00, 1.23]
print("These are gene expression levels! ✅")

# If you had run SE model:
# state emb transform --input random.h5ad --output random_embedded.h5ad
# Then you'd also have:

# adata = sc.read_h5ad("random_embedded.h5ad")
# cell_0_embedding = adata.obsm['X_state'][0, :]
# print(f"X_state shape: {cell_0_embedding.shape}")  # (1536,)
# print(f"X_state values: {cell_0_embedding[:5]}")   # [0.15, -0.82, 0.31, -0.45, 0.67]
# print("These are learned embeddings! ✅")
```

---

### Bottom Line

**`obsm['X_hvg']`** = 🧬 **Biology** (genes you can look up in a textbook)
- "Cell expresses high MYC and TP53"

**`obsm['X_state']`** = 🤖 **AI representation** (learned patterns from millions of cells)
- "Cell embedding is [0.15, -0.82, 0.31, ...]"

Both represent cells, just in different "languages"! 🗣️

---

## State Transition Model: Deep Dive

### Core Innovation: Set-to-Set Learning

Traditional models predict: **single cell → single cell**

STATE predicts: **set of cells → set of cells**

This captures:
- **Population-level effects**: Not all cells respond identically
- **Cell-cell interactions**: Cells can "attend" to each other
- **Distributional shifts**: The entire cell population distribution changes

### Architecture Components

```
┌─────────────────────────────────────────────────────────────┐
│                    STATE TRANSITION MODEL                    │
│                                                               │
│  ┌────────────────┐      ┌────────────────┐                 │
│  │ Control Cells  │      │ Perturbation   │                 │
│  │ [B, S, genes]  │      │ [B, S, pert]   │                 │
│  └───────┬────────┘      └───────┬────────┘                 │
│          │                       │                           │
│          ▼                       ▼                           │
│  ┌────────────────┐      ┌────────────────┐                 │
│  │ Basal Encoder  │      │ Pert Encoder   │                 │
│  │ (MLP layers)   │      │ (MLP layers)   │                 │
│  └───────┬────────┘      └───────┬────────┘                 │
│          │                       │                           │
│          └───────────┬───────────┘                           │
│                      ▼                                       │
│              ┌──────────────┐                                │
│              │ Combined     │                                │
│              │ Input [B,S,H]│                                │
│              └──────┬───────┘                                │
│                     ▼                                        │
│              ┌──────────────┐                                │
│              │ Transformer  │                                │
│              │ Backbone     │                                │
│              │ (GPT-2/LLaMA)│                                │
│              └──────┬───────┘                                │
│                     ▼                                        │
│              ┌──────────────┐                                │
│              │ Output       │                                │
│              │ Projection   │                                │
│              └──────┬───────┘                                │
│                     ▼                                        │
│              ┌──────────────┐                                │
│              │ Predicted    │                                │
│              │ Perturbed    │                                │
│              │ Cells [B,S,G]│                                │
│              └──────────────┘                                │
└─────────────────────────────────────────────────────────────┘

Legend:
B = Batch size
S = Sequence length (cell_set_len, e.g., 512 cells)
H = Hidden dimension (e.g., 696)
G = Gene dimension (e.g., 2000 highly variable genes)
```

### Forward Pass Explanation

1. **Encode Control Cells**
   - Input: Gene expression of control cells [B, S, genes]
   - Basal Encoder (MLP): Projects to hidden space [B, S, hidden_dim]

2. **Encode Perturbation**
   - Input: Perturbation one-hot/embedding [B, S, pert_dim]
   - Pert Encoder (MLP): Projects to hidden space [B, S, hidden_dim]

3. **Combine**
   - Add perturbation effect to control cells: `combined = basal + pert`
   - Shape: [B, S, hidden_dim]

4. **Transformer Processing**
   - Cells attend to each other using GPT-2 or LLaMA architecture
   - Each cell's representation is influenced by all other cells in the set
   - Captures collective behavior and population-level patterns

5. **Output Projection**
   - Project back to gene space [B, S, genes]
   - If `predict_residual=True`: Add original control cells back
     - Predicts **change** not absolute state
   - Apply ReLU (gene expression is non-negative)

### Loss Functions

STATE uses **optimal transport losses** instead of simple MSE:

1. **Energy Loss** (default)
   ```python
   SamplesLoss(loss="energy", blur=0.05)
   ```
   - Measures how much "work" is needed to transform predicted distribution → true distribution
   - Robust to outliers, captures distributional differences

2. **Sinkhorn Loss**
   ```python
   SamplesLoss(loss="sinkhorn", blur=0.05)
   ```
   - Entropic optimal transport
   - Smoother, differentiable
   - Better for small sample sizes

3. **Combined Loss (SE)**
   ```python
   CombinedLoss(sinkhorn_weight=0.001, energy_weight=1.0, blur=0.05)
   ```
   - Best of both worlds: Sinkhorn (smooth) + Energy (robust)

4. **Mixture-Kernel MMD** (Experimental, Mod #2)
   - Multi-scale loss: [0.01, 0.05, 0.1, 0.5]
   - Captures both fine-grained and coarse-grained differences
   - Better for heterogeneous data

---

## Input Format

### Training Data

**Primary Input:** TOML configuration file (e.g., `examples/mixed.toml`)

```toml
# Dataset paths - maps dataset names to their directories
[datasets]
example = "./examples"

# Training specifications
[training]
example = "train"

# Zeroshot specifications - entire cell types go to val or test
[zeroshot]
"example.CT3" = "test"  # Hold out entire cell type for testing

# Fewshot specifications - explicit perturbation lists
[fewshot]
[fewshot."example.CT4"]
val = ["TARGET3"]           # Validation perturbations
test = ["TARGET4", "TARGET5"]  # Test perturbations
```

**Dataset Structure:** AnnData `.h5ad` files

```python
adata = sc.read_h5ad("example_data.h5ad")

# Required fields:
adata.X                    # Gene expression [cells × genes] (or .obsm['X_hvg'])
adata.obs['cell_type']     # Cell type labels
adata.obs['target_gene']   # Perturbation labels (gene knockouts, drugs, etc.)
adata.obs['batch_var']     # Batch/experiment identifiers

# Optional fields:
adata.obsm['X_hvg']        # Highly variable genes (if preprocessed)
adata.uns['hvg_names']     # Names of highly variable genes
```

**Perturbation Types:**
- **Genetic:** Gene knockouts (CRISPR), overexpression
- **Chemical:** Drug treatments (drug name + concentration)
- **Control:** "non-targeting", "DMSO", "TARGET1", etc.

### Training Command

```bash
state tx train \
  data.kwargs.toml_config_path="examples/mixed.toml" \
  data.kwargs.embed_key=X_hvg \
  data.kwargs.num_workers=32 \
  data.kwargs.batch_col=batch_var \
  data.kwargs.pert_col=target_gene \
  data.kwargs.cell_type_key=cell_type \
  data.kwargs.control_pert=TARGET1 \
  training.max_steps=5000 \
  training.batch_size=64 \
  training.lr=1e-4 \
  model=state \
  output_dir="./mixed_for_competition" \
  name="unified_model_mixed_for_the_meeting"
```

**Key Parameters:**
- `embed_key`: Which representation to use (`X_hvg` for highly variable genes, `None` for all genes)
- `cell_set_len`: Number of cells per "sentence" (default: 512)
- `max_steps`: Training steps (e.g., 5000-50000)
- `batch_size`: Number of cell sentences per batch (NOT individual cells!)
- `model`: Model type (`state`, `cpa`, `scvi`, `scgpt-*`, baselines)

---

## Output Format

### Training Outputs

After training, the output directory contains:

```
{output_dir}/{name}/
├── config.yaml                      # Complete training configuration
├── data_module.torch                # Saved data module state
├── cell_type_onehot_map.pkl         # Cell type → one-hot mapping
├── pert_onehot_map.pt               # Perturbation → one-hot mapping
├── batch_onehot_map.pkl             # Batch → one-hot mapping
├── var_dims.pkl                     # Variable dimensions
├── checkpoints/
│   ├── final.ckpt                   # Final model (best for inference)
│   ├── last.ckpt                    # Latest checkpoint (for resuming)
│   ├── step=26000-val_loss=1.8134.ckpt
│   └── ...
├── version_0/
│   ├── hparams.yaml                 # Hyperparameters
│   └── metrics.csv                  # Training/validation metrics
└── wandb_path.txt                   # Weights & Biases run path
```

### Inference Inputs

**Command:**
```bash
state tx infer \
  --model-dir ./mixed_for_competition/unified_model_mixed_for_the_meeting/ \
  --adata competition_support_set/competition_val_template.h5ad \
  --output competition/prediction.h5ad \
  --pert-col target_gene \
  --embed-key X_hvg
```

**Input AnnData (Inference):**
```python
adata = sc.read_h5ad("competition_val_template.h5ad")

# Required:
adata.X or adata.obsm['X_hvg']      # Control cells
adata.obs['target_gene']             # Perturbations to simulate
adata.obs['cell_type']               # Cell types (for grouping)
```

### Inference Outputs

**Output AnnData:**
```python
adata_predicted = sc.read_h5ad("competition/prediction.h5ad")

# Contains:
adata_predicted.X                    # Predicted perturbed gene expression
adata_predicted.obsm['X_hvg']        # Predicted HVG expression (if used)
adata_predicted.obs                  # Original metadata (preserved)

# Each row is now the predicted perturbed state
# Even control cells are "simulated" using the model
```

---

## The Modeling Approach Explained

### Key Concept: Virtual Experiments

STATE performs **virtual experiments**:

1. **Real Experiment (Expensive):**
   - Take cells → Apply drug → Wait hours/days → Sequence → Measure gene expression
   - Cost: $1000s, Time: weeks

2. **Virtual Experiment (Cheap):**
   - Take cell data → Run model → Get predictions in seconds
   - Cost: pennies, Time: seconds

### How Training Works

**1. Data Preparation**

```python
# Example batch structure:
batch = {
    'ctrl_cell_emb': [64, 512, 2000],    # 64 batches × 512 cells × 2000 genes
    'pert_emb': [64, 512, 128],           # Perturbation one-hot vectors
    'pert_cell_emb': [64, 512, 2000],    # True perturbed cells (targets)
    'batch': [64, 512],                   # Batch indices (optional)
    'pert_name': List[str],               # Perturbation names
    'cell_type': List[str],               # Cell type labels
}
```

**2. Cell Sentences**

- Cells are grouped into "sentences" of fixed length (e.g., 512 cells)
- All cells in a sentence have the **same perturbation** (homogeneous)
- This allows the transformer to learn relationships within perturbed populations

**3. Training Loop**

```python
for batch in dataloader:
    # Forward pass
    predicted_perturbed = model(batch)  # [B, S, genes]
    
    # Compute optimal transport loss
    true_perturbed = batch['pert_cell_emb']  # [B, S, genes]
    loss = optimal_transport_loss(predicted_perturbed, true_perturbed)
    
    # Backward pass
    loss.backward()
    optimizer.step()
```

**4. Key Training Features**

- **Early Stopping** (Mod #1): Stops when validation loss plateaus
- **Gradient Clipping**: Prevents exploding gradients
- **Mixed Precision**: BF16 for memory efficiency
- **Distributed Training**: Multi-GPU support via PyTorch Lightning
- **Checkpointing**: Saves best models based on validation loss

### How Inference Works

**1. Virtual Experiment Setup**

```python
# For each cell in input:
1. Identify its perturbation label (e.g., "GENE_X_knockout")
2. Sample matched control cells from the same cell type
3. Encode: control_cells + perturbation_vector
4. Run through transformer
5. Predict: perturbed_cells
6. Write predictions back to AnnData
```

**2. Grouping Strategy**

- Cells are grouped by: **cell type × perturbation**
- Each group is processed independently
- Controls are sampled with replacement from the cell-type-specific control pool
- Fallback to global controls if cell-type-specific controls are unavailable

**3. Homogeneous Forward Passes**

Unlike training (where batches mix perturbations), inference processes **one perturbation at a time**:

```python
for cell_type in unique_cell_types:
    for perturbation in unique_perturbations:
        # All cells in this batch have the same perturbation
        cells_to_predict = adata[(adata.obs['cell_type'] == cell_type) & 
                                   (adata.obs['pert'] == perturbation)]
        
        # Sample matched controls
        controls = sample_controls(cell_type, n=len(cells_to_predict))
        
        # Predict
        predictions = model.predict(controls, perturbation)
        
        # Write back
        adata.X[cells_to_predict.index] = predictions
```

---

## Understanding Input, Perturbation, and Output

This is the **most fundamental concept** in STATE. Let me explain with concrete examples to clear up common confusion.

### The Core Concept

```
INPUT:       Control cells (baseline gene expression)
             ↓
PERTURBATION: What you DO to the cells (e.g., knock out a gene, add a drug)
             ↓
OUTPUT:      Changed cells (gene expression AFTER perturbation)
```

### In Your Data File

Your AnnData file contains **BOTH control and perturbed cells** mixed together:

```python
adata = sc.read_h5ad("examples/random.h5ad")

# Each row is a CELL with these properties:
┌──────┬─────────────┬───────────┬──────────────────────────────┐
│ Cell │ target_gene │ cell_type │ What It Means                │
├──────┼─────────────┼───────────┼──────────────────────────────┤
│ 0    │ TARGET1     │ CT3       │ Control cell (baseline)      │
│ 1    │ TARGET1     │ CT3       │ Control cell (baseline)      │
│ 2    │ TARGET2     │ CT3       │ Perturbed cell (TARGET2)     │
│ 3    │ TARGET2     │ CT3       │ Perturbed cell (TARGET2)     │
│ 4    │ TARGET3     │ CT4       │ Perturbed cell (TARGET3)     │
└──────┴─────────────┴───────────┴──────────────────────────────┘

# Each cell also has GENE EXPRESSION:
adata.obsm['X_hvg'][0, :]  # [2.3, 5.1, 0.0, ..., 3.4] - 2000 gene values
adata.obsm['X_hvg'][2, :]  # [1.8, 6.2, 0.5, ..., 2.9] - Different!
```

### Concrete Example: TARGET1 vs TARGET2

```python
# CONTROL CELLS (TARGET1 = baseline, no perturbation)
INPUT (Control Cell):
┌─────────────────────────────────────────┐
│ Cell Type: CT3                          │
│ Perturbation: TARGET1 (control)         │
│ Gene Expression (2000 genes):           │
│   Gene_0: 2.3  (e.g., MYC expression)  │
│   Gene_1: 5.1  (e.g., TP53 expression) │
│   Gene_2: 0.0  (e.g., CD4 expression)  │
│   ...                                   │
│   Gene_1999: 3.4 (e.g., BCL2)          │
└─────────────────────────────────────────┘

# PERTURBED CELLS (TARGET2 was applied)
OUTPUT (After TARGET2 Perturbation):
┌─────────────────────────────────────────┐
│ Cell Type: CT3 (same cell type)         │
│ Perturbation: TARGET2 (intervention)     │
│ Gene Expression (2000 genes):           │
│   Gene_0: 1.8  ← CHANGED! (was 2.3)    │
│   Gene_1: 6.2  ← CHANGED! (was 5.1)    │
│   Gene_2: 0.5  ← CHANGED! (was 0.0)    │
│   ...                                   │
│   Gene_1999: 2.9 ← CHANGED! (was 3.4)  │
└─────────────────────────────────────────┘

PERTURBATION: "TARGET2" 
(the action/intervention that caused the change)
```

### What the Model Learns

```python
MODEL LEARNS THIS RELATIONSHIP:

Control cells (TARGET1) + Perturbation (TARGET2) → Perturbed cells

Mathematically:
INPUT:  Control gene expression [2.3, 5.1, 0.0, ..., 3.4]
PLUS:   Perturbation "TARGET2" (encoded as a vector)
EQUALS: Output gene expression [1.8, 6.2, 0.5, ..., 2.9]
```

---

## What Does "TARGET" Actually Mean? 🎯

### The Confusion: Multiple Meanings

The term "TARGET" and the column name `target_gene` cause **major confusion**:

#### In Biology (Drug Discovery Context):

```
Traditional Drug Discovery Terminology:
┌─────────────────────────────────────────┐
│ DRUG: Imatinib (the molecule/compound) │
│   ↓ binds to                            │
│ TARGET: BCR-ABL protein (the target)    │
│   ↓ which affects                       │
│ EFFECT: Cancer cells die                │
└─────────────────────────────────────────┘

Example: "Imatinib TARGETS the BCR-ABL protein"
         "BCR-ABL is the TARGET of Imatinib"
```

#### In STATE Code (Generic Perturbation):

```
STATE Terminology:
┌─────────────────────────────────────────┐
│ PERTURBATION: What you DO to cells     │
│   Could be: drug, gene knockout,        │
│   environmental change, etc.            │
│   ↓                                     │
│ LABEL: Stored in 'target_gene' column  │
│   (Misleading name!)                    │
│   ↓                                     │
│ EFFECT: Changed gene expression         │
└─────────────────────────────────────────┘

"target_gene='Imatinib'" means "perturbation is Imatinib"
           ↑
    COLUMN NAME (confusing - could be drug OR gene!)
```

### What "TARGET" Can Represent

The "TARGET" labels in your data could represent:

#### Scenario 1: Gene Knockout Experiments (CRISPR)

```python
# Real biological data:
adata.obs['target_gene'].unique()
# ['non-targeting', 'TP53', 'MYC', 'BRCA1', 'KRAS']
#      ↑              ↑      ↑      ↑        ↑
#   CONTROL       Gene1   Gene2   Gene3    Gene4
#   (no change)   knocked knocked knocked  knocked
#                  out     out     out      out

# Interpretation:
TARGET1 (non-targeting) = Control, no gene modified (INPUT)
TARGET2 (TP53) = TP53 gene was knocked out (OUTPUT)
TARGET3 (MYC) = MYC gene was knocked out (OUTPUT)

# PERTURBATION = "knock out the TP53 gene"
```

#### Scenario 2: Drug Screening

```python
# Real drug screening data:
adata.obs['target_gene'].unique()  # Misleading column name!
# ['DMSO', 'Imatinib', 'Cisplatin', 'Aspirin', 'Doxorubicin']
#    ↑         ↑          ↑           ↑            ↑
# CONTROL   Drug1      Drug2       Drug3        Drug4
# (vehicle) (cancer)   (chemo)     (pain)       (cancer)

# Interpretation:
TARGET1 (DMSO) = Control, no drug (INPUT)
TARGET2 (Imatinib) = Cells treated with Imatinib (OUTPUT)
TARGET3 (Cisplatin) = Cells treated with Cisplatin (OUTPUT)

# PERTURBATION = "treat cells with Imatinib drug"
# Note: Imatinib is the DRUG, not the biological target!
```

#### Scenario 3: Generic Placeholders (Your Data)

```python
# Your example data:
adata.obs['target_gene'].unique()
# ['TARGET1', 'TARGET2', 'TARGET3', 'TARGET4', 'TARGET5']
#      ↑          ↑          ↑          ↑          ↑
#   CONTROL   Pert1      Pert2      Pert3      Pert4

# These are PLACEHOLDERS - actual meaning could be:
Option A - Gene names:
  TARGET1 = non-targeting (control)
  TARGET2 = TP53 (gene knockout)
  TARGET3 = MYC (gene knockout)

Option B - Drug names:
  TARGET1 = DMSO (vehicle control)
  TARGET2 = Imatinib (drug treatment)
  TARGET3 = Aspirin (drug treatment)

Option C - Environmental conditions:
  TARGET1 = Normal oxygen (control)
  TARGET2 = Low oxygen (hypoxia)
  TARGET3 = High glucose

# The model doesn't care what they represent!
# It just learns: Control + Perturbation → Effect
```

---

## Training Data Structure

Your data file contains **the answers**:

```python
# In random.h5ad, you have pairs of:

QUESTIONS (What the model learns):
Q1: "What happens to CT3 cells when you apply TARGET2?"
Q2: "What happens to CT4 cells when you apply TARGET3?"

ANSWERS (Already in the data):
A1: CT3 + TARGET2 cells → Gene expression [1.8, 6.2, ...]
A2: CT4 + TARGET3 cells → Gene expression [2.8, 3.9, ...]

The model LEARNS from these examples, then PREDICTS NEW cases:
Q3: "What happens to CT3 cells with TARGET6?" (NEW, unseen!)
A3: Model predicts → [1.5, 6.8, ...] based on learned patterns
```

### Data Breakdown

```python
# Your data distribution:
adata.obs['target_gene'].value_counts()

# Example output:
TARGET1:  8000 cells  ← CONTROL (baseline) - INPUT
TARGET2:  2000 cells  ← PERTURBED (intervention) - OUTPUT  
TARGET3:  1500 cells  ← PERTURBED (intervention) - OUTPUT
TARGET4:  1000 cells  ← PERTURBED (intervention) - OUTPUT
TARGET5:   500 cells  ← PERTURBED (intervention) - OUTPUT

Total: 13,000 cells
- 8000 controls (INPUT for training)
- 5000 perturbed (OUTPUT for training)
```

---

## The Training Process

```python
# What happens during training:

STEP 1: Sample a batch
├─ Get 512 control cells (TARGET1, CT3)
└─ Get 512 perturbed cells (TARGET2, CT3)

STEP 2: Forward pass
├─ Encode control cells → [512, 2000] gene expression
├─ Encode perturbation "TARGET2" → [512, 128] perturbation vector
├─ Model predicts: control + TARGET2 → [512, 2000] predicted expression
└─ Compare to actual TARGET2 cells

STEP 3: Calculate loss
├─ Optimal transport distance between predicted and actual
└─ Loss = how different are the distributions?

STEP 4: Backpropagation
└─ Update model weights to reduce loss

STEP 5: Repeat with different perturbations
├─ Try TARGET3, TARGET4, TARGET5...
└─ Model learns general perturbation effects
```

---

## Visual Summary

```
YOUR DATA FILE (random.h5ad):
════════════════════════════════════════════════════════════

    CONTROL CELLS                 PERTURBED CELLS
    (Baseline)                    (After intervention)
    
┌─────────────────┐            ┌─────────────────┐
│ Cell: CT3       │            │ Cell: CT3       │
│ Label: TARGET1  │            │ Label: TARGET2  │
│ Genes: [2.3,    │            │ Genes: [1.8,    │
│         5.1,    │            │         6.2,    │
│         0.0,    │            │         0.5,    │
│         ...]    │            │         ...]    │
└─────────────────┘            └─────────────────┘
        ↓                              ↓
      INPUT                         OUTPUT
                        
           What caused the change?
           PERTURBATION: TARGET2
     (Could be: gene knockout, drug, etc.)
```

---

## Key Takeaways

| Component | In Your Data | Meaning |
|-----------|--------------|---------|
| **INPUT** | Cells labeled `TARGET1` | Control/baseline cells (no intervention) |
| **PERTURBATION** | The label (TARGET2, TARGET3, etc.) | What was done to the cells |
| **OUTPUT** | Cells labeled `TARGET2`, etc. | Result after perturbation |
| **`target_gene` column** | Misleading name! | Actually contains perturbation labels |
| **Is TARGET a drug?** | Could be! | Or gene, or condition - depends on experiment |

### In Your Training Command:

```bash
state tx train \
  data.kwargs.control_pert=TARGET1 \  # ← INPUT (control cells)
  data.kwargs.pert_col=target_gene \  # ← Column with perturbation labels
  ...

# The model learns:
# INPUT (TARGET1 cells) + PERTURBATION (TARGET2) → OUTPUT (predicted cells)
```

### The Critical Insight:

**Your h5ad file contains both questions AND answers!**
- Question: "What if we perturb these cells?"
- Answer: "Here's what actually happened" (in the perturbed cells)
- Model learns: "How to predict what will happen"

The confusion arises because:
1. ✅ The file contains **both** control and perturbed cells
2. ✅ The column `target_gene` tells you which is which
3. ✅ "TARGET" can mean drug, gene, or any perturbation
4. ⚠️  The column name `target_gene` is misleading (not always a gene!)

---

## Model Variants

### Available Models

1. **`state`** (Default)
   - Main State Transition model
   - GPT-2 or LLaMA backbone
   - Optimal transport loss
   - Best performance

2. **`pertsets`**
   - Perturbation-aware variant
   - Specialized for drug combinations

3. **`cpa`**
   - Compositional Perturbation Autoencoder
   - Variational approach

4. **`scvi`**
   - scVI-based model
   - Variational inference

5. **`scgpt-*`**
   - scGPT transformer variants
   - Pre-trained on large corpora

### Baseline Models

- **`context_mean`**: Mean of context cells
- **`embed_sum`**: Sum of embeddings
- **`perturb_mean`**: Mean of perturbation effects

---

## Key Innovations

### 1. Set-to-Set Learning
- Not single cell → single cell
- Set of cells → set of cells
- Captures population-level effects

### 2. Optimal Transport Losses
- Energy loss: Robust, captures distributional shifts
- Sinkhorn loss: Smooth, differentiable
- Better than MSE for this task

### 3. Residual Prediction
- Predicts **change** (Δ) not absolute state
- More stable, easier to learn
- Formula: `predicted = control + model(control, pert)`

### 4. Attention Mechanisms
- Cells attend to each other via transformer
- Learns collective behavior patterns
- Models cell-cell interactions

### 5. Confidence Tokens (Optional)
- Learnable tokens that predict model uncertainty
- Helps identify low-confidence predictions

---

## Recent Modifications (Experiment Log)

### Modification #1: Early Stopping ✅
**Problem:** Model was overfitting and experiencing catastrophic forgetting

**Solution:** Added `ImprovedEarlyStopping` callback
- Monitors validation loss
- Stops if no improvement for 5 validation checks
- Default patience: 10,000 steps
- Expected improvement: 10-15% better ranking

**Usage:**
```bash
make train-experiment EXPERIMENT=mod01_early_stopping STEPS=50000
```

### Modification #2: Mixture-Kernel MMD ✅
**Problem:** Single-scale MMD fails with varying data volumes and heterogeneous cell densities

**Solution:** Multi-scale MMD with blur scales [0.01, 0.05, 0.1, 0.5]
- Captures fine → coarse structure
- Adaptive weighting options
- Hierarchical coarse-to-fine variant
- Expected improvement: 8-12% better ranking

**Usage:**
```bash
./run.sh tx train model.kwargs.loss=mixture_mmd ...
```

---

## Evaluation Metrics

### Correlation Metrics
- **Pearson Correlation**: Linear relationship strength
  - Good: > 0.3
  - Poor: < 0.1 or negative

### Error Metrics
- **MSE**: Mean Squared Error (lower is better)
  - Good: < 0.03
- **MAE**: Mean Absolute Error (lower is better)
  - Good: < 0.15

### Discrimination Metrics
- **Discrimination Score**: Can the model distinguish perturbations?
  - Good: > 0.8
  - Measures: L1, L2, Cosine similarity

---

## Common Use Cases

### 1. Drug Discovery
```bash
# Train model on known drug-cell responses
state tx train --config drugs.toml

# Predict effects of new drugs
state tx infer --adata new_drugs.h5ad --output predictions.h5ad
```

### 2. Gene Knockout Prediction
```bash
# Train on CRISPR screen data
state tx train --config crispr_screen.toml

# Predict effects of knocking out novel genes
state tx infer --adata novel_genes.h5ad
```

### 3. Virtual Cell Challenge
```bash
# Train on competition data
make train CONFIG=examples/mixed.toml

# Predict on validation set
state tx infer --model-dir ./mixed_for_competition/unified_model/ \
               --adata competition_val_template.h5ad \
               --output submission.h5ad
```

---

## Performance and Scalability

### Training
- **Hardware:** GPU recommended (CUDA 11.2+)
- **Memory:** 16-32 GB GPU RAM (depends on batch size and cell_set_len)
- **Time:** ~2-6 hours for 5000-50000 steps
- **Distributed:** Multi-GPU support via PyTorch Lightning

### Inference
- **Speed:** ~1000 cells/second (GPU)
- **Memory:** Scales with cell_set_len (512 cells = ~2 GB)
- **Batch Processing:** Automatically handles large datasets

### Optimization Features
- **Mixed Precision (BF16)**: 2× faster, 2× less memory
- **Gradient Accumulation**: Handle larger effective batch sizes
- **Gradient Checkpointing**: Trade compute for memory
- **LoRA Adapters**: Fine-tune efficiently (optional)

---

## Project Structure

```
state/
├── src/state/
│   ├── tx/                          # State Transition model
│   │   ├── models/
│   │   │   ├── state_transition.py  # Main ST model
│   │   │   ├── mixture_kernel_mmd.py # Multi-scale MMD (Mod #2)
│   │   │   ├── cpa/                 # CPA model
│   │   │   ├── scvi/                # scVI model
│   │   │   └── scgpt/               # scGPT models
│   │   ├── callbacks/               # Training callbacks
│   │   │   └── early_stopping.py    # Early stopping (Mod #1)
│   │   └── data/                    # Data loading
│   ├── emb/                         # State Embedding model
│   ├── _cli/                        # Command-line interfaces
│   │   ├── _tx/
│   │   │   ├── _train.py            # Training CLI
│   │   │   ├── _infer.py            # Inference CLI
│   │   │   └── _predict.py          # Evaluation CLI
│   │   └── _emb/                    # Embedding CLI
│   └── configs/                     # Configuration files
│       ├── model/
│       │   ├── state.yaml           # Main model config
│       │   ├── cpa.yaml
│       │   └── ...
│       ├── training/
│       │   └── default.yaml
│       └── wandb/
│           └── default.yaml
├── examples/                        # Example configs
│   ├── mixed.toml                   # Mixed zeroshot + fewshot
│   ├── fewshot.toml
│   └── zeroshot.toml
├── experiments/                     # Experiment outputs
│   ├── mod01_early_stopping/
│   ├── mod02_mixture_kernel_mmd/
│   └── ...
├── Makefile                         # Build automation
├── run.sh                           # Main runner script
├── pyproject.toml                   # Dependencies
└── README.md
```

---

## Dependencies

### Core
- **PyTorch 2.4.1**: Deep learning framework
- **Lightning**: Training infrastructure
- **AnnData**: Single-cell data format
- **Scanpy**: Single-cell analysis

### Model
- **Transformers (Hugging Face)**: GPT-2, LLaMA backbones
- **geomloss**: Optimal transport losses
- **PEFT**: LoRA adapters (optional)

### Data
- **cell-load**: Data loading utilities
- **cell-eval**: Evaluation metrics
- **pandas, numpy, scipy**: Standard scientific computing

### Utilities
- **Weights & Biases**: Experiment tracking
- **Hydra**: Configuration management
- **uv**: Fast Python package manager

---

## Installation

```bash
# Clone repository
git clone --branch experimental_setup git@github.com:DeepSpringAI/state.git
cd state

# Install dependencies (GPU)
make setup

# Or CPU-only
make setup-cpu

# Download data
python data_download.py
```

---

## Quick Start

```bash
# 1. Train a model
make train

# 2. Evaluate on test set
state tx predict \
  --output-dir ./mixed_for_competition/unified_model/ \
  --checkpoint final.ckpt

# 3. Run inference on new data
state tx infer \
  --model-dir ./mixed_for_competition/unified_model/ \
  --adata new_data.h5ad \
  --output predictions.h5ad
```

---

## Troubleshooting

### CUDA Errors
```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU
nvidia-smi

# Apply GPU fix (if on problematic system)
source gpu_fix_optimized.sh
```

### Memory Issues
```bash
# Reduce batch size
make train BATCH=32  # Default: 64

# Reduce cell_set_len
./run.sh tx train model.kwargs.cell_set_len=256  # Default: 512
```

### Slow Training
```bash
# Check num_workers
make train NUM_WORKERS=16  # Adjust based on CPU cores

# Use mixed precision (automatic with scGPT)
```

---

## Citation

If you use STATE, please cite:

```bibtex
@article{state2024,
  title={Predicting cellular responses to perturbation across diverse contexts with State},
  author={Arc Research Institute},
  journal={arXiv},
  year={2024},
  url={https://arcinstitute.org/manuscripts/State}
}
```

---

## License

- **Code:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)
- **Model Weights:** Arc Research Institute State Model Non-Commercial License
- **Subject to:** Arc Research Institute State Model Acceptable Use Policy

---

## Summary

**STATE** is a powerful framework for predicting cellular responses to perturbations using:
- **Set-to-set learning** with transformers
- **Optimal transport losses** for distributional matching
- **Residual prediction** for stability
- **Multi-scale approaches** for robustness

It enables **virtual experiments**, replacing expensive wet-lab work with fast computational predictions, accelerating drug discovery, gene function research, and personalized medicine.

---

## Real-World Example: Drug Discovery Story

### Scenario: Finding a Treatment for Cancer

Let me walk you through a **complete drug discovery workflow** using STATE, from problem to solution.

---

### 🎯 The Problem

**Dr. Sarah Chen**, an oncologist at a pharmaceutical company, faces this challenge:

> "We have a promising new drug candidate called **CompoundX** that showed anti-cancer activity in initial screens. But we don't know:
> - Which cancer cell types it will work on?
> - What genes/pathways it affects?
> - Will it have toxic side effects?
> - How does it compare to existing drugs?"

**Traditional approach:**
- Test in 20 different cancer cell lines → $500K, 6 months
- Screen against 100 drugs → $2M, 1 year
- Total: **$2.5M and 18 months** before knowing if it's worth pursuing

**STATE approach:**
- Virtual experiments on all cell lines → $100, 2 days
- Compare to all known drugs → $50, 1 day
- Total: **$150 and 3 days** 🚀

---

### 📊 Step 1: Gather Input Data

#### What Dr. Chen Has:

```python
# Input Data File: cancer_cells_baseline.h5ad
import scanpy as sc

adata = sc.read_h5ad("cancer_cells_baseline.h5ad")

print(adata.shape)  # (50000 cells, 2000 genes)

# Cell metadata:
print(adata.obs['cell_type'].unique())
# ['K562', 'A549', 'MCF7', 'HeLa', 'Jurkat', 'HepG2', ...]  # 20 cancer types

print(adata.obs['treatment'].unique())
# ['DMSO']  # All cells are untreated controls

# Gene expression:
print(adata.obsm['X_hvg'].shape)  # (50000, 2000) HVG expression

# This is the BASELINE - healthy cancer cells before any treatment
```

#### What Dr. Chen Wants to Predict:

```python
# Goal: Predict what happens when we treat each cell type with CompoundX
# Need to create a "prediction template"

# For each cell type, we want to predict response to CompoundX
test_adata = adata.copy()
test_adata.obs['treatment'] = 'CompoundX'  # Change label to drug

# This will be the input for virtual experiments
test_adata.write_h5ad("cancer_cells_CompoundX_template.h5ad")
```

---

### 🔬 Step 2: Train the ST Model (One-Time Setup)

**Dr. Chen's team had previously trained STATE on a large database of drug responses:**

```python
# Training data: historical_drug_responses.h5ad
# - 100,000 cells
# - 50 different drugs
# - 20 cell types
# - Each drug has control cells + treated cells

training_data = sc.read_h5ad("historical_drug_responses.h5ad")

# Example training data structure:
# Cell 1: K562, DMSO (control)        → Gene expression: [2.3, 5.1, ...]
# Cell 2: K562, Imatinib (treated)    → Gene expression: [1.8, 6.2, ...]
# Cell 3: A549, DMSO (control)        → Gene expression: [3.1, 2.7, ...]
# Cell 4: A549, Cisplatin (treated)   → Gene expression: [2.9, 4.1, ...]
# ...

# Train ST model:
state tx train \
  data.kwargs.toml_config_path="drug_screening.toml" \
  data.kwargs.embed_key=X_hvg \
  data.kwargs.pert_col=treatment \
  data.kwargs.cell_type_key=cell_type \
  data.kwargs.control_pert=DMSO \
  training.max_steps=50000 \
  output_dir="./models" \
  name="drug_response_model"

# Output: Trained model in models/drug_response_model/
```

---

### 🎲 Step 3: Run Virtual Experiment with CompoundX

Now Dr. Chen uses the trained model to predict CompoundX effects:

```bash
# Run inference:
state tx infer \
  --model-dir ./models/drug_response_model/ \
  --adata cancer_cells_CompoundX_template.h5ad \
  --output CompoundX_predictions.h5ad \
  --pert-col treatment \
  --embed-key X_hvg

# Processing...
# ✓ Predicting K562 response to CompoundX... 
# ✓ Predicting A549 response to CompoundX...
# ✓ Predicting MCF7 response to CompoundX...
# ... (20 cell types total)
# Done in 2 minutes! ⚡
```

#### What Just Happened:

```python
# For each cancer cell type:
# 1. Model takes control cells (DMSO-treated)
# 2. Model encodes "CompoundX" perturbation
# 3. Model predicts: control + CompoundX → predicted response
# 4. Output: predicted gene expression after CompoundX treatment

# The model learned from 50 drugs, now predicts response to NEW drug!
```

---

### 📈 Step 4: Analyze Predictions

Dr. Chen examines the results:

```python
import scanpy as sc
import numpy as np
import pandas as pd

# Load predictions
predictions = sc.read_h5ad("CompoundX_predictions.h5ad")

# predictions.X now contains PREDICTED gene expression 
# after CompoundX treatment

# Compare to baseline:
baseline = sc.read_h5ad("cancer_cells_baseline.h5ad")

# For each cell type, calculate drug effect:
results = []

for cell_type in predictions.obs['cell_type'].unique():
    # Get cells of this type
    pred_cells = predictions[predictions.obs['cell_type'] == cell_type]
    base_cells = baseline[baseline.obs['cell_type'] == cell_type]
    
    # Calculate mean expression change
    pred_mean = pred_cells.obsm['X_hvg'].mean(axis=0)
    base_mean = base_cells.obsm['X_hvg'].mean(axis=0)
    
    fold_change = pred_mean / (base_mean + 1e-10)
    
    # Key cancer genes:
    cancer_genes = ['MYC', 'BCL2', 'TP53', 'KRAS', 'EGFR']
    gene_indices = [predictions.var.index.get_loc(g) for g in cancer_genes]
    
    results.append({
        'cell_type': cell_type,
        'MYC_change': fold_change[gene_indices[0]],
        'BCL2_change': fold_change[gene_indices[1]],
        'TP53_change': fold_change[gene_indices[2]],
        'viability_score': 1.0 / fold_change[gene_indices[0]]  # Lower MYC = less viable
    })

results_df = pd.DataFrame(results)
print(results_df)
```

#### Output Analysis:

```python
# Results:
┌───────────┬────────────┬─────────────┬─────────────┬──────────────────┐
│ Cell Type │ MYC Change │ BCL2 Change │ TP53 Change │ Viability Score  │
├───────────┼────────────┼─────────────┼─────────────┼──────────────────┤
│ K562      │ 0.32 ↓     │ 0.45 ↓      │ 1.85 ↑      │ 3.12 (EFFECTIVE) │
│ A549      │ 0.89 →     │ 0.91 →      │ 1.12 ↑      │ 1.12 (WEAK)      │
│ MCF7      │ 0.28 ↓     │ 0.38 ↓      │ 1.92 ↑      │ 3.57 (EFFECTIVE) │
│ HeLa      │ 0.71 ↓     │ 0.68 ↓      │ 1.45 ↑      │ 1.41 (MODERATE)  │
│ Jurkat    │ 0.25 ↓     │ 0.41 ↓      │ 1.78 ↑      │ 4.00 (EFFECTIVE) │
│ HepG2     │ 0.95 →     │ 0.97 →      │ 1.05 →      │ 1.05 (NO EFFECT) │
└───────────┴────────────┴─────────────┴─────────────┴──────────────────┘

Legend:
↓ = Down-regulated (good for cancer treatment!)
↑ = Up-regulated
→ = Unchanged
```

---

### 💡 Step 5: Key Insights

Dr. Chen discovers:

#### ✅ **Positive Findings:**

```python
# 1. CompoundX is SELECTIVE
effective_cells = ['K562', 'MCF7', 'Jurkat']  # Leukemia, breast cancer, T-cell
ineffective_cells = ['A549', 'HepG2']          # Lung cancer, liver cancer

print("CompoundX is effective on blood cancers and breast cancer!")
print("Not effective on solid tumors (lung, liver)")
```

#### ✅ **Mechanism of Action:**

```python
# 2. CompoundX works by:
# - Suppressing MYC (oncogene) → Cancer cells stop growing
# - Suppressing BCL2 (anti-apoptosis) → Cancer cells die
# - Activating TP53 (tumor suppressor) → DNA damage response

print("Mechanism: MYC inhibitor + BCL2 inhibitor")
```

#### ⚠️ **Safety Concerns:**

```python
# 3. Check predictions on normal cells:
normal_cells = predictions[predictions.obs['cell_type'] == 'normal_lymphocytes']

if normal_cells.obsm['X_hvg'].mean() < threshold:
    print("WARNING: CompoundX may be toxic to normal immune cells!")
else:
    print("SAFE: Normal cells are not affected")
```

---

### 🔬 Step 6: Validate Top Prediction

Dr. Chen decides to **validate the top prediction experimentally:**

```python
# Virtual prediction said: "K562 cells most sensitive"
# Predicted viability: 31% (69% cell death)

# Run actual experiment:
# 1. Culture K562 cells
# 2. Treat with CompoundX
# 3. Measure cell viability

# Result after 48 hours: 35% viable (65% cell death)
# Prediction accuracy: 96%! ✅
```

---

### 📊 Step 7: Compare to Existing Drugs

Dr. Chen wants to know: **Is CompoundX better than approved drugs?**

```python
# Run predictions for known drugs on same cells:
known_drugs = ['Imatinib', 'Venetoclax', 'Dasatinib']

for drug in known_drugs:
    # Create template
    test_adata.obs['treatment'] = drug
    test_adata.write_h5ad(f"template_{drug}.h5ad")
    
    # Run inference
    state tx infer \
      --model-dir ./models/drug_response_model/ \
      --adata template_{drug}.h5ad \
      --output {drug}_predictions.h5ad

# Compare efficacy:
comparison = pd.DataFrame({
    'Drug': ['CompoundX', 'Imatinib', 'Venetoclax', 'Dasatinib'],
    'K562_Viability': [0.31, 0.42, 0.38, 0.35],
    'MCF7_Viability': [0.28, 0.78, 0.45, 0.82],
    'Selectivity': [3.2, 1.8, 2.4, 2.1]
})

print(comparison)
```

#### Comparison Results:

```
┌───────────┬───────────────┬───────────────┬─────────────┐
│ Drug      │ K562 Effect   │ MCF7 Effect   │ Selectivity │
├───────────┼───────────────┼───────────────┼─────────────┤
│ CompoundX │ 31% (BEST!)   │ 28% (BEST!)   │ 3.2 (BEST!) │
│ Imatinib  │ 42%           │ 78% (poor)    │ 1.8         │
│ Venetoclax│ 38%           │ 45%           │ 2.4         │
│ Dasatinib │ 35%           │ 82% (poor)    │ 2.1         │
└───────────┴───────────────┴───────────────┴─────────────┘

Conclusion: CompoundX is MORE EFFECTIVE than approved drugs! 🎯
```

---

### 🎯 Step 8: Business Decision

Armed with STATE predictions, Dr. Chen presents to leadership:

#### **Traditional Path (Without STATE):**
```
❌ Cost: $2.5M for initial screening
❌ Time: 18 months
❌ Risk: High - might fail after huge investment
❌ Decision: Leadership says "too risky, shelve CompoundX"
```

#### **STATE-Enabled Path:**
```
✅ Cost: $150 for virtual screening + $50K for validation
✅ Time: 3 days prediction + 2 weeks validation
✅ Risk: Low - validated predictions before big investment
✅ Decision: "CompoundX shows promise - proceed to clinical trials!"

Result: Drug advances 16 months faster, saving $2.4M 💰
```

---

### 📈 Step 9: Extended Applications

Dr. Chen continues using STATE:

#### **A. Combination Therapy:**
```python
# Question: "Should we combine CompoundX with other drugs?"

# Test combinations:
test_adata.obs['treatment'] = 'CompoundX+Venetoclax'
# Run inference...

# Result: Combination is 40% more effective! (viability: 18%)
```

#### **B. Dose Optimization:**
```python
# Question: "What's the optimal dose?"

# Encode different doses as different perturbation strengths
# (If model was trained with dose information)

doses = [1, 5, 10, 50, 100]  # μM
for dose in doses:
    # Predict response at each dose
    # Find minimum effective dose
```

#### **C. Patient Stratification:**
```python
# Question: "Which patients will respond?"

# Patient tumor samples:
patient_cells = sc.read_h5ad("patient_biopsies.h5ad")

# Predict response for each patient:
state tx infer \
  --model-dir ./models/drug_response_model/ \
  --adata patient_cells.h5ad \
  --output patient_predictions.h5ad

# Result: 
# - Patient 1: 85% tumor reduction (HIGH RESPONDER)
# - Patient 2: 25% tumor reduction (LOW RESPONDER)
# - Patient 3: 72% tumor reduction (MODERATE RESPONDER)

# Enroll only high responders in clinical trial → Higher success rate!
```

---

### 🎉 Final Outcome

**6 Months Later:**

Dr. Chen's paper is published:
> "**CompoundX: A Novel Selective MYC/BCL2 Inhibitor for Hematologic Malignancies**"
> 
> Key findings:
> - Phase I clinical trial: 67% response rate (predicted: 69%)
> - Minimal toxicity in normal cells (as predicted)
> - Effective in patient subgroup identified by STATE
> - Patent filed, FDA fast-track designation granted

**Cost Savings:**
- Traditional drug discovery: $2.5M, 5 years to Phase I
- STATE-accelerated: $250K, 1.5 years to Phase I
- **Savings: $2.25M and 3.5 years** ⚡

---

### 🔑 Key Takeaways

#### **What STATE Enabled:**

1. **Rapid Screening**: Test 1000 drugs on 20 cell types in hours (not years)
2. **Mechanism Discovery**: Understand how drugs work before experiments
3. **Selectivity Analysis**: Identify which cancers respond
4. **Safety Prediction**: Flag toxic effects early
5. **Combination Discovery**: Find synergistic drug pairs
6. **Patient Selection**: Personalized medicine - match drug to patient
7. **Cost Reduction**: 90% cheaper than traditional screening
8. **Time Acceleration**: 95% faster than wet-lab experiments

#### **The Input → Output Flow:**

```
INPUT:
┌─────────────────────────────────────┐
│ • Control cells (baseline)          │
│   - 50,000 cells                    │
│   - 20 cancer types                 │
│   - 2,000 genes per cell            │
│ • New drug: CompoundX               │
│ • Trained ST model                  │
└─────────────────────────────────────┘
                ↓
         STATE MODEL
    (Virtual Experiment)
                ↓
OUTPUT:
┌─────────────────────────────────────┐
│ • Predicted gene expression         │
│   after CompoundX treatment         │
│ • Cell viability predictions        │
│ • Mechanism of action               │
│ • Selectivity profile               │
│ • Safety assessment                 │
│ • Comparison to other drugs         │
└─────────────────────────────────────┘
                ↓
DECISIONS:
┌─────────────────────────────────────┐
│ ✓ Which cancers to target           │
│ ✓ Which patients to enroll          │
│ ✓ Which combinations to test        │
│ ✓ Go/No-Go for clinical trials     │
└─────────────────────────────────────┘
```

---

### 💭 Why This Matters

**Before STATE:**
- Drug candidates died in expensive late-stage trials
- 90% of drugs fail in clinical trials
- Average cost: $2.6B per approved drug
- Timeline: 10-15 years

**With STATE:**
- Fail fast on bad drugs (save money)
- Succeed on good drugs (better predictions)
- Personalize treatments (right drug for right patient)
- Accelerate timeline (computational experiments)

**Result:** More drugs reach patients, faster and cheaper 🎯

---

This is how STATE transforms drug discovery from expensive trial-and-error into intelligent, prediction-driven science!

