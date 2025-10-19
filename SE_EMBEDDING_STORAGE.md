# SE Model: Where Embeddings Are Saved

## 🎯 Yes! SE Model Saves Embeddings

The State Embedding (SE) model saves embeddings in **two places**:

---

## 📦 **Storage Location 1: AnnData Object (Primary)**

### **Where:**
```python
adata.obsm[emb_key]
```

### **Default Key:**
- `emb_key = "X_emb"` (default)
- Can be configured to any key (e.g., "X_state", "X_uce", "X_pca")

### **Format:**
```python
adata.obsm['X_emb']: np.ndarray
    Shape: [n_cells, embedding_dim]
    Example: [10000, 1536]
    Dtype: float32
```

### **Code Evidence:**
From `src/state/emb/inference.py` (lines 246-257):

```python
def encode_adata(self, input_adata_path, output_adata_path, emb_key="X_emb", ...):
    # ... encoding process ...
    
    # Concatenate all embeddings
    all_embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    
    # Save to AnnData obsm
    adata.obsm[emb_key] = all_embeddings  # ← SAVED HERE!
    
    # Write to file
    adata.write_h5ad(output_adata_path)
```

---

## 📦 **Storage Location 2: LanceDB Vector Database (Optional)**

### **Where:**
```
./state_embeddings.lancedb/
    └── state_embeddings (table)
```

### **Purpose:**
- Efficient similarity search
- Fast nearest neighbor queries
- Scales to millions of cells

### **Structure:**
```python
{
    "vector": [1536-dim embedding],
    "cell_id": "cell_0001",
    "embedding_key": "X_state",
    "dataset": "my_dataset",
    # ... other metadata from adata.obs ...
}
```

### **Code Evidence:**
From `src/state/emb/vectordb.py` (lines 7-59):

```python
class StateVectorDB:
    def __init__(self, db_path: str = "./state_embeddings.lancedb"):
        self.db = lancedb.connect(db_path)
        self.table_name = "state_embeddings"
    
    def create_or_update_table(self, embeddings, metadata, embedding_key="X_state", ...):
        # Prepare data with metadata
        data = []
        for j in range(len(embeddings)):
            record = {
                "vector": embeddings[j].tolist(),  # ← SAVED HERE!
                "cell_id": metadata.index[j],
                "embedding_key": embedding_key,
                "dataset": dataset_name or "unknown",
                **{col: metadata.iloc[j][col] for col in metadata.columns},
            }
            data.append(record)
        
        # Create or append to table
        if self.table_name in self.db.table_names():
            table = self.db.open_table(self.table_name)
            table.add(data)
        else:
            self.db.create_table(self.table_name, data=data)
```

---

## 🔄 **Complete Flow: How SE Saves Embeddings**

### **Step-by-Step Process:**

```
1. LOAD DATA
   ─────────
   Input: adata from .h5ad file
   Contains: raw gene expression

2. SE MODEL INFERENCE
   ──────────────────
   For each batch of cells:
       raw_expression → SE Model → embeddings
       [batch, n_genes] → [batch, 1536]

3. CONCATENATE ALL BATCHES
   ────────────────────────
   all_embeddings = np.concatenate(batch_embeddings)
   Shape: [n_cells, 1536]

4. SAVE TO ANNDATA (Primary)
   ─────────────────────────
   adata.obsm[emb_key] = all_embeddings
   adata.write_h5ad(output_path)
   
   Result: embeddings saved in .h5ad file!

5. OPTIONAL: SAVE TO LANCEDB
   ──────────────────────────
   IF lancedb_path is provided:
       vectordb.create_or_update_table(
           embeddings=all_embeddings,
           metadata=adata.obs
       )
   
   Result: embeddings also in vector database!
```

---

## 💻 **Command Line Usage**

### **Generate and Save Embeddings:**

```bash
# Using the SE model CLI
state emb encode \
  --input-adata-path data/input.h5ad \
  --output-adata-path data/output_with_embeddings.h5ad \
  --emb-key X_state \
  --checkpoint-path models/se_model.ckpt

# Result: output_with_embeddings.h5ad now has adata.obsm['X_state']
```

### **With LanceDB (Vector Database):**

```bash
state emb encode \
  --input-adata-path data/input.h5ad \
  --output-adata-path data/output_with_embeddings.h5ad \
  --emb-key X_state \
  --checkpoint-path models/se_model.ckpt \
  --lancedb-path ./embeddings.lancedb \
  --update-lancedb
```

---

## 🔍 **Checking If Embeddings Exist**

### **In Python:**

```python
import scanpy as sc

# Load data
adata = sc.read_h5ad('data/output_with_embeddings.h5ad')

# Check available embeddings
print("Available obsm keys:", list(adata.obsm.keys()))
# Output: ['X_hvg', 'X_state', 'X_pca', ...]

# Check if SE embeddings exist
if 'X_state' in adata.obsm:
    print("✓ SE embeddings found!")
    print(f"  Shape: {adata.obsm['X_state'].shape}")
    print(f"  Dimension: {adata.obsm['X_state'].shape[1]}")
else:
    print("✗ No SE embeddings found")

# Access embeddings
se_embeddings = adata.obsm['X_state']
print(f"First cell embedding: {se_embeddings[0][:5]}...")
```

### **Example Output:**

```
Available obsm keys: ['X_hvg', 'X_state']
✓ SE embeddings found!
  Shape: (10000, 1536)
  Dimension: 1536
First cell embedding: [-0.234, 0.567, -0.123, 0.890, 0.456]...
```

---

## 📊 **File Structure After SE Encoding**

### **Before SE Encoding:**
```
data/
├── input.h5ad
    ├── adata.X: [10000, 5000 genes]
    ├── adata.obs: metadata
    └── adata.obsm:
        └── X_hvg: [10000, 2000]  ← Only HVGs
```

### **After SE Encoding:**
```
data/
├── output_with_embeddings.h5ad
    ├── adata.X: [10000, 5000 genes]
    ├── adata.obs: metadata
    └── adata.obsm:
        ├── X_hvg: [10000, 2000]    ← Original HVGs
        └── X_state: [10000, 1536]  ← NEW! SE embeddings

embeddings.lancedb/  (if LanceDB enabled)
└── state_embeddings.lance
    └── 10,000 vectors with metadata
```

---

## 🎯 **Common Embedding Keys**

| Key | Meaning | Typical Dimension | Source |
|-----|---------|-------------------|--------|
| `X_hvg` | Highly Variable Genes | 2000 | Preprocessing |
| `X_pca` | PCA embeddings | 50-100 | Scanpy |
| `X_uce` | UCE embeddings | 1280 | UCE model |
| `X_state` | STATE embeddings | 1536 | SE model |
| `X_emb` | Generic embedding | Varies | SE model (default) |

---

## 🔄 **Integration with ST Model**

### **Workflow:**

```
STEP 1: Generate SE Embeddings (one-time)
──────────────────────────────────────────
$ state emb encode \
    --input-adata-path random.h5ad \
    --output-adata-path random_with_embeddings.h5ad \
    --emb-key X_state

Result: random_with_embeddings.h5ad has adata.obsm['X_state']

STEP 2: Train ST Model Using Embeddings
────────────────────────────────────────
$ state tx train \
    data.kwargs.embed_key=X_state \
    data.kwargs.toml_config_path=config.toml

The ST model will:
1. Load data from random_with_embeddings.h5ad
2. Use adata.obsm['X_state'] as input (not X_hvg!)
3. Train to predict perturbed embeddings
4. Use gene decoder to convert back to genes
```

---

## 💾 **Storage Efficiency**

### **Size Comparison:**

```
Original data (raw genes):
  [10000 cells, 20000 genes, float32]
  = 10000 × 20000 × 4 bytes = 800 MB

SE embeddings:
  [10000 cells, 1536 dims, float32]
  = 10000 × 1536 × 4 bytes = 61.4 MB

Compression ratio: 13x smaller!
```

### **Benefits:**
- ✅ Much smaller file size
- ✅ Faster to load
- ✅ Universal representation (works across datasets)
- ✅ Captures biological meaning

---

## 🔍 **Verify Embeddings Were Saved**

### **Quick Check Script:**

```python
import scanpy as sc
import numpy as np

# Load the output file
adata = sc.read_h5ad('output_with_embeddings.h5ad')

# Verify embeddings
print("=" * 60)
print("CHECKING SE EMBEDDINGS")
print("=" * 60)

if 'X_state' in adata.obsm:
    embeddings = adata.obsm['X_state']
    
    print(f"✓ Embeddings found!")
    print(f"  Key: X_state")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Dtype: {embeddings.dtype}")
    print(f"  Size: {embeddings.nbytes / 1024 / 1024:.2f} MB")
    
    print(f"\n  Statistics:")
    print(f"    Mean: {embeddings.mean():.4f}")
    print(f"    Std:  {embeddings.std():.4f}")
    print(f"    Min:  {embeddings.min():.4f}")
    print(f"    Max:  {embeddings.max():.4f}")
    
    print(f"\n  First cell embedding (first 10 dims):")
    print(f"    {embeddings[0, :10]}")
    
    print("\n✅ Embeddings are ready to use!")
else:
    print("✗ No embeddings found in adata.obsm['X_state']")
    print(f"  Available keys: {list(adata.obsm.keys())}")
```

---

## 📋 **Summary**

### **Question:** Does SE save embeddings somewhere?

### **Answer:** 

**YES!** The SE model saves embeddings in **two places**:

1. **Primary Storage: AnnData `obsm` dictionary**
   - Location: `adata.obsm[emb_key]`
   - Default key: `X_emb` or `X_state`
   - Format: NumPy array [n_cells, 1536]
   - Saved in: `.h5ad` file

2. **Optional Storage: LanceDB vector database**
   - Location: `./state_embeddings.lancedb/`
   - Purpose: Fast similarity search
   - Format: Table with vectors + metadata

### **How to Access:**

```python
import scanpy as sc

# Load data with embeddings
adata = sc.read_h5ad('data_with_embeddings.h5ad')

# Access embeddings
embeddings = adata.obsm['X_state']  # Shape: [n_cells, 1536]

# Use for ST model training
# The ST model reads from adata.obsm['X_state'] automatically
# when you set: embed_key=X_state
```

**The embeddings are automatically saved during the `encode_adata()` process and stored in the output .h5ad file!** 🎉

