# Data Investigation Guide

You're looking at: `examples/random.h5ad`

## 📋 What You Found

```python
random_data.obs.keys()
# Output: Index(['cell_type', 'target_gene', 'batch_var'], dtype='object')
```

These are your **metadata columns** (information about each cell).

---

## 🔍 Commands to Investigate Further

### **1. Basic Structure**

```python
import scanpy as sc

# Load data
adata = sc.read_h5ad('examples/random.h5ad')

# Basic info
print(f"Number of cells: {adata.n_obs}")
print(f"Number of genes: {adata.n_vars}")
print(f"Shape: {adata.shape}")  # (cells, genes)
```

---

### **2. Explore Metadata (obs)**

```python
# Show first few rows
print(adata.obs.head())

# Show all metadata columns
print(adata.obs.keys())

# See what's in each column
print("\nTarget genes (perturbations):")
print(adata.obs['target_gene'].value_counts())

print("\nCell types:")
print(adata.obs['cell_type'].value_counts())

print("\nBatches:")
print(adata.obs['batch_var'].value_counts())
```

---

### **3. Explore Gene Expression Data**

```python
# Check what data is available
print("Main data (adata.X):")
print(f"  Shape: {adata.X.shape if adata.X is not None else 'None'}")
print(f"  Type: {type(adata.X)}")

print("\nAdditional data (adata.obsm):")
print(f"  Keys: {list(adata.obsm.keys())}")

# Check HVG data (used for training)
if 'X_hvg' in adata.obsm:
    print(f"\nHVG data (X_hvg):")
    print(f"  Shape: {adata.obsm['X_hvg'].shape}")
    print(f"  Number of genes: {adata.obsm['X_hvg'].shape[1]}")
```

---

### **4. Look at Single Cells**

```python
# Get one control cell
control_cell = adata[adata.obs['target_gene'] == 'TARGET1'][0]
print("Control cell metadata:")
print(control_cell.obs)

print("\nControl cell gene expression:")
print(control_cell.obsm['X_hvg'][0])

# Get one perturbed cell
perturbed_cell = adata[adata.obs['target_gene'] == 'TARGET2'][0]
print("\nPerturbed cell metadata:")
print(perturbed_cell.obs)

print("\nPerturbed cell gene expression:")
print(perturbed_cell.obsm['X_hvg'][0])
```

---

### **5. Filter and Select Data**

```python
# Get all control cells
control_cells = adata[adata.obs['target_gene'] == 'TARGET1']
print(f"Control cells: {len(control_cells)}")

# Get all TARGET2 cells
target2_cells = adata[adata.obs['target_gene'] == 'TARGET2']
print(f"TARGET2 cells: {len(target2_cells)}")

# Get cells from a specific cell type
ct1_cells = adata[adata.obs['cell_type'] == 'CT1']
print(f"CT1 cells: {len(ct1_cells)}")

# Combine filters (TARGET2 in CT1)
target2_ct1 = adata[(adata.obs['target_gene'] == 'TARGET2') & 
                     (adata.obs['cell_type'] == 'CT1')]
print(f"TARGET2 in CT1: {len(target2_ct1)}")
```

---

### **6. Compare Perturbations**

```python
import numpy as np

# Get control and perturbed samples
control = adata[adata.obs['target_gene'] == 'TARGET1']
target2 = adata[adata.obs['target_gene'] == 'TARGET2']

# Get mean expression for each
control_mean = control.obsm['X_hvg'].mean(axis=0)
target2_mean = target2.obsm['X_hvg'].mean(axis=0)

# Calculate difference
difference = target2_mean - control_mean

print("Mean gene expression comparison:")
print(f"Control (TARGET1): {control_mean}")
print(f"Perturbed (TARGET2): {target2_mean}")
print(f"Difference: {difference}")

# Count changed genes
changed_genes = np.sum(np.abs(difference) > 0.1)
print(f"\nGenes changed (|diff| > 0.1): {changed_genes}/{len(difference)}")
```

---

### **7. Check Gene Names**

```python
# Check if gene names are available
print("Gene names:")
print(adata.var_names)

# If you want gene names for HVG specifically
print("\nFirst 10 gene names:")
for i, gene in enumerate(adata.var_names[:10]):
    print(f"  {i}: {gene}")
```

---

### **8. Statistical Summary**

```python
import pandas as pd

# Summary of all perturbations
summary = []
for pert in adata.obs['target_gene'].unique():
    cells = adata[adata.obs['target_gene'] == pert]
    expr = cells.obsm['X_hvg']
    
    summary.append({
        'perturbation': pert,
        'n_cells': len(cells),
        'mean_expr': expr.mean(),
        'std_expr': expr.std(),
        'min_expr': expr.min(),
        'max_expr': expr.max(),
    })

summary_df = pd.DataFrame(summary)
print(summary_df)
```

---

### **9. Visualize Data**

```python
import matplotlib.pyplot as plt

# Plot gene expression distribution
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(adata.obsm['X_hvg'].flatten(), bins=50)
plt.xlabel('Gene Expression')
plt.ylabel('Count')
plt.title('All Gene Expression Distribution')

plt.subplot(1, 3, 2)
control_expr = adata[adata.obs['target_gene'] == 'TARGET1'].obsm['X_hvg']
plt.hist(control_expr.flatten(), bins=50, alpha=0.7, label='TARGET1')
plt.xlabel('Gene Expression')
plt.ylabel('Count')
plt.title('Control Distribution')
plt.legend()

plt.subplot(1, 3, 3)
target2_expr = adata[adata.obs['target_gene'] == 'TARGET2'].obsm['X_hvg']
plt.hist(target2_expr.flatten(), bins=50, alpha=0.7, label='TARGET2', color='red')
plt.xlabel('Gene Expression')
plt.ylabel('Count')
plt.title('TARGET2 Distribution')
plt.legend()

plt.tight_layout()
plt.savefig('gene_expression_distribution.png')
print("Saved: gene_expression_distribution.png")
```

---

### **10. Quick Data Inspection Script**

```python
"""Quick inspection script - paste this into Python/Jupyter"""
import scanpy as sc
import numpy as np

adata = sc.read_h5ad('examples/random.h5ad')

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Shape: {adata.shape} (cells × genes)")
print(f"Metadata columns: {list(adata.obs.keys())}")
print(f"Expression data keys: {list(adata.obsm.keys())}")

print("\n" + "=" * 60)
print("PERTURBATIONS")
print("=" * 60)
for pert, count in adata.obs['target_gene'].value_counts().items():
    print(f"{pert:10s}: {count:5d} cells")

print("\n" + "=" * 60)
print("CELL TYPES")
print("=" * 60)
for ct, count in adata.obs['cell_type'].value_counts().items():
    print(f"{ct:10s}: {count:5d} cells")

print("\n" + "=" * 60)
print("GENE EXPRESSION (X_hvg)")
print("=" * 60)
print(f"Shape: {adata.obsm['X_hvg'].shape}")
print(f"Mean: {adata.obsm['X_hvg'].mean():.4f}")
print(f"Std:  {adata.obsm['X_hvg'].std():.4f}")
print(f"Min:  {adata.obsm['X_hvg'].min():.4f}")
print(f"Max:  {adata.obsm['X_hvg'].max():.4f}")

print("\n" + "=" * 60)
print("SAMPLE CELL")
print("=" * 60)
sample = adata[0]
print("Metadata:")
for key in adata.obs.keys():
    print(f"  {key}: {sample.obs[key].values[0]}")
print(f"\nGene expression (first 5): {sample.obsm['X_hvg'][0][:5]}")
```

---

## 🎯 **Understanding Your Metadata**

### **`cell_type`** (5 unique values)
```
What it is: The cell line or cell type
Values: CT1, CT2, CT3, CT4, CT5

Why it matters:
- Different cell types may respond differently to perturbations
- Used for matching control cells (same cell type)
- Used for train/test splits (e.g., hold out CT3)
```

### **`target_gene`** (5 unique values)
```
What it is: The perturbation/intervention applied
Values: TARGET1, TARGET2, TARGET3, TARGET4, TARGET5

Why it matters:
- TARGET1 = Control (unperturbed baseline)
- TARGET2-5 = Different perturbations/drugs/interventions
- This is what the model learns to predict!
```

### **`batch_var`** (1 unique value)
```
What it is: Experimental batch ID
Values: 1 (only one batch in your data)

Why it matters:
- Used for batch effect correction
- In your case, all cells are from same batch
- So no batch correction needed
```

---

## 💡 **Quick Investigation Commands**

Copy-paste these into Python:

```python
import scanpy as sc
adata = sc.read_h5ad('examples/random.h5ad')

# Quick look at metadata
adata.obs.head(10)

# Count everything
print(adata.obs['target_gene'].value_counts())
print(adata.obs['cell_type'].value_counts())

# Look at one cell
print(adata[0].obs)
print(adata[0].obsm['X_hvg'][0])

# Compare control vs perturbed
ctrl = adata[adata.obs['target_gene'] == 'TARGET1'][0].obsm['X_hvg'][0]
pert = adata[adata.obs['target_gene'] == 'TARGET2'][0].obsm['X_hvg'][0]
print(f"Control: {ctrl}")
print(f"Perturbed: {pert}")
print(f"Difference: {pert - ctrl}")
```

---

## 🚀 **Next Steps**

1. **Explore the data** using commands above
2. **Understand the structure** of your cells
3. **Visualize patterns** in gene expression
4. **Compare perturbations** to see differences
5. **Run training** when ready: `make train`

Happy investigating! 🔬


