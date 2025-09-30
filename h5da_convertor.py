#!/usr/bin/env python3
import scanpy as sc
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path

IN_H5AD  = "mixed_for_competition/competition/prediction_new.h5ad"
GENES_CSV = "competition_support_set/gene_names.csv"
OUT_H5AD = "mixed_for_competition/competition/prediction_sparse_subset.h5ad"

CHUNK = 2000  # tune: smaller if you still hit OOM

print("• Reading input AnnData in backed mode…")
adata_b = sc.read_h5ad(IN_H5AD, backed="r")  # AnnDataBacked

print("• Loading target gene list…")
genes = pd.read_csv(GENES_CSV, header=None).iloc[:,0].astype(str).tolist()

# map requested genes to indices present in adata
present = [g for g in genes if g in adata_b.var_names]
if not present:
    raise ValueError("None of the requested genes were found in adata.var_names")
idx = np.array([adata_b.var_names.get_loc(g) for g in present], dtype=int)
print(f"  - {len(present)} / {len(genes)} genes found in the matrix")

n_cells = adata_b.n_obs
print(f"• Cells: {n_cells:,} | Genes kept: {len(idx):,}")

# Build X sparsely by chunks
rows = []
for start in range(0, n_cells, CHUNK):
    end = min(start + CHUNK, n_cells)
    # slice from disk; result is a dense numpy array of manageable size
    block = adata_b.X[start:end, idx]

    # ensure float32 to cut memory in half
    if block.dtype != np.float32:
        block = block.astype(np.float32, copy=False)

    rows.append(sp.csr_matrix(block))  # convert this chunk to sparse
    print(f"  - processed rows {start}:{end}")

# stack all chunks into a single CSR
X = sp.vstack(rows, format="csr")
del rows  # free chunk list

# Bring minimal metadata to memory
obs = adata_b.obs.copy()
var = adata_b.var.loc[present].copy()

# fix duplicate cell IDs to avoid downstream crashes
if not obs.index.is_unique:
    obs.index = pd.Index(obs.index).astype(str)
    obs.index = pd.Index(np.where(obs.index.duplicated(), 
                                  obs.index + "_" + pd.Series(range(len(obs))).astype(str),
                                  obs.index))
obs_names = obs.index.astype(str)
var_names = var.index.astype(str)

# Create a new in-memory AnnData and write
from anndata import AnnData
adata_new = AnnData(X=X, obs=obs, var=var)
adata_new.obs_names = obs_names
adata_new.var_names = var_names

print("• Writing sparse subset .h5ad (gzip)…")
adata_new.write(OUT_H5AD, compression="gzip")
print(f"✓ Done: {OUT_H5AD}")