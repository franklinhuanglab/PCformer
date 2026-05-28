#!/usr/bin/env python3
import os
import time
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
# ----------------------------------------------------------------------------------------
# Description: Extract the expression matrix from an h5/h5ad file and export as csv.
#              Optionally, subsets the data to a fraction of the rows.
# 
# ----------------------------------------------------------------------------------------
# USER MODIFIED VARIABLES

root_dir = "home/data/scRNA"
input_file = f"{root_dir}/filtered_feature_bc_matrix.h5ad"
output_file = f"{root_dir}/filtered_feature_bc_matrix.csv.gz"

# Set to None or 0 to disable subsetting.
# If 0 < subset_fraction <= 1: take that fraction of rows.
subset_fraction = None
random_seed = 42

np.random.seed(random_seed)

start = time.time()

# ----------------------------------------------------------------------------------------
# DATA LOADING AND INITIALIZATIONS

if input_file.endswith(".h5ad"):
    print(f"Loading AnnData (.h5ad) from {input_file}")
    adata = anndata.read_h5ad(input_file)
    df = adata.to_df()  # genes = columns, cells = index
elif input_file.endswith(".h5"):
    print(f"Loading 10x HDF5 (.h5) from {input_file}")
    adata = sc.read_10x_h5(input_file)  # returns AnnData
    df = adata.to_df()
else:
    print(f"Loading table (CSV/TSV) from {input_file}")
    comp = 'gzip' if input_file.endswith('.gz') else None
    df = pd.read_csv(input_file, index_col=0, sep=r'[,\t]', engine='python', compression='gzip' if input_file.endswith(".gz") else None)


# ----------------------------------------------------------------------------------------
# SUBSET

def subset_dataset(df, subset_fraction, random_seed=None):
    if subset_fraction in (None, 0, 0.0):
        print("Subsetting disabled; using full data.")
        return df

    n = len(df)
    if 0 < subset_fraction <= 1:
        k = max(1, int(round(subset_fraction * n)))
    elif subset_fraction > 1:
        k = int(subset_fraction)
    else:
        raise ValueError("subset_fraction must be None/0, in (0,1], or >=1.")

    k = min(k, n)
    print(f"Subsetting: selecting {k} of {n} rows.")
    return df.sample(n=k, random_state=random_seed)

df_out = subset_dataset(df, subset_fraction, random_seed)

"""
subset_size = int(subset_fraction * df.shape[0])
random_indices = np.random.choice(df.shape[0], subset_size, replace=False)

subset_df = pd.DataFrame(
    df.values[random_indices],
    columns=df.columns,
    index=df.index[random_indices]
)
"""

# ----------------------------------------------------------------------------------------
# SAVE
df_out.to_csv(output_file, compression='gzip' if output_file.endswith(".gz") else None)

"""
print("Subset shape:", subset_df.shape)
print(subset_df.head())
"""
end = time.time()

print(f"Hybrid NumPy-Pandas time: {end - start:.4f} seconds")
