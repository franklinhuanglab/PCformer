#!/usr/bin/env python3
# ========================================================================================
# Authors: Keila Velazquezk-Arcelay
# Updated: 2025-10-16
# Description: Subset a matrix dataset to a fixed fraction of the rows.
#              Filter Metadata file based on kept barcodes.
# 
# ========================================================================================
import numpy as np
import pandas as pd
import time

start = time.time()

np.random.seed(42)

# ========================================================================================
# USER MODIFIED VARIABLES
data_prefix = '/home/data/spatial/Xenium_AA'

pct = 0.05


# ========================================================================================
# DATA LOADING AND INITIALIZATIONS

# First column (cell names) as index
print("Importing dataset from: ", data_prefix)
df = pd.read_csv(f'{data_prefix}_Matrix.csv.gz', index_col=0, compression='gzip')
md = pd.read_csv(f'{data_prefix}_Metadata.tsv', index_col=0, sep='\t')


# ========================================================================================
# Compute N% of the dataset
subset_size = int(pct * df.shape[0])
print("Subset size will be: ", subset_size)

# Randomly select N% of the total rows
print(f"Randomly selecting {pct}% rows.")
random_indices = np.random.choice(df.shape[0], subset_size, replace=False)

# Subset both data and index
subset_df = pd.DataFrame(
                df.values[random_indices], 
                columns=df.columns, 
                index=df.index[random_indices]
            )

print("Subset shape:", subset_df.shape)

print(subset_df.head())

# Output
print("Saving subset dataset to: ", data_prefix)
subset_df.to_csv(f'{data_prefix}_{int(pct*100)}pct_matrix.csv.gz', compression='gzip')


print("Filtering metadata file.")
md_filt = md.reindex(subset_df.index).reset_index()
# If md was loded without specifying index col
#md_filt = md.set_index("barcode").reindex(subset_df.index).reset_index()

print("Saving metadata to: ", data_prefix)
md_filt.to_csv(f'{data_prefix}_{int(pct*100)}pct_metadata.csv', index=False)



end = time.time()
print(f"Hybrid NumPy-Pandas time: {end - start:.4f} seconds")  # ~0.08s
