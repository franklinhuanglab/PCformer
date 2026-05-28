#!/usr/bin/env python3
import scanpy as sc
import pandas as pd
# ----------------------------------------------------------------------------------------
# Description: Imports h5ad expression file and extracts matching barcodes and class IDs
# 
# ----------------------------------------------------------------------------------------

# USER MODIFIED VARIABLES
input_dir = "filtered_feature_bc_matrix.h5ad"
output_dir = "file_barcode_subcluster.csv"


# ----------------------------------------------------------------------------------------

adata = sc.read_h5ad(data_dir)

# Create a DataFrame with barcodes and subcluster IDs
barcode_subcluster = pd.DataFrame({
    "barcode": adata.obs_names,
    "subcluster_ID": adata.obs["subcluster_ID"].values
})

print(barcode_subcluster.head())

barcode_subcluster.to_csv(output_dir, sep="\t", index=False)
