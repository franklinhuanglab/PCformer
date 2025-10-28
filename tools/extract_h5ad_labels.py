#!/usr/bin/env python3
# ========================================================================================
# Authors: Keila Velazquezk-Arcelay
# Updated: 2025-10-16
# Description: Imports h5ad expression file and extracts matching barcodes and class IDs
# 
# ========================================================================================
import scanpy as sc
import pandas as pd
# ========================================================================================
# USER MODIFIED VARIABLES
input_dir = "all_sn_integrated_nlayers2_nlatent10_final_demux.h5ad"
output_dir = "snRNA_barcode_subcluster.csv"


# ========================================================================================

adata = sc.read_h5ad(data_dir)

# Create a DataFrame with barcodes and subcluster IDs
barcode_subcluster = pd.DataFrame({
    "barcode": adata.obs_names,
    "subcluster_ID": adata.obs["subcluster_ID"].values
})

print(barcode_subcluster.head())

barcode_subcluster.to_csv(output_dir, sep="\t", index=False)
