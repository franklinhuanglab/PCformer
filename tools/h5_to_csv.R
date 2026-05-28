#!/usr/bin/env Rscript
library(Seurat)
library(data.table)
setDTthreads(8)
# ----------------------------------------------------------------------------------------
# Description: Convert an h5 matrix to a csv file
# 
# ----------------------------------------------------------------------------------------
# USER MODIFIED VARIABLES
dir <- "data/scRNA"

basename <- "filtered_feature_bc_matrix"

filename <- paste0(basename, "_matrix.h5")


# ----------------------------------------------------------------------------------------
# DATA LOADING AND INITIALIZATIONS

h5_file <- file.path(dir, filename)
output_file <- file.path(dir, paste0(basename, ".csv.gz"))
# output_file <- file.path(paste0(basename, ".csv.gz"))

mat <- Read10X_h5(h5_file)

# seurat_obj <- CreateSeuratObject(counts = mat)


# ----------------------------------------------------------------------------------------

# Transpose: genes as columns, barcodes as rows
mat_t <- t(mat)

# Convert to df and include barcodes as a column
df <- as.data.frame(as.matrix(mat_t))
df$barcode <- rownames(mat_t)

# Move barcode to first column
df <- df[, c("barcode", setdiff(names(df), "barcode"))]

# Write to gzipped CSV
# write.csv(df, file = gzfile(output_file), row.names = FALSE, quote = FALSE)
fwrite(df, output_file, compress = "gzip")


cat("Saved transposed matrix to:", output_file, "\n")
