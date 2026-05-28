#!/usr/bin/env python3
import re
import pandas as pd
import numpy as np
# ----------------------------------------------------------------------------------------
# Description: For faster import, convert an expression matrix into 3 numpy files: 
#              *_columns.npy, *_index.npy, *_values.npz
# 
# ----------------------------------------------------------------------------------------
# USER MODIFIED VARIABLES
directory = '/home/data/scRNA'
prefix = 'Atlas'


# ----------------------------------------------------------------------------------------
# DATA LOADING AND INITIALIZATIONS
matrix_file = f'{directory}/{prefix}_matrix.csv.gz'
float_np_file = f'{directory}/{prefix}_values.npz'
index_file = f'{directory}/{prefix}_index.npy'
columns_file = f'{directory}/{prefix}_columns.npy'


# ----------------------------------------------------------------------------------------
# HELPERS
def import_full_matrix(data):
    df_full = pd.read_csv(data, index_col=0, compression='gzip')
    
    float_values = df_full.values
    index = df_full.index.to_numpy()
    columns = df_full.columns.to_numpy()
    
    np.savez_compressed(float_np_file, float_values=float_values)
    np.save(index_file, index)
    np.save(columns_file, columns)

    return df_full


# ----------------------------------------------------------------------------------------

#df_full = import_decomposed_df(float_np_file, index_file, columns_file)
df = import_full_matrix(matrix_file)


print("Finished splitting dataframe into Numpy column, index, and value files.")

