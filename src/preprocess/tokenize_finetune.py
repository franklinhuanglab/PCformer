#!/usr/bin/env python3
import os, sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)
import time
import json
import yaml
import logging
import gc
import psutil
import functools
import pandas as pd
import numpy as np
from datasets import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.cuda.amp as amp
from torch.optim.lr_scheduler import StepLR
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
# from src.preprocess import GeneExpressionDatasetRB
# from src.preprocess import DataCollatorForGeneModeling
from src.model import AtlasModelRankBased, ModelArgs
from src.preprocess import GeneTokenizer
from src import *


def preprocess(tokenizer, examples):
    """
    Tokenizes one example (e.g., cell) at a time using the tokenizer module.

    Args:
        tokenizer (GeneTokenizer): Tokenizer for mapping gene names to token IDs
        examples (dict): Dictionary containing `gene_names` and `gene_expressions`

    Returns:
        dict: {'tokenized_genes': <list of token IDs>}
    """
    return {
        'tokenized_genes': tokenizer(
            examples['gene_names'],
            torch.tensor(examples['gene_expressions'], dtype=torch.float32).cpu().numpy()
        )
    }

def load_expression_matrix_auto(input_file: str) -> pd.DataFrame:
    """
    Load expression matrix:
      - Prefer loading split numpy files (`input_file` directory and basename):
          <base>_values.npz (expects key 'float_values')
          <base>_index.npy
          <base>_columns.npy
      - Else fall back to CSV/TSV (optionally gzipped)
    Returns a DataFrame with cells as rows and genes as columns, and index = barcodes.
    """
    # Normalize base path
    base = input_file.replace(".csv.gz", "").replace(".csv", "")
    #base = input_file
    #for ext in (".csv.gz", ".tsv.gz", ".csv", ".tsv"):
    #    if base.endswith(ext):
    #        base = base[: -len(ext)]
    #        break

    values_path  = f"{base}_values.npz"
    index_path   = f"{base}_index.npy"
    columns_path = f"{base}_columns.npy"

    if os.path.exists(values_path) and os.path.exists(index_path) and os.path.exists(columns_path):
        debug_log(f"Using numpy split files:\n  {values_path}\n  {index_path}\n  {columns_path}")
        float_values = np.load(values_path)["float_values"]
        index = np.load(index_path, allow_pickle=True)
        columns = np.load(columns_path, allow_pickle=True)
        df = pd.DataFrame(float_values, index=index, columns=columns)
    else:
        debug_log(f"Numpy split files not found. Loading expression matrix: {input_file}")
            df = pd.read_csv(
            input_file,
            sep=r"[,\t]",
            engine="python",
            compression="gzip" if input_file.endswith(".gz") else None
        )

    # Make barcode index consistent
    if "CellName" in df.columns:
        df = df.set_index("CellName")
    elif "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "CellName"}).set_index("CellName")
    else:
        # assume first col is barcode
        df = df.rename(columns={df.columns[0]: "CellName"}).set_index("CellName")

    return df

def generate_dataset_rows(cell_barcodes, gene_expressions, gene_names):
    """
    Generates per cell gene expression data rows for Dataset.from_generator.

    Handles memory usage by streaming each row instead of storing all data at once using a generator.
    
    Each row contains:
      - barcode
      - gene_names
      - clipped gene expression vector
      
    Args:
        cell_barcodes (list): Barcodes corresponding to each cell
        gene_expressions (np.ndarray): Matrix of gene expression values (cells x genes)
        gene_names (list): Names of all genes shared across all cells

    Returns:
        dict: {'barcode', 'gene_names', 'gene_expressions'}
    """
    max_32bit_int_minus_1 = np.iinfo(np.int32).max - 1  # 2147483646
    for i in tqdm(range(len(cell_barcodes)), desc="Generating dataset rows", miniters=500): # mininterval=5.0, 
        try:
            capped_data = np.clip(
                gene_expressions[i], 
                a_min=None, 
                a_max=max_32bit_int_minus_1
            )
            yield {'barcode': cell_barcodes[i], 'gene_names': gene_names, 'gene_expressions': capped_data}
        except Exception as e:
            print(f"Error processing cell {cell_barcodes[i]}: {e}")

def create_and_cache_tokenized_dataset(debug=False, use_cache=True, tokenizer=None):
    """
    Loads raw gene expression data, runs preprocessing, and caches tokenized datasets.

    Steps:
      - Load matrix from CSV or .npz cache
      - Stream rows into Hugging Face Dataset
      - Split into train/test sets
      - Tokenize gene sequences per cell
      - Optionally cache barcodes and datasets to disk

    Args:
        debug (bool): Enables debug logging
        use_cache (bool): If True, loads from/saves cached expression and token datasets to disk
        tokenizer (GeneTokenizer): Required tokenizer for gene to token conversion

    Returns:
        tuple: (tokenized_train_set, tokenized_test_set), both Hugging Face datasets
    """
    assert tokenizer is not None, "Tokenizer must be provided."
    
    cache_prefix     = f"{prefix}/debug" if not use_cache else f"{prefix}"
    train_cache_file = f"{cache_prefix}/train"
    test_cache_file  = f"{cache_prefix}/test"
    
    with open(genes_file, 'r') as gn:
        gene_names = [line.strip() for line in gn]
            
    if use_cache and os.path.exists(train_cache_file) and os.path.exists(test_cache_file):
        debug_log("Loading tokenized and split datasets from cache.")
        return Dataset.load_from_disk(train_cache_file), Dataset.load_from_disk(test_cache_file)

    debug_log("Cache tokens not found. Generating the gene expression dataset.")

    expressions_cache_file = f"{cache_prefix}_expressions.npz"
    os.makedirs(os.path.dirname(expressions_cache_file), exist_ok=True)


    # READ INPUT MATRIX
    if use_cache and os.path.exists(expressions_cache_file):
        debug_log("Loading cached gene expressions from disk.") 
        data = np.load(expressions_cache_file, allow_pickle=True)
        
        gene_expressions = [row for row in list(data['expressions'])] # list(data["expressions"]) # 
        cell_barcodes    = list(data['barcodes'])
        gene_names       = list(data['gene_names'])
        cell_labels      = list(data['labels'])
        del data
    else:
        debug_log(f"Importing dataset from {input_file}")
        # Load matrix. Prefer numpy split files if present
        df = load_expression_matrix_auto(input_file)   # returns index=barcodes, cols=genes
        
        # Holdout exclusion
        if holdout_file and str(holdout_file).lower() != "null" and os.path.exists(holdout_file):
            with open(holdout_file, "r") as f:
                holdout = set(x.strip() for x in f if x.strip())
            before_n = df.shape[0]
            df = df.loc[~df.index.astype(str).isin(holdout)]
            after_n = df.shape[0]
            debug_log(f"Excluded {before_n - after_n} holdout cells using {holdout_file}. Remaining: {after_n}")
        
        # Convert format
        cell_barcodes = df.index.astype(str).tolist()
        gene_names = df.columns.astype(str).tolist()
        gene_expressions = df.to_numpy(dtype=np.float32)


    debug_log(f"Loaded the matrix with {len(cell_barcodes)} cells and {len(gene_names)} genes.")
        
    labels = pd.read_csv(
        labels_file, 
        sep=r'[,\t]', 
        engine='python')
    debug_log(f"Annotated {len(set(labels['ID'].tolist()))} unique cell type labels.")

    # Rename first metadata column to CellName
    if "CellName" not in labels.columns:
        if "Unnamed: 0" in labels.columns:
            labels.rename(columns={"Unnamed: 0": "CellName"}, inplace=True)
        else:
            labels.rename(columns={labels.columns[0]: "CellName"}, inplace=True)
    
    # Rename first metadata column to ID
    if "ID" not in labels.columns:
        for alt in ("label","Label","cell_type","CellType","annotation","Annotation"):
            if alt in labels.columns:
                labels.rename(columns={alt: "ID"}, inplace=True)
                break
        else:
            labels.rename(columns={labels.columns[1]: "ID"}, inplace=True)

    label_dict = dict(zip(labels["CellName"], labels["ID"]))
    cell_labels = [label_dict.get(barcode, 'Unknown') for barcode in cell_barcodes]  # "Unknown" if not found

    # Set the cap to the maximum 32-bit signed integer value
    max_32bit_int_minus_1 = np.iinfo(np.int32).max - 1  # 2147483646

    processed_data = [
        (np.clip(exp, a_min=None, a_max=max_32bit_int_minus_1), barcode, label)
        for exp, barcode, label in tqdm(zip(gene_expressions, cell_barcodes, cell_labels), 
                                        total=len(cell_barcodes), desc="Processing gene expressions")
    ]

    gene_expressions, barcodes, cell_labels = zip(*processed_data)
    
    debug_log(f"Processed {len(gene_expressions)} cells with {len(gene_names)} genes.")

    # Uncomment to cache expression data
    """ Saves expressions to disk. Huge file
    if use_cache and not os.path.exists(expressions_cache_file):
        debug_log("Saving gene expressions to cache.")
        np.savez(
            expressions_cache_file,
            expressions=np.array(gene_expressions, dtype=np.float32),
            barcodes=np.array(barcodes),
            gene_names=np.array(gene_names),
            labels=np.array(cell_labels)
        )
    """

    # Wrap gene expression data into a streaming generator
    def gen():
        for i in range(len(gene_expressions)):
            yield {
                'gene_names'      : gene_names, 
                'gene_expressions': gene_expressions[i], 
                'cell_label'      : cell_labels[i],
                'barcode'         : barcodes[i]
            }
        
    dataset = Dataset.from_generator(gen)

    # Split dataset into training and testing sets
    split = dataset.train_test_split(test_size=0.10, seed=42)
    train_dataset = split["train"]
    test_dataset   = split["test"]
    
    # Extract and save barcode metadata
    train_barcodes = [x["barcode"] for x in train_dataset]
    test_barcodes   = [x["barcode"] for x in test_dataset]
    
    metadata_dir = os.path.join(cache_prefix, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    os.makedirs(prefix, exist_ok=True)
    
    barcode_files = {
        "train_barcodes.txt": train_barcodes,
        "test_barcodes.txt": test_barcodes,
    }
    for filename, barcodes in barcode_files.items():
        with open(os.path.join(metadata_dir, filename), "w") as f:
            f.write("\n".join(map(str, barcodes)))
    
    debug_log("Tokenizing the train and test genes.")
    
    tokenized_train_set = train_dataset.map(
        lambda examples: preprocess(tokenizer, examples),
        num_proc=num_proc
    )
    tokenized_test_set = test_dataset.map(
        lambda examples: preprocess(tokenizer, examples),
        num_proc=num_proc
    )
    
    # Remove raw columns
    columns_to_remove = ["gene_names", "gene_expressions", "barcode"]
    tokenized_train_set = tokenized_train_set.remove_columns(columns_to_remove)
    tokenized_test_set   = tokenized_test_set.remove_columns(columns_to_remove)
    
    # Save tokenized datasets
    tokenized_train_set.save_to_disk(train_cache_file)
    tokenized_test_set.save_to_disk(test_cache_file)
    
    debug_log("_______________\n")
    debug_log(f"Datasets cached: {train_cache_file}, {test_cache_file}\n")
    
    return tokenized_train_set, tokenized_test_set


@track_performance
def start_loop(debug=False, use_cache=True):
    """
    Initializes tokenizer and creates tokenized datasets.

    Args:
        debug (bool): Enables debug logging
        use_cache (bool): Use cached datasets if available
    """
    with open(genes_file, "r") as gn:
        genes_list = [ln.strip() for ln in gn if ln.strip()]
    tokenizer = GeneTokenizer(genes=genes_list)
    
    train_dataset, test_dataset = create_and_cache_tokenized_dataset(debug=debug, use_cache=use_cache, tokenizer=tokenizer)
    debug_log("DataLoaders setup complete. \n")

    debug_log("Model, optimizer, and loss function setup complete. \n")

if __name__ == "__main__":
    if len(sys.argv) - 1 != 7:
        print(f"Error: Expected 7 arguments, but got {len(sys.argv) - 1}.")
        print("Usage: python preprocess.py <input_file> <genes_file> <prefix> <labels_file> <config_file> <num_proc> [holdout_file|NULL]")
        sys.exit(1)

    input_file  = sys.argv[1]
    genes_file  = sys.argv[2]
    prefix      = sys.argv[3]
    labels_file = sys.argv[4]
    holdout_file = sys.argv[5] if (len(sys.argv) - 1 == 7) else "NULL"
    config_file = sys.argv[6]
    num_proc    = int(sys.argv[7])
    # num_workers = os.cpu_count()

    load_config(config_file)

    log_file = initialize_logging(prefix, context="tokenize_tune")

    start_loop(debug=config['debug'], use_cache=config['use_cache'])
