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
from torch import nn, optim
import torch.cuda.amp as amp
from tqdm import tqdm
from src.preprocess import GeneTokenizer
from src import *


def save_split_data(
    cache_prefix,
    use_cache,
    tokenized_train_set,
    tokenized_test_set,
    holdout_barcodes
):
    """
    Saves tokenized train/test datasets and their corresponding barcode metadata to disk. 
    Also saves holdout barcodes without tokenizing or caching them.

    Steps:
      - Extracts barcodes from tokenized train/test datasets
      - Writes train, test, and holdout barcodes to `metadata/`
      - Saves Hugging Face `Dataset` objects under `train/` and `test/`

    Args:
        cache_prefix (str): Base path for all cache outputs
        use_cache (bool): Whether to persist datasets to disk
        tokenized_train_set (datasets.Dataset): Tokenized train dataset
        tokenized_test_set (datasets.Dataset): Tokenized test dataset
        holdout_barcodes (list[str]): Barcodes reserved strictly for inference

    Returns:
        tuple: Paths to saved train and test dataset directories
    """
    metadata_dir = os.path.join(cache_prefix, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)

    train_barcodes_path   = os.path.join(metadata_dir, "train_barcodes.txt")
    test_barcodes_path    = os.path.join(metadata_dir, "test_barcodes.txt")
    holdout_barcodes_path = os.path.join(metadata_dir, "holdout_barcodes.txt")

    debug_log("Saving the split barcodes.")

    train_barcodes = [ex["barcode"] for ex in tokenized_train_set]
    test_barcodes  = [ex["barcode"] for ex in tokenized_test_set]

    with open(train_barcodes_path, "w") as f:
        f.write("\n".join(train_barcodes))

    with open(test_barcodes_path, "w") as f:
        f.write("\n".join(test_barcodes))

    with open(holdout_barcodes_path, "w") as f:
        f.write("\n".join(holdout_barcodes))

    debug_log(
        "Barcode files saved:\n"
        f"- {train_barcodes_path}\n"
        f"- {test_barcodes_path}\n"
        f"- {holdout_barcodes_path}"
    )

    # Cache only tokenized gene sequences
    # Remove raw gene inputs before saving
    cols_to_keep = [
        c for c in tokenized_train_set.column_names
        if c not in ["gene_names", "gene_expressions"]
    ]

    tokenized_train_set = tokenized_train_set.select_columns(cols_to_keep)
    tokenized_test_set  = tokenized_test_set.select_columns(cols_to_keep)

    train_cache_path = None
    test_cache_path  = None

    if use_cache:
        debug_log("Saving tokenized datasets to disk.")
        train_cache_path = os.path.join(cache_prefix, "train")
        tokenized_train_set.save_to_disk(train_cache_path)

        test_cache_path = os.path.join(cache_prefix, "test")
        tokenized_test_set.save_to_disk(test_cache_path)

    return train_cache_path, test_cache_path



def preprocess(tokenizer, examples):
    """
    Tokenizes one example (e.g., cell) at a time using the tokenizer module.

    Args:
        tokenizer (GeneTokenizer): Tokenizer instance for mapping gene names to token IDs
        examples (dict): A dictionary with `gene_names` and `gene_expressions`

    Returns:
        dict: {'tokenized_genes': <list of token IDs>}
    """
    return {
        'tokenized_genes': tokenizer(
            examples['gene_names'],
            torch.tensor(examples['gene_expressions'], dtype=torch.float32).cpu().numpy()
        )
    }

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
            capped_data = np.clip(gene_expressions[i], a_min=None, a_max=max_32bit_int_minus_1)
            yield {
                'barcode': cell_barcodes[i], 
                'gene_names': gene_names, 
                'gene_expressions': capped_data}
        except Exception as e:
            print(f"Error processing cell {cell_barcodes[i]}: {e}")
                
@track_performance
def create_and_cache_tokenized_dataset(debug=False, use_cache=True, tokenizer=None):
    """
    Loads raw gene expression data, runs preprocessing, and caches tokenized datasets.

    Steps:
      - Load matrix from CSV or .npz cache
      - Stream rows into Hugging Face dataset
      - Split into train/test/holdout sets
      - Tokenize gene sequences per cell
      - Cache barcodes and datasets to disk

    Args:
        debug (bool): Enables debug loggings
        use_cache (bool): If True, loads from/saves cached expression and token datasets to disk
        tokenizer (GeneTokenizer): Required tokenizer for gene to token conversion

    Returns:
        tuple: (tokenized_train_set, tokenized_test_set), both Hugging Face datasets
    """
    assert tokenizer is not None, "Tokenizer must be provided."

    cache_prefix = f"{prefix}/debug" if not use_cache else f"{prefix}"
    train_cache_file = f"{cache_prefix}/train"
    test_cache_file = f"{cache_prefix}/test"

    with open(genes_file, 'r') as gn:
        gene_names = [line.strip() for line in gn]
        
    if use_cache and os.path.exists(train_cache_file) and os.path.exists(test_cache_file):
        debug_log("Loading tokenized and split datasets from cache.")
        return Dataset.load_from_disk(train_cache_file), Dataset.load_from_disk(test_cache_file)
    else:
        debug_log("Cache tokens not found. Generating the gene expression dataset.")
    
        expressions_cache_file = f'{cache_prefix}_expressions.npz'
        os.makedirs(os.path.dirname(expressions_cache_file), exist_ok=True)
    
        if use_cache and os.path.exists(expressions_cache_file):
            debug_log("Loading cached gene expressions from disk.")
            data = np.load(expressions_cache_file, allow_pickle=True)
            gene_expressions = data['expressions']
            cell_barcodes = data['barcodes']
            gene_names = data['gene_names']
        else:
            debug_log(f"Importing dataset from {input_file}")
            if input_file.endswith('.gz'):
                df = pd.read_csv(input_file, index_col=0, sep=r'[,\t]', engine='python', compression='gzip') # , dtype=np.float32
            else:
                df = pd.read_csv(input_file, index_col=0, sep=r'[,\t]', engine='python') # , dtype=np.float32

            cell_barcodes = df.index.tolist()
        
            debug_log(f"Loaded the expression matrix with {len(cell_barcodes)} cells and {len(gene_names)} genes.")

            if use_cache:
                debug_log("Saving the preprocessed gene expressions to an array.")
                gene_expressions = df.to_numpy(dtype=np.float32)
                cell_barcodes = np.array(df.index)
                gene_names = np.array(df.columns)
                # Uncomment to save cache
                # np.savez(expressions_cache_file, expressions=gene_expressions, barcodes=cell_barcodes, gene_names=gene_names)
                
            del df
            gc.collect()
    
        debug_log("Splitting the dataset into train/test/holdout (80/10/10).")
        
        full = Dataset.from_generator(
            lambda: generate_dataset_rows(cell_barcodes, gene_expressions, gene_names)
        )
        
        # 80/20 first
        train_temp = full.train_test_split(test_size=0.20, seed=42)
        
        # split the 20% evenly => 10% test, 10% holdout
        test_holdout = train_temp["test"].train_test_split(test_size=0.5, seed=42)
        
        dataset = {
            "train": train_temp["train"],
            "test": test_holdout["train"],
            "holdout": test_holdout["test"],
        }

    
        del gene_expressions, cell_barcodes, gene_names
        gc.collect()

        def send_test_or_train_to_tokenizer(dataset, test_or_train):
            """
            Tokenizes one split (train/test) of the dataset.
    
            Args:
                dataset (dict): Hugging Face split dictionary.
                test_or_train (str): Either `train` or `test`.
    
            Returns:
                datasets.Dataset: Tokenized split.
            """
            debug_log(f"Tokenizing the {test_or_train} dataset.")
            return dataset[test_or_train].map(
                lambda examples: preprocess(tokenizer, examples),
                batched=False,
                batch_size=config['batch_size'],
                num_proc=num_proc
            )

        tokenized_train_set = send_test_or_train_to_tokenizer(dataset, 'train')
        tokenized_test_set = send_test_or_train_to_tokenizer(dataset, 'test')
        debug_log("Finished tokenization.")

        # Clean up raw columns after tokenization
        tokenized_train_set = tokenized_train_set.remove_columns(["gene_names", "gene_expressions"])
        tokenized_test_set  = tokenized_test_set.remove_columns(["gene_names", "gene_expressions"])


        debug_log("Saving tokens to disk.")
        
        holdout_barcodes = [ex["barcode"] for ex in dataset["holdout"]]

        train_cache_file, test_cache_file = save_split_data(
            cache_prefix,
            use_cache,
            tokenized_train_set,
            tokenized_test_set,
            holdout_barcodes
        )

    
        debug_log(f"Datasets cached: {train_cache_file}, {test_cache_file}.")
        del dataset
        gc.collect()
        
        return tokenized_train_set, tokenized_test_set

@track_performance
def start_loop(debug=False, use_cache=True):
    """
    Initializes tokenizer and creates tokenized datasets.

    Args:
        debug (bool): Enable debug logging
        use_cache (bool): Use cached datasets if available

    Returns:
        None
    """
    with open(genes_file, "r") as gn:
        genes_list = [ln.strip() for ln in gn if ln.strip()]
    tokenizer = GeneTokenizer(genes=genes_list)
    
    train_dataset, test_dataset = create_and_cache_tokenized_dataset(
        debug=debug, 
        use_cache=use_cache, 
        tokenizer=tokenizer
    )

    debug_log("Data preprocessing run finished. \n")


if __name__ == "__main__":
    if len(sys.argv) - 1 != 5:
        print(f"Error: Expected 5 arguments, but got {len(sys.argv) - 1}.")
        print("Usage: python preprocess.py <input_file> <genes_file> <prefix> <config_file> <num_proc>")
        sys.exit(1)

    input_file = sys.argv[1]
    genes_file = sys.argv[2]
    prefix = sys.argv[3]
    config_file = sys.argv[4]
    num_proc = int(sys.argv[5])
    # num_workers = os.cpu_count()
    print(f"Available CPUs: {num_proc}")

    load_config(config_file)

    log_file = initialize_logging(prefix, context="tokenize")

    start_loop(debug=config['debug'], use_cache=config['use_cache'])
