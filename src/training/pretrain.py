#!/usr/bin/env python3
import os, sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)
import time
import json
import yaml
import pickle
import shutil
import logging
import gc
import psutil
import pandas as pd
import numpy as np
import functools
from datasets import Dataset
from collections import OrderedDict
import torch
from torch import nn, optim
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import pyarrow as pa
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from src.model import AtlasModelRankBased, ModelArgs
from src.preprocess import DataCollatorForGeneModeling
from src.preprocess import GeneExpressionDatasetRB
from src.preprocess import GeneTokenizer
from src import *


def preprocess(tokenizer, examples):
    """
    Tokenizes one example/cell at a time using the tokenizer module.

    Args:
        tokenizer (GeneTokenizer): Tokenizer for mapping gene names to token IDs
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

@track_performance
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
    assert tokenizer is not None, "Tokenizer must be provided"

    cache_prefix = f"{prefix}/debug" if not use_cache else f"{prefix}"
    train_cache_file = f"{cache_prefix}/train"
    test_cache_file = f"{cache_prefix}/test"

    with open(genes_file, 'r') as gn:
        gene_names = [line.strip() for line in gn]

    if use_cache and os.path.exists(train_cache_file) and os.path.exists(test_cache_file):
        debug_log("Loading tokenized and split datasets from cache.")
        train_dataset = Dataset.load_from_disk(train_cache_file)
        test_dataset = Dataset.load_from_disk(test_cache_file)
        return train_dataset, test_dataset, len(gene_names)
    else:
        debug_log("Cache files not found. Need to generate dataset.")
        raise RuntimeError("Dataset cache missing and no dataset generation logic implemented. Aborting.")


def validate_model(model, test_loader, criterion, writer, epoch, device):
    """
    Runs validation on the test set and logs loss and accuracy.

    Args:
        model (nn.Module): Model in eval mode during validation
        test_loader (DataLoader): Batches from the test (validation) dataset
        criterion (callable): Loss function applied to logits and labels
        writer (SummaryWriter): TensorBoard logger for metrics
        epoch (int): Epoch index for logging
        device (torch.device): Device to run validation on
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids, padding_mask, labels = batch['input_ids'].to(device), batch['padding_mask'].to(device), batch['labels'].to(device)
            outputs = model(input_ids, key_padding_mask=padding_mask)
            
            loss = criterion(outputs.view(-1, model.params.vocab_size), labels.view(-1))
            total_loss += loss.item()

            # Compute accuracy (assuming outputs are logits)
            #predicted = torch.argmax(outputs, dim=-1)  # Get highest probability class
            #correct += (predicted == labels).sum().item()
            #total += labels.numel()
            # Token-wise accuracy excluding ignored labels (must match criterion's ignore_index)
            ignore_index = -100
            predicted = torch.argmax(outputs, dim=-1)
            valid_mask = (labels != ignore_index)
            correct += ((predicted == labels) & valid_mask).sum().item()
            total   += valid_mask.sum().item()

    accuracy = correct / total if total > 0 else 0
    debug_log(f"Validation Loss: {total_loss/len(test_loader):.4f}, Accuracy: {accuracy:.4f}")

    debug_log(f'Validation Loss: {total_loss/len(test_loader)}, {epoch}')
    debug_log(f'Validation Accuracy: {accuracy}, {epoch}')
    writer.add_scalar('Validation Loss', total_loss/len(test_loader), epoch)
    writer.add_scalar('Validation Accuracy', accuracy, epoch)
    debug_log("Validation complete.")
    
    torch.cuda.empty_cache()
    
@track_performance
def train_model(model, train_loader, test_loader, optimizer, scheduler, criterion, writer, num_epochs):
    """
    Trains the model over N epochs, processing data by batch and performing validation after each epoch.

    Args:
        model (nn.Module): Model in eval mode during validation
        train_loader (DataLoader): Training data loader
        test_loader (DataLoader): Validation data loader
        optimizer (Optimizer): Optimizer instance
        scheduler (LRScheduler): Scheduler for learning rate
        criterion (Loss): Loss function
        writer (SummaryWriter): TensorBoard writer
        num_epochs (int): Number of training epochs
    """
    device = torch.device(
		"cuda"
		if torch.cuda.is_available() 
		else "mps" 
		if torch.backends.mps.is_available() 
		else "cpu")
    model.to(device)

    debug_log("Training started.")

    model.train()

    if device == 'cuda':
        scaler = amp.GradScaler()
        debug_log(f"CUDA device detected; running in mixed precision mode. CUDA memory checkpoint: \n {torch.cuda.memory_summary()}")
    elif device == 'mps':
        debug_log("MPS device detected; running in standard precision mode.")
    else:
        debug_log(f"Device: {device}.")

    for epoch in range(num_epochs):
        debug_log(f"Epoch {epoch+1}.")
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", miniters=500) # , mininterval=10.0

        for batch in progress_bar:
            input_ids, padding_mask, labels = batch['input_ids'].to(device), batch['padding_mask'].to(device), batch['labels'].to(device)
            #print("Max token ID in batch:", input_ids.max().item())

            model.zero_grad() # Zero gradients before the forward pass

            if device == 'cuda':
                with amp.autocast():
                    outputs = model(input_ids, key_padding_mask=padding_mask) # Forward pass: Get model predictions -> (batch_size, sequence_length, vocab_size)
                    loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1)) # Labels shape: (batch_size, sequence_length)
                scaler.scale(loss).backward()  # Backward pass: Compute gradient of the loss with respect to model parameters
                scaler.step(optimizer) # Update parameters
                scaler.update()
            elif device == 'mps':
                # MPS does not support mixed precision (with autocast and GradScaler).
                outputs = model(input_ids, key_padding_mask=padding_mask)
                loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                loss.backward()
                optimizer.step()
            else:
                outputs = model(input_ids, key_padding_mask=padding_mask)
                loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                loss.backward()       # Backward pass
                optimizer.step()      # Update parameters

            scheduler.step()          # Update learning rate
            total_loss += loss.item() # Accumulate loss for reporting

        avg_loss = total_loss / len(train_loader)

        debug_log(f"Epoch: {epoch+1}, Avg Loss: {avg_loss:.4f}")
        debug_log(f'Training Loss: {avg_loss}, {epoch}')
        writer.add_scalar('Training Loss', avg_loss, epoch)

        # Validation
        validate_model(model, test_loader, criterion, writer, epoch, device)

    writer.close()
    torch.cuda.empty_cache()

    save_path = output if output.endswith(".pt") else output + ".pt"
    if os.path.exists(save_path):
        debug_log(f"Warning: Overwriting existing model at {save_path}.")
    
    model_state_dict_with_prefix = OrderedDict({f"base_model.{k}" if not k.startswith("base_model.") else k: v for k, v in model.state_dict().items()}) # Otherwise, `model.state_dict()` is saved without the model's keys prefix
    torch.save(model_state_dict_with_prefix, save_path)
    debug_log(f"Model saved: {output}.")
    
    debug_log("Training complete.")

@track_performance
def start_loop(debug=False, use_cache=True):
    """
    Initializes the training loop: sets up dataset, model, optimizer, scheduler, and begins training.

    Args:
        debug (bool): Enable debug logs
        use_cache (bool): Whether to use cached tokenized data
    """
    debug_log("Preparing dataset.")
    with open(genes_file, "r") as gn:
        genes_list = [ln.strip() for ln in gn if ln.strip()]
    tokenizer = GeneTokenizer(genes=genes_list)
    
    train_dataset, test_dataset, genes = create_and_cache_tokenized_dataset(debug=debug, use_cache=use_cache, tokenizer=tokenizer)
    
    # DataLoader setup
    collator_function = DataCollatorForGeneModeling(tokenizer=tokenizer)
    
    train_loader = DataLoader(
                        train_dataset['tokenized_genes'], 
                        batch_size=config['batch_size'], 
                        shuffle=True, 
                        collate_fn=collator_function, 
                        num_workers=num_proc, 
                        pin_memory=True
    )
    
    test_loader = DataLoader(
                        test_dataset['tokenized_genes'], 
                        batch_size=config['batch_size'], 
                        shuffle=False, 
                        collate_fn=collator_function, 
                        num_workers=num_proc, 
                        pin_memory=True
    )
    
    debug_log("DataLoaders setup complete.")

    # Model, optimizer, and loss function setup
    debug_log("Model, optimizer, and loss function setup complete.")

    # print("Vocab Size:", vocab_size)
    # print("Embed Dim:", embed_dim)

    torch.cuda.empty_cache()  
    args = ModelArgs(vocab_size=genes + 3)
    model = AtlasModelRankBased(args)
	
    optimizer = optim.AdamW(
                    model.parameters(), 
                    lr=float(config['pretrain_lr']), 
                    betas=(config['beta1'], config['beta2']), 
                    weight_decay=config['weight_decay']
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100)  # Ignore padding for loss calculation

    # Learning rate scheduler setup
    num_epochs = config['num_epochs']
    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup

    scheduler = get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_steps,
                    num_cycles=config['num_cycles'],  # Half cycle for cosine decay
                    last_epoch=-1
    )

    # Initialize TensorBoard writer
    os.makedirs(output, exist_ok=True)
    writer = SummaryWriter(log_dir=output) # TensorBoard defaults to runs/ and time.time()

    # Hyperparameter dictionary for TensorBoard
    keys = ['embed_dim', 'num_layers', 'num_heads', 'vocab_size', 'norm_eps', 'max_seq_len', 'dropout', 'forward_expansion']
    hparam_dict = {key: config[key] for key in keys}
    hparam_dict['warmup_steps'] = warmup_steps
    # Write the hyperparameters to TensorBoard
    writer.add_hparams(hparam_dict, {})
    
    debug_log(f"Hyperparameters: {hparam_dict}")
    
    train_model(model, train_loader, test_loader, optimizer, scheduler, criterion, writer, num_epochs)
    
    del model, optimizer, scheduler, criterion, train_loader, test_loader
    torch.cuda.empty_cache()


if __name__ == "__main__":
    if len(sys.argv) - 1 != 6:
        print(f"Error: Expected 6 arguments, but got {len(sys.argv) - 1}.")
        print("Usage: python preprocess.py <input_file> <genes_file> <prefix> <config_file> <output> <num_proc>")
        sys.exit(1)

    input_file = sys.argv[1]
    genes_file = sys.argv[2]
    prefix = sys.argv[3]
    config_file = sys.argv[4]
    output = sys.argv[5]
    num_proc = int(sys.argv[6])
    # num_workers = os.cpu_count()
    print(f"Available CPUs: {num_proc}")

    load_config(config_file)

    log_file = initialize_logging(prefix, context="pretrain")

    start_loop(debug=config['debug'], use_cache=config['use_cache'])
