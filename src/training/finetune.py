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
import pandas as pd
import numpy as np
import functools
from datasets import Dataset
from collections import Counter
from collections import defaultdict
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.cuda.amp as amp
from torch.optim.lr_scheduler import StepLR
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from src.model import AtlasModelRankBased, ModelArgs
from src.preprocess import GeneExpressionDatasetRB
from src.preprocess import GeneTokenizer
# from data_utils import DataCollatorForClassification
from src import *


class ClassificationModel(nn.Module):
    """
    Wrapper model that adds a classification head on top of the transformer encoder

        - Processes tokenized gene expression input through pretrained AtlasModelRankBased transformer
        - Applies 1D average pooling across sequence length
        - Outputs class logits via a linear classifier

    Args:
        base_model (nn.Module): The pretrained AtlasModelRankBased model
        num_classes (int): Number of output classes for classification
    """
    def __init__(self, base_model, num_classes):
        super(ClassificationModel, self).__init__()
        self.base_model = base_model
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(base_model.params.embed_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        x = self.base_model.embeddings(input_ids)
        if isinstance(x, (tuple, list)):
            x = x[0]  # take the hidden states

        for encoder in self.base_model.encoders:
            x = encoder(x, attention_mask)
            if isinstance(x, (tuple, list)):
                x = x[0]  # keep only the tensor

        pooled = self.pooling(x.transpose(1, 2)).squeeze(-1)
        logits = self.classifier(pooled)
        return logits

class DataCollatorForClassification:
    """
    Data collator for classification tasks using tokenized gene expression data.

    Pads sequences to a fixed length and creates attention masks and label tensors.

    Args:
        tokenizer (GeneTokenizer): Tokenizer for gene tokens
        max_seq_length (int): Maximum sequence length for padding/truncation
        label2id (dict): Maps string labels to numeric class indices
        id2label (dict): Inverse mapping of label2id
    """
    def __init__(self, tokenizer, max_seq_length=None, label2id=None, id2label=None):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id
        self.label2id = label2id if label2id is not None else {}
        self.id2label = id2label if id2label is not None else {}
        # self.unknown_label_id = self.label2id.get("<UNK>", -1)  # Default to -1 if unknown label is not defined

    def __call__(self, batch):
        max_length = self.max_seq_length
        batch_input_ids = []
        batch_attention_masks = []
        batch_labels = []
    
        for item in batch:
            input_ids = item['tokenized_genes'][:max_length]
            label = item['cell_label']
            label_id = self.label2id[label]
    
            padded_input_ids = self.pad_tokens(input_ids, max_length)
            attention_mask = [i == self.pad_token_id for i in padded_input_ids]
    
            batch_input_ids.append(padded_input_ids)
            batch_attention_masks.append(attention_mask)
            batch_labels.append(label_id)
    
        # Convert lists to tensors
        batch_input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
        batch_attention_masks = torch.tensor(batch_attention_masks, dtype=torch.bool)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)
    
        return {
            'input_ids': batch_input_ids,
            'attention_mask': batch_attention_masks,
            'labels': batch_labels
        }

    def pad_tokens(self, input_ids, max_length):
        padding_length = max_length - len(input_ids)
        return input_ids + [self.pad_token_id] * padding_length if padding_length > 0 else input_ids


def preprocess(tokenizer, examples):
    """
    Tokenizes one example (e.g., cell) at a time using the tokenizer module.

    Args:
        tokenizer (GeneTokenizer): Tokenizer for mapping gene names to token IDs
        examples (dict): Dictionary with `gene_names` and `gene_expressions`

    Returns:
        dict: {'tokenized_genes': <list of token IDs>}
    """
    # tokenized_output = tokenizer(examples['gene_names'], examples['gene_expressions'])
    return {
        'tokenized_genes': tokenizer(
            examples['gene_names'],
            torch.tensor(examples['gene_expressions'], dtype=torch.float32).cpu().numpy()
        )
    }

def create_and_cache_tokenized_dataset(debug=False, use_cache=True, tokenizer=None):
    """
    Loads raw gene expression data, runs preprocessing, and caches tokenized datasets.

    Steps:
      - Load matrix from CSV or .npz cache
      - Stream rows into Hugging Face dataset
      - Split into train/test sets
      - Tokenize gene sequences per cell
      - Cache barcodes and datasets to disk

    Args:
        debug (bool): Enables debug logging
        use_cache (bool): If True, loads from/saves cached expression and token datasets to disk
        tokenizer (GeneTokenizer): Required tokenizer for gene to token conversion

    Returns:
        tuple: (tokenized_train_set, tokenized_test_set), both Hugging Face datasets
    """
    assert tokenizer is not None, "Tokenizer must be provided"
    
    cache_prefix = 'debug' if not use_cache else prefix
    train_cache_file = f"{cache_prefix}/train"
    test_cache_file = f"{cache_prefix}/test"

    with open(genes_file, 'r') as gn:
        gene_names = [line.strip() for line in gn]
    
    if os.path.exists(train_cache_file) and os.path.exists(test_cache_file):
        debug_log("Loading the tokenized and split datasets from cache.")
        train_dataset = Dataset.load_from_disk(train_cache_file)
        test_dataset = Dataset.load_from_disk(test_cache_file)
        return train_dataset, test_dataset, len(gene_names)
    else:
        csv_file = 'debug_data.csv' if not use_cache else input_file
        debug_log(f"Tokenizing and splitting dataset from: {csv_file}")
        
        if csv_file.endswith('.gz'):
            df = pd.read_csv(csv_file, sep=r'[,\t]', engine='python', compression='gzip')
        else:
            df = pd.read_csv(csv_file, sep=r'[,\t]', engine='python')
        
        gene_expressions = [np.array(eval(exp), dtype=np.float32) for exp in df['gene_expressions']]
    
        dataset = Dataset.from_dict({
            'gene_names': [gene_names for _ in range(len(df))],
            'gene_expressions': gene_expressions
        }).train_test_split(test_size=0.2, seed=42)
    
        tokenized_train_set = dataset['train'].map(lambda examples: preprocess(tokenizer, examples))
        tokenized_test_set = dataset['test'].map(lambda examples: preprocess(tokenizer, examples))
        
        # Cache cleaned split datasets
        tokenized_train_set = tokenized_train_set.remove_columns(['gene_names', 'gene_expressions'])
        tokenized_test_set = tokenized_test_set.remove_columns(['gene_names', 'gene_expressions'])
    
        tokenized_train_set.save_to_disk(train_cache_file)
        tokenized_test_set.save_to_disk(test_cache_file)
        debug_log(f"Datasets cached: {train_cache_file}, {test_cache_file}")
    
        return tokenized_train_set, tokenized_test_set, len(gene_names)

def generate_confusion_matrix(true_labels, pred_labels, id2label):
    """
    Generate and save a confusion matrix.

    Args:
        true_labels (list[int]): Ground-truth label IDs
        pred_labels (list[int]): Predicted label IDs
        id2label (dict[int, str]): Maps class ID to label for axis tick labels
    """
    present = sorted(set(true_labels) | set(pred_labels))
    cm = confusion_matrix(true_labels, pred_labels, labels=present)
    labels = [id2label.get(i, f"Class {i}") for i in present]

    with np.errstate(invalid='ignore', divide='ignore'):
        col = cm.astype(np.float32) / np.where(cm.sum(axis=0, keepdims=True) == 0, 1, cm.sum(axis=0, keepdims=True))  # precision
        row = cm.astype(np.float32) / np.where(cm.sum(axis=1, keepdims=True) == 0, 1, cm.sum(axis=1, keepdims=True))  # recall

    # Minimal saves
    np.savetxt(f"{output}_conf_matrix.txt", cm, fmt="%d", delimiter="\t")
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(f"{output}_confusion_matrix.csv")
    for name, mat in (("precision", col), ("recall", row)):
        np.savetxt(f"{output}_conf_matrix_{name}.txt", mat, fmt="%.6f", delimiter="\t")
        pd.DataFrame(mat, index=labels, columns=labels).to_csv(f"{output}_confusion_matrix_{name}.csv")

    # One plotting loop
    n = len(labels)
    fig_size = (max(8, 0.55*n), max(6, 0.45*n))
    annotate = (n <= 40)

    for name, mat in (("precision", col), ("recall", row)):
        plt.figure(figsize=fig_size, dpi=200)
        ax = sns.heatmap(
            mat, annot=annotate, fmt=".2f" if annotate else "",
            cmap="Blues", xticklabels=labels, yticklabels=labels,
            cbar_kws={"shrink": 0.8}
        )
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("True Labels")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
        plt.tight_layout()
        plt.savefig(f"{output}_conf_matrix_{name}.png", bbox_inches="tight")
        plt.savefig(f"{output}_conf_matrix_{name}.pdf", bbox_inches="tight")
        plt.close()

    print("Confusion matrices saved (raw, precision, recall).")

def start_loop(debug=False, use_cache=True):
    """
    Initializes the training loop: 
        - Loads tokenized data
        - Initializes model, loss, optimizer, and DataLoaders
        - Trains for N epochs
        - Evaluates on validation set and saves predictions, reports, and confusion matrices

    Args:
        debug (bool): Enables verbose logging
        use_cache (bool): Enables reuse of cached tokenized datasets
    """
    with open(genes_file, "r") as gn:
        genes_list = [ln.strip() for ln in gn if ln.strip()]
    tokenizer = GeneTokenizer(genes=genes_list)

    train_dataset, test_dataset, genes = create_and_cache_tokenized_dataset(debug=debug, use_cache=use_cache, tokenizer=tokenizer)

    args = ModelArgs(vocab_size=genes + 3)
    full_model = AtlasModelRankBased(args) # , weights_only=True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state = torch.load(model_file, weights_only=True, map_location=device)
    
    # unwrap {"state_dict": ...} if needed
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    
    # strip "base_model." prefix and drop classifier keys
    if any(k.startswith("base_model.") for k in state.keys()):
        state = {k.replace("base_model.", "", 1): v for k, v in state.items()}
    state = {k: v for k, v in state.items() if not k.startswith("classifier.")}
    
    # Filter to matching shapes, then load non strict
    msd = full_model.state_dict()
    state = {k: v for k, v in state.items() if k in msd and msd[k].shape == v.shape}

    full_model.load_state_dict(state, strict=False)
    #full_model.load_state_dict(torch.load(model_file)) # , map_location="cpu"

    def create_labels():
        labels = pd.read_csv(labels_file, sep=r'[,\t]', engine='python')
        unique_labels = labels['ID'].unique()

        unique_labels = sorted(unique_labels)
        
        # Label mappings: label2id and id2label
        label2id = {label: idx for idx, label in enumerate(unique_labels)}
        id2label = {idx: label for idx, label in enumerate(unique_labels)}

        debug_log(f"Inference Label Mapping: {label2id}")
        debug_log(f"Inverse Label Mapping: {id2label}")

        return label2id, id2label, len(unique_labels)

    label2id, id2label, len_labels = create_labels()
    
    collator_function = DataCollatorForClassification(
        tokenizer=tokenizer, 
        max_seq_length=config['max_seq_len'], 
        label2id=label2id, 
        id2label=id2label)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        collate_fn=collator_function, 
        num_workers=num_proc, 
        pin_memory=True)
    valid_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        collate_fn=collator_function, 
        num_workers=num_proc, 
        pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    debug_log("Training started \n")
    classifier_model = ClassificationModel(full_model, num_classes=len_labels)
    classifier_model.to(device)

    optimizer = optim.AdamW(classifier_model.parameters(), lr=3e-5, weight_decay=0.01)

    # ---------------------
    # Compute class weights
    all_labels_list = list(label2id.values())  # pass labels as numerical IDs
    class_weights = compute_class_weight("balanced", classes=np.array(all_labels_list), y=np.array(all_labels_list))
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    # Apply class weights in training
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # ---------------------
    # criterion = nn.CrossEntropyLoss(ignore_index=-100)  # Ignore padding for loss calculation

    # Epoch level losses
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    num_epochs = config['num_epochs']
    preds_dict = defaultdict(list)

    for epoch in range(num_epochs):
        classifier_model.train()
        train_correct = 0
        train_total = 0
        train_loss = 0
        
        train_correct = 0
        train_total = 0
        train_loss = 0  

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = classifier_model(input_ids, attention_mask)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            train_loss += loss.item()  # Accumulate loss for the epoch

            
            train_accuracy = 100 * train_correct / train_total
    
            # Batch-level log metrics
            writer.add_scalar('Train/Loss', loss.item(), epoch * len(train_loader) + batch['input_ids'].shape[0])
            writer.add_scalar('Train/Accuracy', train_accuracy, epoch * len(train_loader) + batch['input_ids'].shape[0])

        # Epoch level log metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        writer.add_scalar('Train/Epoch/Loss', avg_train_loss, epoch)
        writer.add_scalar('Train/Epoch/Accuracy', train_accuracy, epoch)

        # Validation phase
        classifier_model.eval()
        valid_correct = 0
        valid_total = 0
        valid_loss = 0
        all_labels = []
        all_predictions = []
        attention_scores = []
        embeddings_list = []
        
        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = classifier_model(input_ids, attention_mask)
                """
                try:
                    # Extraction of attention scores is failing
                    attention = classifier_model.base_model.encoders[-1].attention(input_ids, attention_mask)[1]
                    attention_scores.append(attention.cpu().numpy())
                except AttributeError as e:
                    print(f"Attention extraction failed: {e}")
                        
                embeddings = classifier_model.base_model.embeddings(input_ids)
                attention_output, attention_weights = classifier_model.base_model.encoders[-1].attention(
    embeddings, embeddings, embeddings, attn_mask=attention_mask
)
                embeddings_list.append(embeddings.cpu().numpy())
                """
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                
                valid_total += labels.size(0)
                valid_correct += (predicted == labels).sum().item()
                valid_loss += loss.item()  # Accumulate validation loss

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        avg_valid_loss = valid_loss / len(valid_loader)    
        valid_accuracy = 100 * valid_correct / valid_total
        val_losses.append(avg_valid_loss)
        val_accuracies.append(valid_accuracy)

        # Validation metrics
        writer.add_scalar('Valid/Loss', avg_valid_loss, epoch)
        writer.add_scalar('Valid/Accuracy', valid_accuracy, epoch)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Valid Accuracy: {valid_accuracy:.2f}%")

        # Compute and save classification report
        unique_labels_present = sorted(set(all_labels) | set(all_predictions))
        class_report = classification_report(
            # all_labels, all_predictions, target_names=[id2label[i] for i in range(len(id2label))]
            # DEBUG: "UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior."
            all_labels, 
            all_predictions, 
            labels=list(id2label.keys()),
            target_names=[id2label[i] for i in unique_labels_present], 
            zero_division=0
        )
        debug_log(f"\nClassification Report:\n {class_report}")

        report_file = f"{output}_classification_report.txt"
        with open(report_file, "a") as f:
            f.write(f"Epoch {epoch+1} Classification Report:\n")
            f.write(class_report + "\n")

        # ==========================================================================
        # CM Plot
        generate_confusion_matrix(all_labels, all_predictions, id2label)
        # ==========================================================================


    debug_log("Checking model before saving:")
    debug_log(f"Pos_embeddings: {'pos_embeddings' in dir(classifier_model.base_model.embeddings)}")
    debug_log(f"Classifier weight: {'classifier.weight' in classifier_model.state_dict()}")

    # Apply base_model prefix
    model_state_dict_with_prefix = OrderedDict({
        (f"base_model.{k}" if not k.startswith("base_model.") and "classifier" not in k else k): v
        for k, v in classifier_model.state_dict().items() # Otherwise, `model.state_dict()` is saved without the model's keys prefix
    })
    """
    new_state_dict = {
        ("base_model." + k if not k.startswith("base_model.") and "classifier" not in k else k): v
        for k, v in state_dict.items()
    }
    classifier.load_state_dict(new_state_dict, strict=False)
    """
    # Make sure classifier weights exist
    if "classifier.weight" not in model_state_dict_with_prefix or "classifier.bias" not in model_state_dict_with_prefix:
        debug_log("classifier.weight is missing! Initializing...")
        torch.nn.init.xavier_uniform_(classifier_model.classifier.weight)
        torch.nn.init.zeros_(classifier_model.classifier.bias)
        model_state_dict_with_prefix["classifier.weight"] = classifier_model.classifier.weight
        model_state_dict_with_prefix["classifier.bias"] = classifier_model.classifier.bias
    
    torch.save(model_state_dict_with_prefix, f"{output}.pt")
    
    debug_log(f"Model saved as {output}.pt with classifier weights and base_model prefix included.")

    # Save model and predictions
    # torch.save(classifier_model.state_dict(), f"{output}.pt")
    # torch.save(classifier_model.base_model.state_dict(), f'{output}.pt')
    preds_df = pd.DataFrame({"labels": all_labels, "predictions": all_predictions})
    preds_df.to_parquet(f"{output}_preds_dict.parquet", index=False)

    # Save losses and accuracies
    np.savetxt(f"{output}_train_losses.txt", np.array(train_losses))
    np.savetxt(f"{output}_val_losses.txt", np.array(val_losses))
    np.savetxt(f"{output}_train_accuracies.txt", np.array(train_accuracies))
    np.savetxt(f"{output}_val_accuracies.txt", np.array(val_accuracies))
    
    
if __name__ == "__main__":
    if len(sys.argv) - 1 != 8:
        print(f"Error: Expected 8 arguments, but got {len(sys.argv) - 1}.")
        print("Usage: python preprocess.py <input_file> <genes_file> <prefix> <model_file> <labels_file> <config_file> <output> <num_proc>")
        sys.exit(1)

    input_file = sys.argv[1]
    genes_file = sys.argv[2]
    prefix = sys.argv[3]
    model_file = sys.argv[4]
    labels_file = sys.argv[5]
    config_file = sys.argv[6]
    output = sys.argv[7]
    num_proc = int(sys.argv[8])

    load_config(config_file)

    log_file = initialize_logging(prefix, context="finetune")

    # TensorBoard writer
    writer = SummaryWriter(log_dir=f'{output}')

    start_loop(debug=config['debug'], use_cache=config['use_cache'])
