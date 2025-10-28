#!/usr/bin/env python3
import os, sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)
import csv
import time
import pandas as pd
import numpy as np
import yaml
import logging
import gc
import psutil
import functools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import Dataset
from collections import defaultdict
from collections import Counter
from tqdm import tqdm
from datasets.utils.logging import disable_progress_bar
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import concatenate_datasets
from src.preprocess import GeneTokenizer
from src.model import AtlasModelRankBased, ModelArgs
# from gene_expression_datasets import GeneExpressionDatasetRB
from src import *


class ClassificationModel(nn.Module):
    """
    Classification wrapper on top of the AtlasModelRankBased model.

    Applies the transformer encoder to the input sequence, performs adaptive average pooling 
    across sequence length, then outputs class logits. Softmax probabilities are returned with 
    temperature scaling.

    Args:
        base_model (nn.Module): Pretrained transformer backbone
        num_classes (int): Number of classification categories
    """
    def __init__(self, base_model, num_classes):
        super(ClassificationModel, self).__init__()
        self.base_model = base_model
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(base_model.params.embed_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)  # Add softmax layer to get probabilities

    def forward(self, input_ids, attention_mask):
        # Pass input through the base model, but stop before the LM head
        embeddings = self.base_model.embeddings(input_ids)
        for encoder in self.base_model.encoders:
            embeddings, _ = encoder(embeddings, attention_mask, return_attn=False) # Since AtlasEncoderRB.forward returns (x, None) when return_attn=False, then unpack here
            #embeddings = encoder(embeddings, attention_mask)

        # Pooling across the sequence length dimension to get a single vector per sample
        pooled_output = self.pooling(embeddings.transpose(1, 2)).squeeze(-1)

        # Final classification layer
        logits = self.classifier(pooled_output)
        # print("Sample classifier logits before softmax:", logits[0].detach().cpu().numpy())

        # probs = self.softmax(logits)  # Get probabilities
        def temperature_softmax(logits, T=2.0):
            return torch.nn.functional.softmax(logits / T, dim=-1)

        probs = temperature_softmax(logits, T=2.0)
        # print("Sample softmax outputs:", probs[0].detach().cpu().numpy())

        # print(logits)
        # print("-----")
        return logits, probs

class DataCollatorForClassification:
    """
    Collator for gene classification.
    
    Pads token sequences and prepares attention masks and labels for model input.

    Args:
        tokenizer (GeneTokenizer): Tokenizer to identify pad tokens
        max_seq_length (int): Maximum token sequence length
        label2id (dict, optional): Maps label strings to integer IDs
        id2label (dict, optional): Maps integer IDs back to label strings
    """
    def __init__(self, tokenizer, max_seq_length, label2id=None, id2label=None):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id
        self.label2id = label2id if label2id is not None else {}
        self.id2label = id2label if id2label is not None else {}
        # self.unknown_label_id = self.label2id.get("<UNK>", -1)  # Default to -1 if unknown label is not defined

    def __call__(self, batch):
        batch_input_ids = []
        batch_attention_masks = []
        batch_labels = []
        batch_cell_indices = []

        for item in batch:
            input_ids = item['tokenized_genes'][:self.max_seq_length]
            cell_index = item['cell_index']
            label = item['cell_label']
            label_id = -1

            # label_id = self.label2id[label]

            # Match the length of input_ids with that of max_length
            padded_input_ids = self.pad_tokens(input_ids, self.max_seq_length)
            attention_mask = [i == self.pad_token_id for i in padded_input_ids]

            batch_input_ids.append(padded_input_ids)
            batch_attention_masks.append(attention_mask)
            batch_cell_indices.append(cell_index)
            batch_labels.append(label_id)

        # Convert lists to tensors
        batch_input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
        batch_attention_masks = torch.tensor(batch_attention_masks, dtype=torch.bool)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)
        batch_cell_indices = torch.tensor(batch_cell_indices, dtype=torch.int)

        return {
            'cell_indices': batch_cell_indices,
            'input_ids': batch_input_ids,
            'attention_mask': batch_attention_masks,
            'labels': batch_labels
        }

    def pad_tokens(self, input_ids, max_length):
        padding_length = max_length - len(input_ids)
        return input_ids + [self.pad_token_id] * padding_length if padding_length > 0 else input_ids[:max_length]


def create_labels():
    """
    Build label maps from the global `classes` (barcode indexed).
    Expects `classes.index` = barcodes and a column 'ID' with class names.
    """
    if 'ID' not in classes.columns:
        raise ValueError("Metadata file must contain an 'ID' column with class labels.")

    # Get unique labels from metadata
    unique_classes = sorted(pd.Series(classes['ID'].dropna().astype(str).unique()).tolist())

    label2id = {label: idx for idx, label in enumerate(unique_classes)}
    id2label = {v: k for k, v in label2id.items()}

    debug_log(f"Inference Label Mapping: {label2id}")
    debug_log(f"Inverse Label Mapping: {id2label}")
    #debug_log(f"n_classes from metadata: {len(unique_classes)}  --> {unique_classes}")

    return label2id, id2label, len(unique_classes)


def load_true_labels(true_labels_file):
    """
    Loads a barcode to ground-truth label mapping from a file (CSV/TSV).

    Args:
        true_labels_file (str): Path to file the containing barcodes and ground-truth labels

    Returns:
        dict | None: {barcode: label} mapping, or None if the file is missing
    """
    if not os.path.exists(true_labels_file):
        print("No true labels file found. Skipping confusion matrix generation.")
        return None

    df = pd.read_csv(true_labels_file, sep=r'[,\t]', index_col=0, engine='python')
    true_labels = df.to_dict()['ID']
    
    print(f"Loaded {len(true_labels)} true labels.")
    return true_labels

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

    # Minimal saves (keep your originals + explicit precision/recall)
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

    
    print("Confusion matrix saved.")


def compute_and_save_metrics(output_prefix, val_loss_total, val_correct, val_total,
                             all_labels, all_predictions, id2label, attention_scores, embeddings_list):
    """
    Saves inference results: 
        metrics, attention scores, embeddings, and classification report
    """
    avg_val_loss = val_loss_total / val_total if val_total > 0 else np.nan
    val_accuracy = val_correct / val_total if val_total > 0 else np.nan

    debug_log(f"Final Metrics -> Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    np.savetxt(f'{output_prefix}_val_losses.txt', np.array([avg_val_loss]))
    np.savetxt(f'{output_prefix}_val_accuracies.txt', np.array([val_accuracy]))

    print(f"Metrics saved: {output_prefix}_val_losses.txt, {output_prefix}_val_accuracies.txt")

    if attention_scores:
        attention_scores = np.vstack(attention_scores).astype(np.float16)
        np.savez_compressed(f'{output_prefix}_attention_scores.npz', attention_scores)
    else:
        debug_log("No attention scores collected. Skipping save.")

    if embeddings_list:
        embeddings_list = np.vstack(embeddings_list).astype(np.float16)
        np.savez_compressed(f'{output_prefix}_embeddings.npz', embeddings_list)
    else:
        print("No embeddings collected. Skipping save.")

    present_classes = sorted(set(all_labels))
    target_names_present = [id2label[i] for i in present_classes]
    # Compute and save classification report
    if any(lbl != -1 for lbl in all_labels):
        class_report = classification_report(
            all_labels, all_predictions, 
            labels=present_classes,
            target_names=target_names_present
            # labels=list(range(len(id2label))),
            # target_names=[id2label[i] for i in range(len(id2label))]
        )
        with open(f'{output_prefix}_classification_report.txt', 'w') as f:
            f.write(class_report + '\n')
    else:
        debug_log("No true labels available. Skipping classification report.")

    debug_log(f"Inference outputs saved at {output_prefix}")

def save_inference_outputs(
    output_prefix,
    all_labels,
    all_predictions,
    id2label,
    attention_scores=None,
    embeddings_list=None,
    train_losses=None,
    val_losses=None,
    train_accuracies=None,
    val_accuracies=None,
    train_loss_total=None,
    train_correct=None,
    train_total=None,
    val_loss_total=None,
    val_correct=None,
    val_total=None
):
    """
    Saves inference outputs: 
        metrics, attention scores, embeddings, and classification report
    
    Supports both full history lists and summary totals
    """

    # Save per epoch history if lists exist
    if train_losses and val_losses and train_accuracies and val_accuracies:
        history_file = f'{output_prefix}_metrics_history.csv'
        with open(history_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Train Accuracy', 'Val Accuracy'])
            for i in range(len(train_losses)):
                writer.writerow([
                    i + 1,
                    train_losses[i],
                    val_losses[i],
                    train_accuracies[i],
                    val_accuracies[i]
                ])
        debug_log(f"Saved per-epoch metrics to {history_file}")
    else:
        debug_log("Per-epoch metric history not available. Skipping history save.")
    
    # Compute and save final summary
    if train_loss_total is not None and train_total:
        avg_train_loss = train_loss_total / train_total
        train_accuracy = train_correct / train_total
    else:
        avg_train_loss = train_losses[-1] if train_losses else None
        train_accuracy = train_accuracies[-1] if train_accuracies else None
    
    if val_loss_total is not None and val_total:
        avg_val_loss = val_loss_total / val_total
        val_accuracy = val_correct / val_total
    else:
        avg_val_loss = val_losses[-1] if val_losses else None
        val_accuracy = val_accuracies[-1] if val_accuracies else None
    
    summary_file = f'{output_prefix}_metrics_summary.csv'
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Train', 'Validation'])
        writer.writerow(['Loss', avg_train_loss, avg_val_loss])
        writer.writerow(['Accuracy', train_accuracy, val_accuracy])
    debug_log(f"Saved final summary to {summary_file}")

    # Attention scores
    if attention_scores:
        attention_scores = np.vstack(attention_scores).astype(np.float16)
        np.savez_compressed(f'{output_prefix}_attention_scores.npz', attention_scores)
    else:
        debug_log("No attention scores collected. Skipping save.")

    # Embeddings
    if embeddings_list:
        embeddings_list = np.vstack(embeddings_list).astype(np.float16)
        np.savez_compressed(f'{output_prefix}_embeddings.npz', embeddings_list)
    else:
        debug_log("No embeddings collected. Skipping save.")

    # Classification report
    if any(lbl != -1 for lbl in all_labels):
        present_classes = sorted(set(all_labels))
        target_names_present = [id2label[i] for i in present_classes]
        class_report = classification_report(
            all_labels,
            all_predictions,
            labels=present_classes,
            target_names=target_names_present
        )
        with open(f'{output_prefix}_classification_report.txt', 'w') as f:
            f.write(class_report + '\n')
    else:
        debug_log("No valid labels. Skipping classification report.")

    debug_log(f"Inference outputs saved at {output_prefix}")
    
def preprocess(tokenizer, examples):
    """
    Tokenizes one example/cell at a time using the tokenizer module

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

def dataframe_to_hf_dataset(df, chunk_size=10000):
    """
    Converts gene expression matrix into a Hugging Face Dataset
    
    Pads missing genes, aligns column order, and slices them into chunks

    Args:
        df (pd.DataFrame): Expression matrix with cells as rows
        chunk_size (int): Max number of cells per dataset chunk

    Returns:
        tuple: (Hugging Face Dataset, number of genes)
    """
    with open(genes_file, 'r') as gn:
        gene_library = [line.strip() for line in gn]

    gene_names = list(df.columns)
    missing_genes = [gene for gene in gene_library if gene not in gene_names]
    debug_log(f"Input gene count: {len(gene_names)}")
    debug_log(f"Vocabulary size: {len(gene_library)}")
    debug_log(f"Missing genes: {len(missing_genes)}") # {len(missing_genes)} - {missing_genes}

    # To avoid "high fragmentation" error, join all columns from missing genes at once
    # and concatenate instead of inserting one column at a time
    missing_gene_df = pd.DataFrame(0.0, index=df.index, columns=missing_genes)
    df = pd.concat([df, missing_gene_df], axis=1)

    df = df[gene_library] # Reorder the columns to match gene_names.txt

    assert df.shape[1] == len(gene_library)
    debug_log(f"DataFrame shape after filtering and padding: {df.shape}")

    debug_log(f"First 10 gene names from DataFrame: {gene_names[:10]}\n")

    num_chunks = (len(df) // chunk_size) + 1

    datasets = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, len(df))

        max_value = np.iinfo(np.int32).max - 1  # 2147483646
        gene_expressions = [np.clip(np.array(row, dtype=np.float32), a_min=None, a_max=max_value).astype(np.float32) for _, row in df.iloc[start:end].iterrows()]

        # Get barcodes (cell IDs) from DataFrame index
        barcodes = df.index.tolist()
        cell_labels = [classes.loc[barcode, 'ID'] if barcode in classes.index else 'UNKNOWN'
                        for barcode in df.index[start:end]]

        # cell_labels = [classes.loc[barcode, 'ID'] if barcode in classes.index else 'UNKNOWN' for barcode in barcodes]

        cell_indices = list(range(start, end))
        chunk_dataset = Dataset.from_dict({
            'gene_names': [gene_library for _ in range(end - start)],
            'gene_expressions': gene_expressions,
            'cell_label': cell_labels,
            'cell_index': cell_indices  # include cell_index
        })

        datasets.append(chunk_dataset)

    # Concatenate the smaller datasets
    final_dataset = concatenate_datasets(datasets)
    print(f"Number of chunks: {num_chunks}")
    for i, ds in enumerate(datasets):
        print(f"Chunk {i + 1}/{num_chunks} - Number of rows: {len(ds)}")

    return final_dataset, len(gene_library)

def prepare_inference_components_for_inference(df, true_labels_file=None):
    """
    Prepares components required for inference
    
    Tokenizes and converts the DataFrame into a Hugging Face Dataset and creates label mappings
    
    Loads the model and its state dictionary
    
    Initializes the DataCollator and DataLoader

    Args:
        df (pd.DataFrame): Gene expression matrix
        true_labels_file (str, optional): Optional file containing true labels

    Returns:
        tuple: (dataset, train_loader, classifier, device, label2id, id2label, true_labels_dict)
    """
    with open(genes_file, "r") as gn:
        genes_list = [ln.strip() for ln in gn if ln.strip()]
    tokenizer = GeneTokenizer(genes=genes_list)

    dataset, genes = dataframe_to_hf_dataset(df)

    dataset = dataset.map(lambda examples: preprocess(tokenizer, examples))
    dataset = dataset.remove_columns(['gene_names', 'gene_expressions'])

    label2id, id2label, len_labels = create_labels()

    #barcodes_in_data = set(df.index)
    #barcodes_in_labels = set(label2id.keys())
    #overlap = barcodes_in_data & barcodes_in_labels

    barcodes_in_data   = set(df.index)
    barcodes_in_labels = set(classes.index)  # classes must be barcode-indexed
    overlap = barcodes_in_data & barcodes_in_labels
    debug_log(f"[DEBUG] {len(overlap)} / {len(barcodes_in_data)} barcodes have matching labels")
    print(f"[DEBUG] {len(overlap)} / {len(barcodes_in_data)} barcodes have matching labels")

    args = ModelArgs(vocab_size=genes + 3)
    base_model = AtlasModelRankBased(args)

    classifier = ClassificationModel(base_model, num_classes=len_labels)
    debug_log(f"Number of labels: {len_labels}")

    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else ('mps' if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else 'cpu')
    )

    
    classifier.eval()
    classifier.to(device)

    # Load model state dict
    state_dict = torch.load(model_file, weights_only=True, map_location=device)
    classifier.load_state_dict(state_dict, strict=False)

    # Initialize DataCollator and DataLoader
    collator_function = DataCollatorForClassification(
                                            tokenizer=tokenizer,
                                            max_seq_length=config['max_seq_len'],
                                            label2id=label2id,
                                            id2label=id2label
                        )

    train_loader = DataLoader(
                        dataset,
                        batch_size=config['batch_size'],
                        shuffle=False,
                        collate_fn=collator_function,
                        num_workers=num_proc,
                        pin_memory=torch.cuda.is_available()
                    )

    # Load true labels if provided and file exists
    true_labels_dict = {}
    if true_labels_file and os.path.exists(true_labels_file):
        df_true = pd.read_csv(true_labels_file, sep=r'[,\t]', index_col=0, engine='python')
        true_labels_dict = df_true.to_dict()['ID']  # barcode -> label mapping
        debug_log(f"Loaded {len(true_labels_dict)} true labels from {true_labels_file}")

    debug_log(f"Inference started on device: {device}")
    
    return dataset, train_loader, classifier, device, label2id, id2label, true_labels_dict

def inference_loop(dataset, train_loader, classifier, device, label2id, id2label, true_labels_dict, df):
    """
    Runs the inference loop.
    
    Iterates over batches, collecting logits, probabilities, embeddings, and attention scores
    
    Supports validation loss and accuracy if ground-truth labels are available

    Args:
        dataset (Hugging Face Dataset): Preprocessed gene expression dataset
        train_loader (DataLoader): DataLoader for inference
        classifier (nn.Module): The classification model
        device (torch.device): Run inference on CPU or GPU
        label2id (dict): Mapping of label strings to IDs
        id2label (dict): Mapping of IDs to label strings
        true_labels_dict (dict): Barcode -> label mapping
        df (pd.DataFrame): Original gene expression matrix (used for barcode lookup)

    Returns:
        list: [CellName, PredictedLabel, Confidence] for each cell.
    """
    # toggles
    save_attn_inputs = config.get('save_attn_inputs', False)
    save_embeddings  = config.get('save_embeddings',  False)
    debug_log(f"FLAGS: save_attn_inputs={save_attn_inputs}, save_embeddings={save_embeddings}")

    preds_dict = defaultdict(list)
    true_labels, pred_labels = [], []
    probs_list = []  # Collect all softmax outputs across batches for log_loss
    logits_list = [] # Initialize list to store raw logits for each cell
    
    # Validation accumulators to track losses and accuracies across batches
    val_loss_total = 0.0
    val_correct = 0
    val_total = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    # Only if saving
    embeddings_list = [] if save_embeddings else None
    
    ids_list       = [] if save_attn_inputs else None
    mask_list      = [] if save_attn_inputs else None
    attn_list_save = [] if save_attn_inputs else None

    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            cell_indices = batch['cell_indices'].to(device)
            labels = batch['labels'].to(device)

            # Model inference
            #outputs, probs = classifier(input_ids, attention_mask)
            #logits_list.extend(outputs.cpu().numpy())  # Accumulate the logits from each batch
            #probs_list.extend(probs.cpu().numpy())     # Accumulate per-sample probs across all batches
            #_, predicted = torch.max(outputs.data, 1)
            outputs, _ = classifier(input_ids, attention_mask)          # ignore model's probs
            # Make fresh probs that exactly match the logits you pass around
            probs = torch.softmax(outputs.float(), dim=-1)               # [B, C]
            probs_np = probs.detach().cpu().numpy().astype(np.float64)
            # Extra safety against the sklearn warning:
            probs_np = np.clip(probs_np, 1e-12, 1.0)
            probs_np /= probs_np.sum(axis=1, keepdims=True)
            
            logits_list.extend(outputs.detach().cpu().numpy())
            probs_list.extend(probs_np)
            _, predicted = torch.max(outputs.data, 1)

            # For each cell, record predictions and optionally true label
            for index, pred, prob in zip(cell_indices, predicted.cpu().numpy(), probs.cpu().numpy()):  # GPU not supported
                barcode = df.index[int(index)]
                preds_dict['predictions'].append(pred)
                preds_dict['confidence'].append(prob[pred])
                preds_dict['cell_names'].append(barcode)

                # If true labels are available for this barcode, use them; otherwise, use -1.
                if barcode in true_labels_dict:  # or: if barcode in metadata.index
                    cell_type_label = true_labels_dict[barcode]
                    # cell_type_label = metadata.at[barcode, "ID"]
                    true_labels.append(label2id[cell_type_label])
                    # true_labels.append(label2id[true_labels_dict[barcode]]) # Equivalent
                else:
                    # debug_log(f"Warning: Barcode {barcode} not found in metadata. Assigning label -1.")
                    true_labels.append(-1)

                pred_labels.append(pred)

            # Collect embeddings 
            if save_embeddings:
                emb = classifier.base_model.embeddings(input_ids)                     # [B, L, D]
                embeddings_list.append(emb.detach().cpu().numpy())
            
            # Collect attention
            if save_attn_inputs:
                x = classifier.base_model.embeddings(input_ids)
                attn_last = None
                for li, enc in enumerate(classifier.base_model.encoders):
                    ret = (li == len(classifier.base_model.encoders) - 1)  # last layer only
                    x, aw = enc(x, attention_mask, return_attn=ret)        # aw: [B,H,L,L] on last
                    if aw is not None:
                        attn_last = aw
            
                # assert attn_last is not None and attn_last.ndim == 4
                # assert attention_mask.dtype is torch.bool
            
                ids_list.append(input_ids.detach().cpu().numpy())                          # [B, L]
                mask_list.append(attention_mask.detach().cpu().numpy())                    # [B, L]
                attn_list_save.append(attn_last.detach().to(torch.float16).cpu().numpy())  # [B,H,L,L]


        print("DEBUG: input_ids shape:", input_ids.shape)
        if save_attn_inputs and ids_list and len(ids_list) > 0:
            ids_all  = np.vstack(ids_list)
            mask_all = np.vstack(mask_list)
            attn_all = np.concatenate(attn_list_save, axis=0)
            np.savez_compressed(f"{output}_attn_inputs.npz",
                                attn=attn_all, input_ids=ids_all, key_pad=mask_all)
        else:
            debug_log("Attention inputs saving disabled or no batches; skipping _attn_inputs.npz.")


        
    print("val_total =", val_total)
    print("val_loss_total =", val_loss_total)

    # Save the logits
    # np.savez_compressed(f'{output}_logits.npz', np.array(logits_list))
    np.savetxt(f'{output}_logits.txt', np.array(logits_list), fmt='%.6f')
    
    # Filter valid true labels and their corresponding predictions
    filtered = [(t, p) for t, p in zip(true_labels, pred_labels) if t != -1]
    if filtered:
        valid_true_labels, valid_pred_labels = zip(*filtered)
        valid_true_labels = list(valid_true_labels)
        valid_pred_labels = list(valid_pred_labels)
    else:
        valid_true_labels, valid_pred_labels = [], []
    
    val_total = len(valid_true_labels)
    val_correct = sum([t == p for t, p in zip(valid_true_labels, valid_pred_labels)])
    val_accuracy = val_correct / val_total if val_total > 0 else np.nan
    
    # Validation loss
    try:
    #    val_probs = [p for p, t in zip(probs_list, true_labels) if t != -1] # Use the accumulated probs_list
    #    #val_probs = [probs[i] for i, t in enumerate(true_labels) if t != -1] # This grabs probs from the last batch
    #    val_loss_total = log_loss(valid_true_labels, val_probs, labels=list(range(len(id2label))))
    #except Exception as e:
    #    debug_log(f"[WARN] log_loss failed: {e}")
    #    val_loss_total = np.nan
    #    # Collect only the probabilities for samples that have a valid true label
        val_probs = np.asarray([p for p, t in zip(probs_list, true_labels) if t != -1], dtype=np.float64)
    
        # Safety: ensure each row sums to 1.0 before passing to sklearn
        val_probs = np.clip(val_probs, 1e-12, 1.0)
        val_probs /= val_probs.sum(axis=1, keepdims=True)
    
        val_loss_total = log_loss(valid_true_labels, val_probs, labels=list(range(len(id2label))))
    except Exception as e:
        debug_log(f"[WARN] log_loss failed: {e}")
        val_loss_total = np.nan

    
    # Generate confusion matrix
    if valid_true_labels and valid_pred_labels:
        generate_confusion_matrix(valid_true_labels, valid_pred_labels, id2label)
    else:
        debug_log("Skipping confusion matrix: No valid true labels found.")

    # If there are no valid labels, pass empty lists to skip classification report calculations
    if not valid_true_labels:
        debug_log("No valid true labels available; skipping classification report in metrics.")
        true_labels_to_use = []
        pred_labels_to_use = []
    else:
        # Otherwise, use the filtered (valid) labels
        true_labels_to_use = valid_true_labels
        pred_labels_to_use = valid_pred_labels

    compute_and_save_metrics(
            output_prefix=output,
            val_loss_total=val_loss_total,
            val_correct=val_correct,
            val_total=val_total,
            all_labels=true_labels_to_use,
            all_predictions=pred_labels_to_use,
            id2label=id2label,
            attention_scores=None,
            embeddings_list=embeddings_list
    )
    """
    save_inference_outputs(
        output_prefix=output,
        train_loss_total=train_loss_total,
        train_correct=train_correct,
        train_total=train_total,
        val_loss_total=val_loss_total,
        val_correct=val_correct,
        val_total=val_total,
        all_labels=true_labels_to_use,
        all_predictions=pred_labels_to_use,
        id2label=id2label,
        attention_scores=attention_scores,
        embeddings_list=embeddings_list
    )
    """

    debug_log("Saving tokenized datasets to disk.")
    cache_prefix = f"{prefix}"
    os.makedirs(os.path.dirname(cache_prefix), exist_ok=True)
    dataset.save_to_disk(cache_prefix)

    del dataset
    gc.collect()

    return [[cell_name, id2label[pred], conf] for cell_name, pred, conf in zip(
        preds_dict['cell_names'], preds_dict['predictions'], preds_dict['confidence']
    )]

def run_inference(df, true_labels_file=None):
    """
    Calls the main functions to prepare the inference components and to run the inference loop

    Args:
        df (pd.DataFrame): Gene expression matrix
        true_labels_file (str, optional): File containing true labels for evaluation

    Returns:
        list: [CellName, PredictedLabel, Confidence] for each cell
    """
    # Prepare inference components (dataset, model, dataloader, etc.)
    dataset, train_loader, classifier, device, label2id, id2label, true_labels_dict = \
        prepare_inference_components_for_inference(df, true_labels_file)

    # Run the inference loop using the prepared components.
    return inference_loop(
        dataset, 
        train_loader, 
        classifier, 
        device, 
        label2id, 
        id2label, 
        true_labels_dict, 
        df)
    
def run_no_batching(df):
    """
    Run inference on full dataframe without batching
    
    Saves output to disk as CSV
    """
    final_results = run_inference(df, true_labels_file)

    with open(f'{output}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['CellName', 'PredictedLabel', 'Confidence'])
        writer.writerows(final_results)

def run_with_batching(df, columns_per_slice):
    """
    Slice expression matrix and run inference per batch

    Args:
        df (pd.DataFrame): Gene expression matrix
        columns_per_slice (int): Number of columns (cells) per batch
    """
    total_columns = df.shape[1]
    num_slices = (total_columns // columns_per_slice) + (1 if total_columns % columns_per_slice != 0 else 0)

    with open(f'{output}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['CellName', 'PredictedLabel', 'Confidence'])

    # For batching the csv
    for i in range(num_slices):
        start_col = i * columns_per_slice
        end_col = min(start_col + columns_per_slice, total_columns)
        print(f"Batch {start_col + 1} to {end_col}:")
        df_slice = df.iloc[start_col:end_col, :]

        final_results = run_inference(df_slice, true_labels_file)

        with open(f'{output}.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(final_results)

        del df_slice
        del final_results
        gc.collect()
        print()

@track_performance
def start_loop():
    """
    Initializes the inference loop: 
        - Loads and optionally filters the expression matrix
        - Runs batched or full inference
        - Saves outputs to disk
    """
    debug_log(f"Importing the expression matrix from {input_file}\n")
    df = pd.read_csv(input_file, index_col=0, sep=r'[,\t]', engine='python', compression='gzip') if input_file.endswith('.gz') else pd.read_csv(input_file, index_col=0, sep=r'[,\t]', engine='python')

    """
    float_values = np.load('/home/data/spatial/Xenium_AA_values.npz')['float_values']
    index = np.load('/home/data/spatial/Xenium_AA_index.npy', allow_pickle=True)
    columns = np.load('/home/data/spatial/Xenium_AA_columns.npy', allow_pickle=True)
    df = pd.DataFrame(float_values, index=index, columns=columns)
    """
    """ 
    float_values = np.load('data/Victor_values_filtered.npz')['float_values']
    index = np.load('data/Victor_index_filtered.npy', allow_pickle=True)
    columns = np.load('data/Victor_columns_filtered.npy', allow_pickle=True)
    df = pd.DataFrame(float_values, index=index, columns=columns)
    """ 

    def filter_df(df):
        """Filter the dataset to retain only the hold-out barcodes."""
        with open(barcodes_file, "r") as f:
            inf_barcodes = set(f.read().splitlines())
        # filtered_df.set_index(df.columns[0], inplace=True)
        # filtered_df = df[df["CellName"].isin(inf_barcodes)]
        df_filtered = df[df.index.isin(inf_barcodes)]

        print(df_filtered.shape)

        return df_filtered

    if barcodes_file.lower() == "null" or not os.path.exists(barcodes_file):
        debug_log("No inference set found. Continuing to run inference on the unfiltered dataset.\n")
    else:
        df = filter_df(df)
        df_size = len(df)
        debug_log(f"Extracted the inference set, {((df_size - len(df)) / df_size) * 100}%, from the dataset.")

    debug_log(f"Matrix size: {len(df)} cells")
    debug_log(f"Running inference...\n")

    columns_per_slice = 20000
    # run_with_batching(df, columns_per_slice)

    run_no_batching(df)

    print("Finished")

if __name__ == "__main__":	
    if len(sys.argv) - 1 != 10:
        print(f"Error: Expected 10 arguments, but got {len(sys.argv) - 1}.")
        print("Usage: python preprocess.py <input_file> <genes_file> <prefix> <model_file> <labels_file> <true_labels_file> <barcodes_file> <config_file> <output> <num_proc>")
        sys.exit(1)

    input_file = sys.argv[1]
    genes_file = sys.argv[2]
    prefix = sys.argv[3]
    model_file = sys.argv[4]
    labels_file = sys.argv[5]
    true_labels_file = sys.argv[6]
    barcodes_file = sys.argv[7]
    config_file = sys.argv[8]
    output = sys.argv[9]
    num_proc = int(sys.argv[10])
    print(f'Parameters: {input_file}, {genes_file}, {prefix}, {model_file}, {labels_file}, {barcodes_file}, {config_file}, {output}, {num_proc}\n')

    load_config(config_file)

    config.setdefault('save_attn_inputs', False)   # writes <output>_attn_inputs.npz
    config.setdefault('save_embeddings',  False)   # writes <output>_embeddings.npz


    log_file = initialize_logging(prefix, context="inference")

    #classes = pd.read_csv(labels_file, sep=r'[,\t]', engine='python') # index_col=0,
    classes = pd.read_csv(labels_file, sep=r'[,\t]', engine='python', index_col=0)

    if true_labels_file == 'NULL':
        true_labels_file = ''

    start_loop()
