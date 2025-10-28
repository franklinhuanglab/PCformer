#!/usr/bin/env python3
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from src import config

class DataCollatorForClassification:
    """
    Data collator for classification tasks using gene token sequences

    Processes each sample in a batch by:
      - Trimming or truncating the tokenized gene list to `max_seq_length`
      - Padding sequences to uniform length using the tokenizer's pad token
      - Creating attention masks to identify padded positions
      - Converting string cell labels into numeric IDs using `label2id`

    Args:
        tokenizer (GeneTokenizer): Tokenizer that provides `pad_token_id` and handles gene-to-ID mapping
        max_seq_length (int, optional): Maximum number of tokens per sequence. Defaults to 512
        label2id (dict, optional): Mapping from string labels to numeric label IDs
        id2label (dict, optional): Reverse mapping from numeric IDs to string labels
    """
    def __init__(self, tokenizer, max_seq_length=512, label2id=None, id2label=None):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id
        self.label2id = label2id if label2id is not None else {}
        self.id2label = id2label if id2label is not None else {}
        # self.unknown_label_id = self.label2id.get("<UNK>", -1)  # Default to -1 if unknown label is not defined

    def __call__(self, batch):
        max_length = min(self.max_seq_length, max([len(seq['tokenized_genes']) for seq in batch]))
        batch_input_ids = []
        batch_attention_masks = []
        batch_labels = []

        for item in batch:
            input_ids = item['tokenized_genes'][:max_length]  # Trim or use the whole input_ids if shorter than max_length
            label = item['cell_label']

            # Convert labels to ids
            label_id = self.label2id[label]

            # Match input_ids length to max_length via padding
            padded_input_ids = self.pad_tokens(input_ids, max_length)
            padding_mask = [i == self.pad_token_id for i in input_ids]

            batch_input_ids.append(padded_input_ids)
            batch_attention_masks.append(padding_mask)
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
        """
        Pads or truncates a sequence of token IDs to a fixed length.

        Args:
            input_ids (list): Sequence of token IDs
            max_length (int): Target length for the sequence

        Returns:
            list: Token sequence of length `max_length`, padded with the pad token
        """
        padding_length = max_length - len(input_ids)
        input_ids.extend([self.pad_token_id] * (max_length - len(input_ids)))
        return input_ids + [self.pad_token_id] * padding_length if padding_length > 0 else input_ids[:max_length]


class DataCollatorForGeneModeling:
    """
    Data collator for masked language modeling (MLM) on gene token sequences.

    Prepares batches for self-supervised training by:
      - Trimming sequences to `max_seq_length`.
      - Randomly masking tokens with a specified probability (`mlm_probability`)
      - Producing labels that contain original tokens only at masked positions (others set to -100)
      - Padding sequences and labels to a fixed length

    Args:
        tokenizer (GeneTokenizer): Tokenizer providing `pad_token_id` and `mask_token_id`
        mlm_probability (float, optional): Probability of masking each token. Defaults to 0.15
        max_seq_length (int, optional): Maximum length of token sequences. Defaults to 512
    """
    def __init__(self, tokenizer, mlm_probability=0.15, max_seq_length=512): # 512, 2048
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = tokenizer.mask_token_id

    def __call__(self, batch):
        max_length = min(self.max_seq_length, max([len(seq) for seq in batch]))
        batch_input_ids = []
        batch_padding_mask = []
        batch_labels = []

        for seq in batch:
            # Trim sequences to max_seq_length
            trimmed_seq = seq[:max_length]
            input_ids, labels = self.mask_tokens(trimmed_seq, max_length)
            padding_mask = [i == self.pad_token_id for i in input_ids]

            batch_input_ids.append(input_ids)
            batch_padding_mask.append(padding_mask)
            batch_labels.append(labels)

        batch_input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
        batch_padding_mask = torch.tensor(batch_padding_mask, dtype=torch.bool)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)

        return {
            'input_ids': batch_input_ids,
            'padding_mask': batch_padding_mask,
            'labels': batch_labels
        }

    def mask_tokens(self, input_ids, max_length):
        """
        Applies masked language modeling (MLM) token masking to a gene sequence.

        Masking strategy:
          - 80% of masked tokens are replaced with [MASK]
          - 10% are replaced with a random token
          - 10% are left unchanged

        Args:
            input_ids (list): Original sequence of token IDs
            max_length (int): Target length after padding

        Returns:
            tuple:
                input_ids (list): Modified input sequence with applied masking
                labels (list): Original tokens at masked positions; -100 elsewhere
        """
        # Create labels array
        labels = [-100] * max_length

        # Determine which tokens to mask for MLM
        probability_matrix = np.full(len(input_ids), self.mlm_probability)
        masked_indices = np.random.rand(len(input_ids)) < probability_matrix

        for idx in range(len(input_ids)):
            if masked_indices[idx]:
                labels[idx] = input_ids[idx]  # Original token is the label for MLM
                if np.random.rand() < 0.8:  # 80% -> MASK
                    input_ids[idx] = self.mask_token_id
                elif np.random.rand() < 0.5:  # 10% -> Random token
                    input_ids[idx] = np.random.randint(3, len(self.tokenizer.idx2token.keys())) # Special tokens: 0, 1, 2

        # Pad input_ids to max_length
        input_ids.extend([self.pad_token_id] * (max_length - len(input_ids)))

        return input_ids, labels
