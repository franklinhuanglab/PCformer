#!/usr/bin/env python3
import numpy as np
import torch
from torch.utils.data import Dataset
from src import config


class GeneExpressionDatasetRB(Dataset):
    """
    PyTorch data handler for gene expression data with built-in support for train/test splitting

    Steps:
        - Loads and splits gene expression data into train and test sets
        - Stores each split as a `SplitDataset` for indexed access
        - Assumes input gene expression values are strings representing NumPy arrays, which are evaluated and converted

    Args:
        split_ratio (float): Fraction of data to reserve for testing (validation) (default: 0.2)
        train (SplitDataset): Training subset
        test (SplitDataset or None): Test subset; `None` if no samples after split

    Raises:
        AssertionError: If lengths of gene_names and gene_expressions do not match
    """
    def __init__(self, gene_names, gene_expressions, split_ratio=0.2):
        assert len(gene_names) == len(gene_expressions), "Mismatched gene expressions and names lengths."

        self.split_ratio = split_ratio
        self.data = {'train': None, 'test': None}

        self._prepare_data(gene_names, gene_expressions)

    def _prepare_data(self, gene_names, gene_expressions):
        """
        Splits input data into train and test subsets

        This function:
            - Evaluates gene_expressions from strings to NumPy arrays
            - Randomly shuffles indices
            - Creates `SplitDataset` objects for each subset

        Args:
            gene_names (array): List of gene names
            gene_expressions (array): Stringified expression arrays to be evaluated and converted
        """
        total_size = len(gene_expressions)
        gene_expressions = np.array(eval(gene_expressions))
        train_size = int((1.0 - self.split_ratio) * total_size)
        indices = torch.randperm(total_size).tolist()
        train_indices, test_indices = indices[:train_size], indices[train_size:]

        self.train = SplitDataset(gene_names[train_indices], gene_expressions[train_indices])

        if test_indices:
            self.test = SplitDataset(gene_names[test_indices], gene_expressions[test_indices])
        else:
            self.test = None

    def __getitem__(self, split):
        """
        Returns the dataset split specified by the `split` key

        Args:
            split (str): One of 'train' or 'test'

        Returns:
            SplitDataset: Corresponding dataset split

        Raises:
            KeyError: If an invalid split name is provided
        """
        if split == 'train':
            return self.train
        elif split == 'test':
            return self.test
        else:
            raise KeyError("Split not recognized. Use 'train' or 'test'.")

    def __repr__(self):
        """
        Returns a human-readable string representation of the dataset

        Shows whether training and testing splits exist

        Returns:
            str: Summary of dataset structure
        """
        return f"GeneExpressionDatasetRB({{\n    train: {self.train},\n    test: {self.test}\n}})"


class SplitDataset(Dataset):
    """
    A wrapper class representing a single subset (train or test) of the gene expression dataset

    Enables index-based access to corresponding gene names and expression values
    """
    def __init__(self, gene_names, gene_expressions):
        """
        Initializes a split dataset

        Args:
            gene_names (array-like): Gene identifiers
            gene_expressions (array-like): Corresponding expression arrays
        """
        self.gene_names = gene_names
        self.gene_expressions = gene_expressions

    def __len__(self):
        """
        Returns:
            int: Number of samples in the split
        """
        return len(self.gene_expressions)

    def __getitem__(self, idx):
        """
        Retrieves a sample at the given index

        Args:
            idx (int): Index of the sample to retrieve

        Returns:
            dict: {
                'gene_names': gene name(s) at index,
                'gene_expressions': expression values at index
            }
        """
        return {
            "gene_names": self.gene_names[idx],
            "gene_expressions": self.gene_expressions[idx]
        }

    def __repr__(self):
        """
        Returns:
            str: String representation showing structure of the dataset split
        """
        return f"({{\n features: ['gene_names', gene_expressions], \n    num_rows: {self.__len__()}\n}})"
