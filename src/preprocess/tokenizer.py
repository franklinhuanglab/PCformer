#!/usr/bin/env python3
import numpy as np
#from src.model import PCA_GENES
#from src.model import snRNA_GENES
from src.model import XENIUM_GENES

GENES = XENIUM_GENES

class GeneTokenizer:
    """
    Tokenizer used to convert gene names into token IDs and vice versa,
    based on a predefined set of valid genes (PCA_GENES, snRNA_GENES, XENIUM_GENES).

    Special tokens:
        <PAD>: Padding token
        <MASK>: Used for masked modeling tasks
        <UNK>: Represents unknown or invalid genes
    """
    def __init__(self, pad_token="<PAD>", mask_token="<MASK>", unknown_token="<UNK>", genes=None):
        # If a custom gene list is passed, use it; otherwise keep the current default (GENES)
        self.valid_genes = list(genes) if genes is not None else GENES
        self.token2idx = {pad_token: 0, mask_token: 1, unknown_token: 2}
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
        self.idx2token.update({idx + 3: gene for idx, gene in enumerate(self.valid_genes)})
        self.token2idx.update({gene: idx + 3 for idx, gene in enumerate(self.valid_genes)})

        self.pad_token_id = self.token2idx[pad_token]
        self.mask_token_id = self.token2idx[mask_token]
        self.unknown_token_id = self.token2idx[unknown_token]

    def tokenize(self, gene_names, gene_expressions, use_unknown=False):
        """
        Converts the list of gene names and their expression values into token IDs.

        Args:
            gene_names (list): List of gene names
            gene_expressions (list or np.array): Corresponding expression values
            use_unknown (bool): If True, replace unknown genes with <UNK>
                                If False, unknown genes are dropped

        Returns:
            np.ndarray: Sequence of token IDs (int16)
        """
        gene_names = np.array(gene_names)
        gene_expressions = np.array(gene_expressions, dtype=np.float32)
        assert isinstance(gene_expressions, np.ndarray), "Expressions must be a NumPy object array."

        # Process each sample individually
        tokenized_genes, valid_expressions = [], []
        clean_names, clean_expr = self._remove_zeros(gene_names, gene_expressions)
        ranked_names, ranked_expr = self._rank_genes(clean_names, clean_expr)
        valid_names = self._filter_invalid_genes(ranked_names, ranked_expr, use_unknown)
        tokenized_genes = self._convert_to_token(valid_names, use_unknown)

        tokenized_genes = np.array(tokenized_genes, dtype=np.int16)
        return tokenized_genes

    def detokenize(self, token_ids):
        """
        Converts token IDs back to gene names.

        Args:
            token_ids (list[int]): Sequence of token IDs

        Returns:
            list[str]: Corresponding gene names
        """
        return [self.idx2token.get(idx, self.unknown_token_id) for idx in token_ids]

    def __call__(self, gene_names, gene_expressions, use_unknown=False):
        """
        Shortcut for `self.tokenize(...)`.

        Returns:
            np.ndarray: Sequence of token IDs
        """
        return self.tokenize(gene_names, gene_expressions, use_unknown)
        
    ##################### Helper functions #####################
    @staticmethod
    def normalize_expressions(gene_expressions):
        """
        Normalizes gene expression values to sum to 1.

        Args:
            gene_expressions (np.ndarray): Raw expression values

        Returns:
            np.ndarray: Normalized expression values
        """
        # Normalize each gene's expression by the total expression in the cell
        total_expression = np.sum(gene_expressions)
        if total_expression == 0:
            return gene_expressions  # Handle zero-expression cases
        return gene_expressions / total_expression

    def _remove_zeros(self, gene_names, gene_expressions):
        """
        Excludes genes with zero expression.

        Returns:
            Tuple of filtered (gene_names, gene_expressions)
        """
        nonzero_indices = np.nonzero(gene_expressions)
        return gene_names[nonzero_indices], gene_expressions[nonzero_indices]

    def _rank_genes(self, gene_names, gene_expressions):
        """
        Ranks genes by their expression value in each cell in descending order.

        Returns:
            Tuple of (ranked_gene_names, ranked_expressions)
        """
        ranked_indices = np.argsort(-gene_expressions)
        return [gene_names[i] for i in ranked_indices], gene_expressions[ranked_indices]

    def _filter_invalid_genes(self, gene_names, gene_expressions, use_unknown):
        """
        Filters out invalid genes or replaces them with <UNK>.

        Returns:
            List of filtered or replaced gene names
        """
        if use_unknown:
            valid_names = [name if name in self.valid_genes else "<UNK>" for name in gene_names]
        else:
            valid_indices = [i for i, name in enumerate(gene_names) if name in self.valid_genes]
            valid_names = [gene_names[i] for i in valid_indices]
            # gene_expressions = gene_expressions[valid_indices]
        return valid_names

    def _convert_to_token(self, gene_names, use_unknown):
        """
        Converts gene names to token IDs.

        Args:
            gene_names (list): List of gene names
            use_unknown (bool): Whether to replace unknowns with <UNK>

        Returns:
            list[int]: List of token IDs
        """
        return [self.token2idx.get(name, self.unknown_token_id if use_unknown else None) for name in gene_names]
