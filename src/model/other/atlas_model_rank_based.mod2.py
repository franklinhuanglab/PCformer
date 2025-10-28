import numpy as np
import torch
from torch import nn
from typing import Optional
from src import config


class ModelArgs:
    """
    Configuration container for the AtlasModelRankBased transformer model

    Attributes:
        embed_dim (int): Dimensionality of token embeddings
        num_layers (int): Number of transformer encoder layers
        num_heads (int): Number of attention heads in each encoder
        vocab_size (int): Total number of unique tokens (genes + special tokens)
        norm_eps (float): Small epsilon used in normalization for numerical stability
        max_seq_len (int): Maximum number of tokens per input sequence
        dropout (float): Dropout probability used throughout the model
        forward_expansion (int): Width expansion factor in the feed-forward sublayer
    """
    def __init__(self, vocab_size=13748):
        self.embed_dim = config["embed_dim"]
        self.num_layers = config["num_layers"]
        self.num_heads = config["num_heads"]
        self.vocab_size = vocab_size
        self.norm_eps = float(config["norm_eps"])
        self.max_seq_len = config["max_seq_len"]
        self.dropout = config["dropout"]
        self.forward_expansion = config["forward_expansion"]


class AtlasEmbeddingsRB(nn.Module):
    """
    Embedding layer for the Atlas transformer model

    Combines:
        - Gene (token) embeddings
        - Positional embeddings
        - RMS normalization
        - Dropout

    Args:
        args (ModelArgs): Model configuration object
    """
    def __init__(self, args: ModelArgs):
        super(AtlasEmbeddingsRB, self).__init__()

        self.vocab_size = args.vocab_size
        self.embed_dim = args.embed_dim
        self.max_seq_len = args.max_seq_len
        self.gene_embeddings = nn.Embedding(args.vocab_size, args.embed_dim)
        self.pos_embeddings = nn.Embedding(args.max_seq_len, args.embed_dim)
        # self.norm = nn.LayerNorm(args.embed_dim)
        self.norm = RMSNorm(args.embed_dim, args.norm_eps)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, input_ids_BL):
        """
        Computes combined token and positional embeddings

        Args:
            input_ids_BL (Tensor): Input tensor of token indices, shape [B, L]

        Returns:
            Tensor: Embedded sequence, shape [B, L, D]
        """
        seq_length = input_ids_BL.size(1)
        position_ids_1L = torch.arange(seq_length, dtype=torch.long, device=input_ids_BL.device).unsqueeze(0)

        gene_embeddings_BLD = self.gene_embeddings(input_ids_BL)
        pos_embeddings_BLD = self.pos_embeddings(position_ids_1L)

        embeddings_BLD = gene_embeddings_BLD + pos_embeddings_BLD
        embeddings_BLD = self.norm(embeddings_BLD)
        embeddings_BLD = self.dropout(embeddings_BLD)

        return embeddings_BLD


class AtlasEncoderRB(nn.Module):
    """
    Transformer encoder block with attention and feed-forward layers

    Components:
        - Multi-head self-attention
        - Two residual connections
        - Two RMSNorm layers
        - Feed-forward network (SiLU activation)

    Args:
        args (ModelArgs): Model configuration object
    """
    def __init__(self, args: ModelArgs):
        super(AtlasEncoderRB, self).__init__()
        self.embed_dim = args.embed_dim
        self.num_heads = args.num_heads
        self.forward_expansion = args.forward_expansion
        self.attention = nn.MultiheadAttention(args.embed_dim, args.num_heads, args.dropout, kdim=args.embed_dim, vdim=args.embed_dim)

        # Use RMSNorm for normalization
        # self.norm1 = nn.LayerNorm(args.embed_dim)
        self.norm1 = RMSNorm(args.embed_dim, args.norm_eps)

        self.feed_forward = nn.Sequential(
            nn.Linear(args.embed_dim, args.embed_dim * args.forward_expansion),
            nn.SiLU(),
            nn.Linear(args.embed_dim * args.forward_expansion, args.embed_dim)
        )
        # self.norm2 = nn.LayerNorm(args.embed_dim)
        self.norm2 = RMSNorm(args.embed_dim, args.norm_eps)

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, embeddings_BLD, key_padding_mask):
        """
        Forward pass of an encoder block

        Args:
            embeddings_BLD (Tensor): Input embeddings, shape [B, L, D]
            key_padding_mask (Tensor): Boolean mask of padded positions, shape [B, L]

        Returns:
            Tensor: Output embeddings, shape [B, L, D]
        """
        # Permute to shape [L, B, D] for MultiheadAttention
        x_LBD = embeddings_BLD.permute(1, 0, 2)
        attention_output_LBD, attn_weights = self.attention(x_LBD, x_LBD, x_LBD, key_padding_mask=key_padding_mask)
        # Permute back to [B, L, D]
        attention_output_BLD = attention_output_LBD.permute(1, 0, 2)
        # Residual connection and normalization
        x_BLD = self.norm1(attention_output_BLD + embeddings_BLD)
        x_BLD = self.dropout(x_BLD)
        # Feed-forward network
        ff_output_BLD = self.feed_forward(x_BLD)
        x_BLD = self.norm2(ff_output_BLD + x_BLD)
        output_BLD = self.dropout(x_BLD)
		
        self.last_attn = attn_weights  # [L, L] per head

        return output_BLD, attn_weights


class AtlasLMHeadRB(nn.Module):
    """
    Language modeling head for converting transformer outputs to vocabulary logits

    Components:
        - Linear projection
        - RMSNorm
        - Output decoder to vocab space

    Args:
        args (ModelArgs): Model configuration object
    """
    def __init__(self, args: ModelArgs):
        super(AtlasLMHeadRB, self).__init__()
        self.dense = nn.Linear(args.embed_dim, args.embed_dim)
        # self.norm = nn.LayerNorm(args.embed_dim)
        self.norm = RMSNorm(args.embed_dim, args.norm_eps)
        self.decoder = nn.Linear(args.embed_dim, args.vocab_size)

    def forward(self, features_BLD):
        """
        Computes logits over the vocabulary

        Args:
            features_BLD (Tensor): Transformer outputs, shape [B, L, D]

        Returns:
            Tensor: Vocabulary logits, shape [B, L, V]
        """
        x_BLD = self.dense(features_BLD)
        x_BLD = self.norm(x_BLD)
        preds_BLV = self.decoder(x_BLD)
        
        return preds_BLV


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)

    Normalizes the input based on the RMS of the last dimension
    and scales it using a learned weight vector

    Args:
        embed_dim (int): Size of the input vector
        norm_eps (float): Small constant to avoid division by zero
    """
    def __init__(self, embed_dim: int, norm_eps: float):
        super().__init__()
        self.norm_eps = norm_eps
        self.weight = nn.Parameter(torch.ones(embed_dim))

    def _norm(self, x):
        """
        Internal method for unscaled RMS normalization

        Args:
            x (Tensor): Input tensor

        Returns:
            Tensor: Normalized tensor
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.norm_eps)

    def forward(self, x):
        """
        Applies RMS normalization with learned scaling

        Args:
            x (Tensor): Input tensor

        Returns:
            Tensor: Normalized and scaled output
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class AtlasModelRankBased(nn.Module):
    """
    Transformer model for ranked gene expression modeling

    Designed for self-supervised tasks like masked token prediction on ranked-value scRNA-seq data

    Architecture:
        - Embedding layer: AtlasEmbeddingsRB
        - Transformer encoder stack: AtlasEncoderRB
        - LM prediction head: AtlasLMHeadRB

    Tensor Shapes:
        B: Batch size
        L: Sequence length
        D: Embedding dimension
        V: Vocabulary size

    Args:
        params (ModelArgs): Model configuration object
    """
    def __init__(self, params: ModelArgs):
        super(AtlasModelRankBased, self).__init__()
        self.params = params
        self.embeddings = AtlasEmbeddingsRB(params)
        self.encoders = nn.ModuleList([AtlasEncoderRB(params) for _ in range(params.num_layers)])
        self.lm_head = AtlasLMHeadRB(params)

    def forward_bkp(self, input_ids_BL, key_padding_mask, attn_mask=None):
        """
        Full forward pass from input tokens to vocabulary logits

        Args:
            input_ids_BL (Tensor): Input token indices, shape [B, L]
            key_padding_mask (Tensor): Mask indicating padded tokens, shape [B, L]
            attn_mask (Tensor, optional): Additional attention mask (unused)

        Returns:
            Tensor: Logits over vocabulary, shape [B, L, V]
        """
        embeddings_BLD = self.embeddings(input_ids_BL)
        for encoder in self.encoders:
            embeddings_BLD, _ = encoder(embeddings_BLD,key_padding_mask)

        preds_BLV = self.lm_head(embeddings_BLD)
        
        return preds_BLV

    def forward(self, input_ids, key_padding_mask):
        embeddings = self.embeddings(input_ids)
        all_attns = []
        for encoder in self.encoders:
            embeddings, attn = encoder(embeddings, key_padding_mask)
            all_attns.append(attn)   # e.g. [batch, heads, seq, seq]
        return { 
          "hidden_states": embeddings, 
          "attentions": all_attns 
        }
