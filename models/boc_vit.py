"""
Bag-of-Channels Vision Transformer (BoC-ViT) for CHAMMI.

Implements supervised BoC-ViT with:
- PerChannelEncoder: Shared ViT-Tiny encoder for independent channel encoding
- BagAggregator: Permutation-invariant aggregation (mean or attention pooling)
- BoCViT: End-to-end model with supervised head (CE or metric learning)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional, Tuple, Union


class PatchEmbedding(nn.Module):
    """Patch embedding for single-channel images."""
    
    def __init__(self, img_size: int = 128, patch_size: int = 16, embed_dim: int = 192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding: 1 channel -> embed_dim
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W) single-channel images
        Returns:
            patches: (B, num_patches, embed_dim)
        """
        # (B, 1, H, W) -> (B, embed_dim, H//patch_size, W//patch_size)
        x = self.proj(x)
        # Flatten spatial dimensions: (B, embed_dim, H', W') -> (B, embed_dim, H'*W')
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H'*W', embed_dim)
        return x


class TransformerBlock(nn.Module):
    """Standard transformer encoder block."""
    
    def __init__(self, embed_dim: int = 192, num_heads: int = 3, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, num_patches, embed_dim)"""
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class PerChannelEncoder(nn.Module):
    """
    Shared ViT-Tiny encoder for independent channel encoding.
    
    Processes each channel independently with shared weights.
    No channel embeddings or IDs - channels are treated as unordered set.
    """
    
    def __init__(
        self,
        img_size: int = 128,
        patch_size: int = 16,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_cls_token: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.use_cls_token = use_cls_token
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, embed_dim)
        
        # CLS token (optional)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + (1 if use_cls_token else 0), embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode channels independently.
        
        Args:
            x: (B, C, H, W) multi-channel images or (B*C, 1, H, W) flattened single-channel images
               If (B, C, H, W), will reshape to (B*C, 1, H, W) internally
        
        Returns:
            z_channels: (B, C, embed_dim) per-channel embeddings
        """
        # Handle input shape
        if x.dim() == 4 and x.shape[1] > 1:
            # (B, C, H, W) -> (B*C, 1, H, W)
            B, C, H, W = x.shape
            x = x.view(B * C, 1, H, W)
            batch_size = B
            num_channels = C
        elif x.dim() == 4 and x.shape[1] == 1:
            # Already (B*C, 1, H, W) - need to infer B and C from context
            # For now, assume caller provides (B*C, 1, H, W) and we need to track batch info
            # This is a bit tricky - let's require (B, C, H, W) input for clarity
            raise ValueError("Please provide input as (B, C, H, W) for multi-channel images")
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        # Patch embedding: (B*C, 1, H, W) -> (B*C, num_patches, embed_dim)
        x = self.patch_embed(x)  # (B*C, num_patches, embed_dim)
        
        # Add CLS token if used
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # (B*C, 1, embed_dim)
            x = torch.cat([cls_tokens, x], dim=1)  # (B*C, num_patches+1, embed_dim)
        
        # Add positional embeddings
        x = x + self.pos_embed  # (B*C, num_patches+1, embed_dim)
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Extract representation
        if self.use_cls_token:
            # Use CLS token: (B*C, 1, embed_dim) -> (B*C, embed_dim)
            z = x[:, 0]
        else:
            # Mean pooling: (B*C, num_patches, embed_dim) -> (B*C, embed_dim)
            z = x.mean(dim=1)
        
        # Reshape back to (B, C, embed_dim)
        z = z.view(batch_size, num_channels, self.embed_dim)
        
        return z


class BagAggregator(nn.Module):
    """
    Permutation-invariant aggregator over channel embeddings.
    
    Two modes:
    - "mean": DeepSets-style mean pooling + optional MLP
    - "attn": Attention pooling with learnable query (returns attention weights for visualization)
    """
    
    def __init__(
        self,
        embed_dim: int = 192,
        mode: Literal["mean", "attn"] = "mean",
        mlp_hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.mode = mode
        
        if mode == "mean":
            # Optional MLP after mean pooling
            if mlp_hidden_dim is not None:
                self.mlp = nn.Sequential(
                    nn.Linear(embed_dim, mlp_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_hidden_dim, embed_dim),
                )
            else:
                self.mlp = nn.Identity()
        elif mode == "attn":
            # Learnable query for attention pooling
            self.query = nn.Parameter(torch.randn(1, embed_dim))
            # Optional query projection
            self.query_proj = nn.Linear(embed_dim, embed_dim)
            self.scale = embed_dim ** -0.5
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def forward(
        self,
        z_channels: torch.Tensor,
        return_attn: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Aggregate channel embeddings into bag representation.
        
        Args:
            z_channels: (B, C, embed_dim) per-channel embeddings
            return_attn: If True and mode="attn", also return attention weights
        
        Returns:
            z_bag: (B, embed_dim) bag embedding
            attn_weights (optional): (B, C) attention weights (only if mode="attn" and return_attn=True)
        """
        if self.mode == "mean":
            # Mean pooling: (B, C, embed_dim) -> (B, embed_dim)
            z_mean = z_channels.mean(dim=1)
            z_bag = self.mlp(z_mean)
            
            if return_attn:
                # For mean mode, return uniform attention weights
                B, C = z_channels.shape[:2]
                attn_weights = torch.ones(B, C, device=z_channels.device) / C
                return z_bag, attn_weights
            return z_bag
        
        elif self.mode == "attn":
            B, C, D = z_channels.shape
            
            # Compute attention scores
            # Query: (1, embed_dim) -> (B, embed_dim)
            q = self.query_proj(self.query).expand(B, -1)  # (B, embed_dim)
            
            # Keys: (B, C, embed_dim)
            # Attention: (B, embed_dim) @ (B, embed_dim, C) -> (B, C)
            # Or: (B, 1, embed_dim) @ (B, embed_dim, C) -> (B, 1, C)
            q = q.unsqueeze(1)  # (B, 1, embed_dim)
            scores = torch.bmm(q, z_channels.transpose(1, 2))  # (B, 1, C)
            scores = scores.squeeze(1) * self.scale  # (B, C)
            
            # Softmax over channels
            attn_weights = F.softmax(scores, dim=1)  # (B, C)
            
            # Weighted sum: (B, C, embed_dim) * (B, C, 1) -> (B, embed_dim)
            attn_weights_expanded = attn_weights.unsqueeze(2)  # (B, C, 1)
            z_bag = (z_channels * attn_weights_expanded).sum(dim=1)  # (B, embed_dim)
            
            if return_attn:
                return z_bag, attn_weights
            return z_bag


class BoCViT(nn.Module):
    """
    Bag-of-Channels Vision Transformer for supervised learning.
    
    Architecture:
    1. PerChannelEncoder: Independent encoding of each channel
    2. BagAggregator: Permutation-invariant aggregation
    3. Head: Supervised head (CE or metric learning)
    """
    
    def __init__(
        self,
        img_size: int = 128,
        patch_size: int = 16,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        aggregator_mode: Literal["mean", "attn"] = "mean",
        aggregator_mlp_hidden: Optional[int] = None,
        head_mode: Literal["ce", "proxynca"] = "ce",
        num_classes: Optional[int] = None,
        metric_embed_dim: Optional[int] = None,
        use_cls_token: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_mode = head_mode
        self.aggregator_mode = aggregator_mode
        
        # Per-channel encoder
        self.encoder = PerChannelEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            use_cls_token=use_cls_token,
        )
        
        # Bag aggregator
        self.aggregator = BagAggregator(
            embed_dim=embed_dim,
            mode=aggregator_mode,
            mlp_hidden_dim=aggregator_mlp_hidden,
            dropout=dropout,
        )
        
        # Head
        if head_mode == "ce":
            if num_classes is None:
                raise ValueError("num_classes must be specified for CE head")
            self.head = nn.Linear(embed_dim, num_classes)
        elif head_mode == "proxynca":
            metric_embed_dim = metric_embed_dim or embed_dim
            self.head = nn.Sequential(
                nn.Linear(embed_dim, metric_embed_dim),
                nn.LayerNorm(metric_embed_dim),
            )
        else:
            raise ValueError(f"Unknown head_mode: {head_mode}")
    
    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: (B, C, H, W) multi-channel images
            return_attn: If True and aggregator_mode="attn", return attention weights
        
        Returns:
            If head_mode="ce":
                logits: (B, num_classes)
            If head_mode="proxynca":
                embedding: (B, metric_embed_dim) L2-normalized
            If return_attn=True and aggregator_mode="attn":
                (output, attn_weights) where attn_weights: (B, C)
        """
        # Encode channels independently: (B, C, H, W) -> (B, C, embed_dim)
        z_channels = self.encoder(x)
        
        # Aggregate: (B, C, embed_dim) -> (B, embed_dim)
        if return_attn and self.aggregator_mode == "attn":
            z_bag, attn_weights = self.aggregator(z_channels, return_attn=True)
        else:
            z_bag = self.aggregator(z_channels, return_attn=False)
            attn_weights = None
        
        # Head
        if self.head_mode == "ce":
            logits = self.head(z_bag)
            if return_attn and attn_weights is not None:
                return logits, attn_weights
            return logits
        elif self.head_mode == "proxynca":
            embedding = self.head(z_bag)
            embedding = F.normalize(embedding, p=2, dim=1)  # L2 normalize
            if return_attn and attn_weights is not None:
                return embedding, attn_weights
            return embedding
    
    def extract_channel_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract per-channel embeddings without aggregation.
        
        Args:
            x: (B, C, H, W) multi-channel images
        Returns:
            z_channels: (B, C, embed_dim) per-channel embeddings
        """
        return self.encoder(x)
    
    def extract_bag_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract bag embedding (after aggregation, before head).
        
        Args:
            x: (B, C, H, W) multi-channel images
        Returns:
            z_bag: (B, embed_dim) bag embedding
        """
        z_channels = self.encoder(x)
        z_bag = self.aggregator(z_channels, return_attn=False)
        return z_bag
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the embedding space optimized by the training loss.
        
        - For head_mode == 'proxynca': post-head, L2-normalized embedding.
        - For head_mode == 'ce': pre-head bag embedding.
        
        Args:
            x: (B, C, H, W) multi-channel images
        Returns:
            embedding: (B, D) where D is metric_embed_dim for proxynca, embed_dim for ce
        """
        if self.head_mode == "proxynca":
            return self(x)  # Already returns post-head, L2-normalized embedding
        elif self.head_mode == "ce":
            return self.extract_bag_embedding(x)  # Pre-head bag embedding
        else:
            raise ValueError(f"Unknown head_mode: {self.head_mode}")

