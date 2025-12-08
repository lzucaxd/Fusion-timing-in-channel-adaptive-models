"""
Hierarchical Bag-of-Channels Set Transformer Vision Transformer (HierBoCSetViT) for CHAMMI.

Implements a hierarchical BoC model with:
- PerChannelEncoderTiny: Pretrained timm ViT-Tiny for patch-level attention within each channel
- ChannelSetTransformer: Set Transformer over channel embeddings (permutation-equivariant + PMA pooling)
- HierBoCSetViT: End-to-end model with channel permutation/dropout augmentation and supervised head

Key design principles:
- Patch-level hierarchical attention within each channel (pretrained ViT-Tiny)
- Set-like processing of channels (no positional encodings, permutation-equivariant)
- Permutation-invariant bag embedding via Pooling-by-Multihead-Attention (PMA)
- Channel permutation and dropout during training for robustness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional, Tuple, Union

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not available. PerChannelEncoderTiny will not work without timm.")


class PerChannelEncoderTiny(nn.Module):
    """
    Per-channel encoder using pretrained timm ViT-Tiny.
    
    Adapts a pretrained ViT-Tiny to single-channel input by:
    1. Replacing the 3-channel patch embedding with 1-channel (initialized by averaging RGB weights)
    2. Processing each channel independently with shared weights
    3. Returning per-channel CLS token or mean-pooled embeddings
    """
    
    def __init__(
        self,
        img_size: int = 128,
        embed_dim: Optional[int] = None,
        pretrained: bool = True,
        use_cls_token: bool = True,
    ):
        super().__init__()
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for PerChannelEncoderTiny. Install with: pip install timm")
        
        self.img_size = img_size
        self.use_cls_token = use_cls_token
        
        # Load pretrained ViT-Tiny from timm
        # vit_tiny_patch16_224 has embed_dim=192, depth=12, num_heads=3
        self.vit = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=pretrained,
            num_classes=0,  # Remove classifier head
            img_size=224,   # Original training size (we'll adapt pos embeddings)
        )
        
        # Get actual embed_dim from timm model
        actual_embed_dim = self.vit.embed_dim
        self.embed_dim = actual_embed_dim
        
        # Verify embed_dim parameter matches if provided
        if embed_dim is not None and embed_dim != actual_embed_dim:
            raise ValueError(
                f"embed_dim={embed_dim} does not match timm ViT-Tiny embed_dim={actual_embed_dim}. "
                f"Use embed_dim=None to auto-detect or embed_dim={actual_embed_dim}."
            )
        
        # Adapt patch embedding from 3 channels to 1 channel
        # Original: Conv2d(3, embed_dim, kernel_size=16, stride=16)
        original_proj = self.vit.patch_embed.proj
        
        # Create new 1-channel projection
        new_proj = nn.Conv2d(
            in_channels=1,
            out_channels=original_proj.out_channels,
            kernel_size=original_proj.kernel_size,
            stride=original_proj.stride,
            padding=original_proj.padding,
            bias=False if original_proj.bias is None else True,
        )
        
        # Initialize by averaging RGB weights
        with torch.no_grad():
            new_proj.weight.copy_(original_proj.weight.mean(dim=1, keepdim=True))
            if original_proj.bias is not None:
                new_proj.bias.copy_(original_proj.bias)
        
        self.vit.patch_embed.proj = new_proj
        
        # Adapt positional embeddings if img_size != 224
        # Original: 224x224 with patch_size=16 → 14×14 = 196 patches
        # Target: 128x128 with patch_size=16 → 8×8 = 64 patches
        if img_size != 224:
            patch_size = 16  # Fixed patch size
            old_grid_size = 224 // patch_size  # 14
            new_grid_size = img_size // patch_size  # 8 for 128x128
            
            if new_grid_size != old_grid_size:
                # Resize positional embeddings from old_grid_size to new_grid_size
                # pos_embed shape: (1, 1 + num_patches, embed_dim) = (1, 1 + 196, 192)
                old_pos_embed = self.vit.pos_embed.data  # (1, 197, 192)
                
                # Extract CLS token and patch embeddings separately
                cls_token = old_pos_embed[:, :1, :]  # (1, 1, 192)
                patch_pos_embed = old_pos_embed[:, 1:, :]  # (1, 196, 192)
                
                # Reshape patch embeddings to spatial grid: (1, 196, 192) → (1, 14, 14, 192)
                patch_pos_embed = patch_pos_embed.reshape(1, old_grid_size, old_grid_size, actual_embed_dim)
                
                # Interpolate to new grid size: (1, 14, 14, 192) → (1, 8, 8, 192)
                patch_pos_embed = F.interpolate(
                    patch_pos_embed.permute(0, 3, 1, 2),  # (1, 192, 14, 14)
                    size=(new_grid_size, new_grid_size),
                    mode='bilinear',
                    align_corners=False,
                ).permute(0, 2, 3, 1)  # (1, 8, 8, 192)
                
                # Flatten back: (1, 8, 8, 192) → (1, 64, 192)
                patch_pos_embed = patch_pos_embed.reshape(1, new_grid_size * new_grid_size, actual_embed_dim)
                
                # Concatenate CLS token back: (1, 1, 192) + (1, 64, 192) → (1, 65, 192)
                new_pos_embed = torch.cat([cls_token, patch_pos_embed], dim=1)
                
                # Replace positional embeddings
                self.vit.pos_embed = nn.Parameter(new_pos_embed)
                
                # Update patch_embed attributes to match new image size
                self.vit.patch_embed.img_size = (img_size, img_size)
                self.vit.patch_embed.num_patches = new_grid_size * new_grid_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode each channel independently.
        
        Args:
            x: (B, C, H, W) multi-channel images
        Returns:
            z_channels: (B, C, embed_dim) per-channel embeddings
        """
        B, C, H, W = x.shape
        
        # Verify input size matches expected img_size
        if H != self.img_size or W != self.img_size:
            raise ValueError(
                f"Input image size ({H}, {W}) does not match expected img_size={self.img_size}. "
                f"Please ensure inputs are {self.img_size}x{self.img_size}."
            )
        
        # Collapse channels into batch dimension: (B, C, H, W) -> (B*C, 1, H, W)
        x_flat = x.view(B * C, 1, H, W)
        
        # Use timm's forward_features method which handles positional embeddings automatically
        # forward_features returns tokens after all blocks and norm
        tokens = self.vit.forward_features(x_flat)  # (B*C, num_patches+1, embed_dim)
        
        # Extract representation
        if self.use_cls_token:
            # CLS token is first token (timm ViT always includes CLS token)
            z = tokens[:, 0]  # (B*C, embed_dim)
        else:
            # Mean pooling over all tokens (skip CLS token)
            z = tokens[:, 1:].mean(dim=1)  # (B*C, embed_dim)
        
        # Reshape back: (B*C, embed_dim) -> (B, C, embed_dim)
        z_channels = z.view(B, C, self.embed_dim)
        
        return z_channels


class PerChannelEncoderSmall(nn.Module):
    """
    Per-channel encoder using pretrained timm ViT-Small.
    
    Adapts a pretrained ViT-Small to single-channel input by:
    1. Replacing the 3-channel patch embedding with 1-channel (initialized by averaging RGB weights)
    2. Processing each channel independently with shared weights
    3. Returning per-channel CLS token or mean-pooled embeddings
    """
    
    def __init__(
        self,
        img_size: int = 224,
        embed_dim: Optional[int] = None,
        pretrained: bool = True,
        use_cls_token: bool = True,
    ):
        super().__init__()
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for PerChannelEncoderSmall. Install with: pip install timm")
        
        self.img_size = img_size
        self.use_cls_token = use_cls_token
        
        # Load pretrained ViT-Small from timm
        # vit_small_patch16_224 has embed_dim=384, depth=12, num_heads=6
        self.vit = timm.create_model(
            "vit_small_patch16_224",
            pretrained=pretrained,
            num_classes=0,  # Remove classifier head
            img_size=224,   # Original training size
        )
        
        # Get actual embed_dim from timm model
        actual_embed_dim = self.vit.embed_dim
        self.embed_dim = actual_embed_dim
        
        # Verify embed_dim parameter matches if provided
        if embed_dim is not None and embed_dim != actual_embed_dim:
            raise ValueError(
                f"embed_dim={embed_dim} does not match timm ViT-Small embed_dim={actual_embed_dim}. "
                f"Use embed_dim=None to auto-detect or embed_dim={actual_embed_dim}."
            )
        
        # Adapt patch embedding from 3 channels to 1 channel
        original_proj = self.vit.patch_embed.proj
        
        # Create new 1-channel projection
        new_proj = nn.Conv2d(
            in_channels=1,
            out_channels=original_proj.out_channels,
            kernel_size=original_proj.kernel_size,
            stride=original_proj.stride,
            padding=original_proj.padding,
            bias=False if original_proj.bias is None else True,
        )
        
        # Initialize by averaging RGB weights
        with torch.no_grad():
            new_proj.weight.copy_(original_proj.weight.mean(dim=1, keepdim=True))
            if original_proj.bias is not None:
                new_proj.bias.copy_(original_proj.bias)
        
        self.vit.patch_embed.proj = new_proj
        
        # For 224x224, no positional embedding adaptation needed (matches pretrained model)
        # 224x224 with patch_size=16 → 14×14 = 196 patches (matches pretrained)
        if img_size != 224:
            patch_size = 16
            old_grid_size = 224 // patch_size  # 14
            new_grid_size = img_size // patch_size
            
            if new_grid_size != old_grid_size:
                # Resize positional embeddings
                old_pos_embed = self.vit.pos_embed.data
                cls_token = old_pos_embed[:, :1, :]
                patch_pos_embed = old_pos_embed[:, 1:, :]
                
                patch_pos_embed = patch_pos_embed.reshape(1, old_grid_size, old_grid_size, actual_embed_dim)
                patch_pos_embed = F.interpolate(
                    patch_pos_embed.permute(0, 3, 1, 2),
                    size=(new_grid_size, new_grid_size),
                    mode='bilinear',
                    align_corners=False,
                ).permute(0, 2, 3, 1)
                
                patch_pos_embed = patch_pos_embed.reshape(1, new_grid_size * new_grid_size, actual_embed_dim)
                new_pos_embed = torch.cat([cls_token, patch_pos_embed], dim=1)
                
                self.vit.pos_embed = nn.Parameter(new_pos_embed)
                self.vit.patch_embed.img_size = (img_size, img_size)
                self.vit.patch_embed.num_patches = new_grid_size * new_grid_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode each channel independently.
        
        Args:
            x: (B, C, H, W) multi-channel images
        Returns:
            z_channels: (B, C, embed_dim) per-channel embeddings
        """
        B, C, H, W = x.shape
        
        # Verify input size matches expected img_size
        if H != self.img_size or W != self.img_size:
            raise ValueError(
                f"Input image size ({H}, {W}) does not match expected img_size={self.img_size}. "
                f"Please resize images to {self.img_size}x{self.img_size}."
            )
        
        # Collapse channels into batch dimension: (B, C, H, W) -> (B*C, 1, H, W)
        x_flat = x.view(B * C, 1, H, W)
        
        # Use timm's forward_features method
        tokens = self.vit.forward_features(x_flat)  # (B*C, num_patches+1, embed_dim)
        
        # Extract CLS token or mean-pool patches
        if self.use_cls_token:
            z = tokens[:, 0]  # CLS token: (B*C, embed_dim)
        else:
            z = tokens[:, 1:].mean(dim=1)  # Mean over patches: (B*C, embed_dim)
        
        # Reshape back: (B*C, embed_dim) -> (B, C, embed_dim)
        z_channels = z.view(B, C, self.embed_dim)
        
        return z_channels


class ChannelBlock(nn.Module):
    """
    Self-attention block for Set Transformer over channel embeddings.
    
    Permutation-equivariant: no positional encodings along channel dimension.
    """
    
    def __init__(
        self,
        embed_dim: int = 192,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
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
        """
        Args:
            x: (B, C, D) channel embeddings
        Returns:
            x: (B, C, D) transformed channel embeddings
        """
        # Self-attention over channels (permutation-equivariant)
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x


class ChannelSetTransformer(nn.Module):
    """
    Set Transformer over channel embeddings.
    
    Architecture:
    1. Stack of permutation-equivariant self-attention blocks (no positional encodings)
    2. Pooling-by-Multihead-Attention (PMA) with learned bag query → permutation-invariant bag embedding
    """
    
    def __init__(
        self,
        embed_dim: int = 192,
        depth: int = 2,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Stack of equivariant self-attention blocks
        self.blocks = nn.ModuleList([
            ChannelBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Learned bag query for PMA (Pooling-by-Multihead-Attention)
        self.bag_query = nn.Parameter(torch.randn(1, 1, embed_dim))
        nn.init.normal_(self.bag_query, std=0.02)
    
    def forward(
        self,
        z_channels: torch.Tensor,
        return_attn: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            z_channels: (B, C, D) per-channel embeddings
            return_attn: if True, also return channel attention weights from PMA
        Returns:
            z_bag: (B, D) permutation-invariant bag embedding
            attn_weights: (B, C) if return_attn=True (PMA attention over channels)
        """
        x = z_channels  # (B, C, D)
        
        # Equivariant self-attention blocks over channels
        for blk in self.blocks:
            x = blk(x)  # (B, C, D)
        
        B, C, D = x.shape
        
        # Pooling-by-Multihead-Attention (PMA)
        # Learned bag query
        q = self.bag_query.expand(B, 1, D)  # (B, 1, D)
        
        # Compute attention scores: query attends to channel embeddings
        scores = torch.bmm(q, x.transpose(1, 2)) / (D ** 0.5)  # (B, 1, C)
        attn = scores.softmax(dim=-1)  # (B, 1, C)
        
        # Weighted sum over channels → permutation-invariant bag embedding
        z_bag = torch.bmm(attn, x).squeeze(1)  # (B, D)
        
        if return_attn:
            attn_weights = attn.squeeze(1)  # (B, C)
            return z_bag, attn_weights
        
        return z_bag


class HierBoCSetViT(nn.Module):
    """
    Hierarchical Bag-of-Channels Set Transformer Vision Transformer.
    
    Architecture:
    1. PerChannelEncoderTiny: Pretrained ViT-Tiny for patch-level attention within each channel
    2. ChannelSetTransformer: Set Transformer over channel embeddings (equivariant + PMA pooling)
    3. Head: Supervised head (CE or ProxyNCA++)
    
    Features:
    - Channel permutation and dropout during training for robustness
    - Permutation-invariant bag embedding
    - Optional attention weight visualization
    """
    
    def __init__(
        self,
        img_size: int = 128,
        embed_dim: int = 192,
        encoder_pretrained: bool = True,
        encoder_type: Literal["tiny", "small"] = "tiny",
        aggregator_depth: int = 2,
        aggregator_num_heads: int = 3,
        aggregator_mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        head_mode: Literal["ce", "proxynca"] = "ce",
        num_classes: Optional[int] = None,
        metric_embed_dim: Optional[int] = None,
        channel_dropout_p: float = 0.3,
        use_cls_token: bool = True,
    ):
        super().__init__()
        self.head_mode = head_mode
        self.channel_dropout_p = channel_dropout_p
        
        # Per-channel encoder (pretrained ViT-Tiny or ViT-Small)
        if encoder_type == "tiny":
            self.encoder = PerChannelEncoderTiny(
                img_size=img_size,
                embed_dim=embed_dim,  # Will auto-detect from timm model if None
                pretrained=encoder_pretrained,
                use_cls_token=use_cls_token,
            )
        elif encoder_type == "small":
            self.encoder = PerChannelEncoderSmall(
                img_size=img_size,
                embed_dim=embed_dim,  # Will auto-detect from timm model if None
                pretrained=encoder_pretrained,
                use_cls_token=use_cls_token,
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}. Must be 'tiny' or 'small'.")
        
        # Get actual embed_dim from encoder (in case it was auto-detected)
        self.embed_dim = self.encoder.embed_dim
        
        # Channel Set Transformer aggregator
        self.aggregator = ChannelSetTransformer(
            embed_dim=self.embed_dim,
            depth=aggregator_depth,
            num_heads=aggregator_num_heads,
            mlp_ratio=aggregator_mlp_ratio,
            dropout=dropout,
        )
        
        # Head
        if head_mode == "ce":
            if num_classes is None:
                raise ValueError("num_classes must be specified for CE head")
            self.head = nn.Linear(self.embed_dim, num_classes)
        elif head_mode == "proxynca":
            metric_embed_dim = metric_embed_dim or (self.embed_dim // 2)
            self.head = nn.Sequential(
                nn.Linear(self.embed_dim, metric_embed_dim),
                nn.LayerNorm(metric_embed_dim),
            )
        else:
            raise ValueError(f"Unknown head_mode: {head_mode}")
    
    def _apply_channel_permutation_and_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply random channel permutation and optional channel dropout during training.
        
        Args:
            x: (B, C, H, W) multi-channel images
        Returns:
            x_aug: (B, C, H, W) augmented images
        """
        B, C, H, W = x.shape
        
        if self.training:
            # Random permutation per sample
            perms = torch.stack(
                [torch.randperm(C, device=x.device) for _ in range(B)],
                dim=0
            )  # (B, C)
            
            # Apply permutation: x[b, perms[b], :, :]
            batch_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, C)
            x = x[batch_indices, perms, :, :]
            
            # Channel dropout: with probability p, drop exactly 1 random channel per sample
            if self.channel_dropout_p > 0:
                mask = torch.ones(B, C, device=x.device, dtype=x.dtype)
                drop_idx = torch.randint(0, C, (B,), device=x.device)
                drop_sample = torch.rand(B, device=x.device) < self.channel_dropout_p
                
                for b in range(B):
                    if drop_sample[b]:
                        mask[b, drop_idx[b]] = 0.0
                
                x = x * mask[:, :, None, None]
        
        return x
    
    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: (B, C, H, W) multi-channel images
            return_attn: if True, also return channel attention weights from aggregator
        Returns:
            If head_mode="ce":
                logits: (B, num_classes)
                optionally (logits, attn_weights) if return_attn=True
            If head_mode="proxynca":
                embedding: (B, metric_embed_dim) L2-normalized
                optionally (embedding, attn_weights) if return_attn=True
        """
        # Apply channel augmentation (permutation + dropout)
        x = self._apply_channel_permutation_and_dropout(x)
        
        # Hierarchy level 1: patch-level attention within each channel
        z_channels = self.encoder(x)  # (B, C, embed_dim)
        
        # Hierarchy level 2: Set Transformer over channel embeddings
        if return_attn:
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
        No channel permutation/dropout here - raw representation.
        
        Args:
            x: (B, C, H, W) multi-channel images
        Returns:
            z_channels: (B, C, embed_dim) per-channel embeddings
        """
        return self.encoder(x)
    
    def extract_bag_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract bag embedding (after aggregation, before head).
        No channel permutation/dropout here - raw representation.
        
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

