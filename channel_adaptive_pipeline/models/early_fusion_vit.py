"""
Early Fusion Vision Transformer for Channel-Adaptive Models.

In Early Fusion, channels are mixed immediately at the patch embedding layer.
The model accepts variable input channels (3, 4, or 5) and processes them together.
"""

import torch
import torch.nn as nn
import timm
from typing import Optional


class ChannelAdaptivePatchEmbed(nn.Module):
    """
    Patch embedding layer that adapts to variable input channels.
    
    For ViT-Tiny with patch size 16, the standard embedding expects 3 channels.
    This layer dynamically handles 3, 4, or 5 channels by:
    1. Using a learned projection for each possible channel count
    2. Or using a single projection and handling channel mismatch
    """
    
    def __init__(self, embed_dim: int = 192, patch_size: int = 16, img_size: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        
        # Calculate number of patches
        self.num_patches = (img_size // patch_size) ** 2
        
        # Create separate patch embeddings for each channel count
        # This allows the model to learn channel-specific features
        self.patch_embeds = nn.ModuleDict({
            '3': nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size),
            '4': nn.Conv2d(4, embed_dim, kernel_size=patch_size, stride=patch_size),
            '5': nn.Conv2d(5, embed_dim, kernel_size=patch_size, stride=patch_size),
        })
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W) where C ∈ {3, 4, 5}
        Returns:
            Patch embeddings of shape (B, num_patches, embed_dim)
        """
        B, C, H, W = x.shape
        
        # Select appropriate patch embedding based on channel count
        if C not in [3, 4, 5]:
            raise ValueError(f"Unsupported channel count: {C}. Expected 3, 4, or 5.")
        
        patch_embed = self.patch_embeds[str(C)]
        
        # Apply patch embedding: (B, C, H, W) -> (B, embed_dim, H//patch_size, W//patch_size)
        x = patch_embed(x)
        
        # Flatten spatial dimensions: (B, embed_dim, H//patch_size, W//patch_size) -> (B, embed_dim, num_patches)
        x = x.flatten(2)
        
        # Transpose to (B, num_patches, embed_dim)
        x = x.transpose(1, 2)
        
        return x


class EarlyFusionViT(nn.Module):
    """
    Early Fusion Vision Transformer for channel-adaptive models.
    
    Channels are mixed immediately at the patch embedding layer.
    This is the simplest fusion strategy where all channels are processed together.
    
    Architecture:
    - Channel-adaptive patch embedding (handles 3, 4, or 5 channels)
    - Standard ViT encoder (no channel separation)
    - Classification head
    """
    
    def __init__(
        self,
        num_classes: int,
        img_size: int = 128,
        patch_size: int = 16,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        pretrained: bool = False,
    ):
        """
        Args:
            num_classes: Number of classification classes
            img_size: Input image size (default 128 for CHAMMI)
            patch_size: Patch size for ViT (default 16)
            embed_dim: Embedding dimension (192 for ViT-Tiny)
            depth: Number of transformer blocks (12 for ViT-Tiny)
            num_heads: Number of attention heads (3 for ViT-Tiny)
            mlp_ratio: MLP expansion ratio
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            drop_path_rate: Drop path rate for stochastic depth
            pretrained: Whether to use pretrained weights (not applicable for channel-adaptive)
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        
        # Channel-adaptive patch embedding
        self.patch_embed = ChannelAdaptivePatchEmbed(
            embed_dim=embed_dim,
            patch_size=patch_size,
            img_size=img_size
        )
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # Dropout after patch embedding
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer blocks - use timm's VisionTransformerBlock
        from timm.models.vision_transformer import Block as VisionTransformerBlock
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            VisionTransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU,
            )
            for i in range(depth)
        ])
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        # Initialize class token
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize positional embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Initialize patch embeddings
        for patch_embed in self.patch_embed.patch_embeds.values():
            nn.init.trunc_normal_(patch_embed.weight, std=0.02)
            if patch_embed.bias is not None:
                nn.init.zeros_(patch_embed.bias)
        
        # Initialize classification head
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W) where C ∈ {3, 4, 5}
        
        Returns:
            Logits of shape (B, num_classes)
        """
        B, C, H, W = x.shape
        
        if C not in [3, 4, 5]:
            raise ValueError(f"Unsupported channel count: {C}. Expected 3, 4, or 5.")
        
        # Patch embedding: (B, C, H, W) -> (B, num_patches, embed_dim)
        x = self.patch_embed(x)
        
        # Add class token: (B, num_patches, embed_dim) -> (B, num_patches + 1, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        x = self.blocks(x)
        
        # Layer norm
        x = self.norm(x)
        
        # Extract class token (first token)
        cls_token_final = x[:, 0]
        
        # Classification head
        logits = self.head(cls_token_final)
        
        return logits
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before classification head.
        
        Args:
            x: Input tensor of shape (B, C, H, W) where C ∈ {3, 4, 5}
        
        Returns:
            Features of shape (B, embed_dim) - class token after norm layer
        """
        B, C, H, W = x.shape
        
        if C not in [3, 4, 5]:
            raise ValueError(f"Unsupported channel count: {C}. Expected 3, 4, or 5.")
        
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, 1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        
        # Return class token (first token)
        return x[:, 0]
    
    def get_attention_maps(self, x: torch.Tensor):
        """
        Get attention maps for visualization (optional).
        
        Args:
            x: Input tensor of shape (B, C, H, W) where C ∈ {3, 4, 5}
        
        Returns:
            List of attention maps from each transformer block
        """
        B, C, H, W = x.shape
        
        if C not in [3, 4, 5]:
            raise ValueError(f"Unsupported channel count: {C}. Expected 3, 4, or 5.")
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Collect attention maps (simplified - timm blocks don't directly return attention)
        # This would need custom attention extraction if needed
        attention_maps = []
        for block in self.blocks:
            # Note: timm blocks don't expose attention directly in forward
            # Would need to hook into attention layers for visualization
            x = block(x)
            attention_maps.append(None)  # Placeholder
        
        return attention_maps

