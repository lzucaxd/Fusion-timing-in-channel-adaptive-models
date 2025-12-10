"""
Early Fusion Vision Transformer for Channel-Adaptive Models.

In Early Fusion, channels are mixed immediately at the patch embedding layer.
The model accepts variable input channels (3, 4, or 5) and processes them together.
"""

import torch
import torch.nn as nn
import timm

def generate_2d_sincos_pos_embeddings(
    embedding_dim: int,
    length: int,
    scale: float = 10000.0,
    use_class_token: bool = True,
    num_modality: int = 1,
):
    linear_positions = torch.arange(length, dtype=torch.float32)
    height_mesh, width_mesh = torch.meshgrid(linear_positions, linear_positions, indexing="ij")

    positional_dim = embedding_dim // 4
    positional_weights = torch.arange(positional_dim, dtype=torch.float32) / positional_dim
    positional_weights = 1.0 / (scale**positional_weights)

    height_weights = torch.outer(height_mesh.flatten(), positional_weights)
    width_weights = torch.outer(width_mesh.flatten(), positional_weights)

    positional_encoding = torch.cat([
        torch.sin(height_weights),
        torch.cos(height_weights),
        torch.sin(width_weights),
        torch.cos(width_weights),
    ], dim=1)[None, :, :]

    positional_encoding = positional_encoding.repeat(1, num_modality, 1)

    if use_class_token:
        class_token = torch.zeros([1, 1, embedding_dim], dtype=torch.float32)
        positional_encoding = torch.cat([class_token, positional_encoding], dim=1)

    return nn.Parameter(positional_encoding, requires_grad=False)


class ChannelAdaptivePatchEmbed(nn.Module):
    def __init__(
        self,
        embed_dim: int = 192,          # <– if you want ViT-small, use 384
        patch_size: int = 16,
        img_size: int = 128,
        max_in_chans: int = 5,         # <– NEW
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.max_in_chans = max_in_chans

        self.num_patches = (img_size // patch_size) ** 2

        # Single projection for up to max_in_chans channels
        self.proj = nn.Conv2d(
            max_in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W), C in {3,4,5}
        """
        B, C, H, W = x.shape
        if C > self.max_in_chans:
            raise ValueError(f"Got {C} channels, but max_in_chans={self.max_in_chans}")

        # Pad channels up to max_in_chans with zeros
        if C < self.max_in_chans:
            pad = self.max_in_chans - C
            # pad along channel dim: (left,right,top,bottom,front,back...) – here channels first
            x = torch.cat([x, torch.zeros(B, pad, H, W, device=x.device, dtype=x.dtype)], dim=1)

        # (B, max_in_chans, H, W) -> (B, embed_dim, H//P, W//P)
        x = self.proj(x)

        # -> (B, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x



class EarlyFusionViT(nn.Module):
    def __init__(
        self,
        num_classes: int,
        img_size: int = 128,
        patch_size: int = 16,
        embed_dim: int = 192,      # <– ViT-tiny
        depth: int = 12,
        num_heads: int = 3,        # <– ViT-tiny
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        pretrained: bool = False,
        max_in_chans: int = 5,     # <– NEW: we support 3,4,5 channels
    ):
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
            img_size=img_size,
            max_in_chans=max_in_chans,   # <– pass here
        )

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        length = img_size // patch_size
        self.pos_embed = generate_2d_sincos_pos_embeddings(
            embedding_dim=embed_dim,
            length=length,
            use_class_token=True,
            num_modality=1,   # early-fusion → no modality repetition
)

        self.pos_drop = nn.Dropout(p=drop_rate)

        from timm.models.vision_transformer import Block as VisionTransformerBlock
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
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

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()


    def _init_weights(self):
        # class token
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # DO NOT touch self.pos_embed if using sin-cos (it’s fixed)

        # patch proj
        nn.init.trunc_normal_(self.patch_embed.proj.weight, std=0.02)
        if self.patch_embed.proj.bias is not None:
            nn.init.zeros_(self.patch_embed.proj.bias)

        # head
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
<<<<<<< Updated upstream
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
=======
        B, C, H, W = x.shape
        if C not in [3, 4, 5]:
            raise ValueError(f"Unsupported channel count: {C}. Expected 3, 4, or 5.")

        x = self.patch_embed(x)              # (B, N, D)

        cls_tokens = self.cls_token.expand(B, 1, -1)
>>>>>>> Stashed changes
        x = torch.cat([cls_tokens, x], dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)

        cls_token_final = x[:, 0]
        logits = self.head(cls_token_final)
        return logits
<<<<<<< Updated upstream
    
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
=======

    def get_attention_maps(self, x: torch.Tensor):
        B, C, H, W = x.shape

        # Patch embedding (handles 3/4/5 chans internally)
>>>>>>> Stashed changes
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        attention_maps = []
        for block in self.blocks:
            x = block(x)
            attention_maps.append(None)  # still placeholder

        return attention_maps
