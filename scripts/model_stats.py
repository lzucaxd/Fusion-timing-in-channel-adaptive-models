#!/usr/bin/env python3
"""
Print model parameter counts and architecture details for README.
"""

import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hier_boc_setvit import HierBoCSetViT

def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_stats(encoder_type, image_size=128, num_classes=20, 
                      channel_embed_mode="attn_pool", pma_num_seeds=1):
    """Print model statistics."""
    model = HierBoCSetViT(
        num_classes=num_classes,
        encoder_type=encoder_type,
        img_size=image_size,
        embed_dim=None,  # Auto-detect from encoder
        channel_embed_mode=channel_embed_mode,
        pma_num_seeds=pma_num_seeds,
        use_channel_gating=False,
    )
    
    total_params = count_parameters(model)
    
    # Count encoder params
    encoder_params = 0
    if hasattr(model, 'encoder'):
        encoder_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    
    aggregator_params = 0
    if hasattr(model, 'channel_set_transformer'):
        aggregator_params = sum(p.numel() for p in model.channel_set_transformer.parameters() if p.requires_grad)
    
    head_params = 0
    if hasattr(model, 'head'):
        head_params = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    
    # Architecture details
    if encoder_type == "tiny":
        embed_dim = 192
        depth = 12
        num_heads = 3
        patch_size = 16
    elif encoder_type == "small":
        embed_dim = 384
        depth = 12
        num_heads = 6
        patch_size = 16
    else:
        embed_dim = depth = num_heads = patch_size = "Unknown"
    
    print(f"\n=== HierBoCSetViT-{encoder_type.capitalize()} (image_size={image_size}) ===")
    print(f"Total parameters: {total_params:,}")
    print(f"  - Encoder (per-channel): ~{encoder_params:,}")
    print(f"  - Aggregator: ~{aggregator_params:,}")
    print(f"  - Head: ~{head_params:,}")
    print(f"\nArchitecture:")
    print(f"  - Embedding dimension: {embed_dim}")
    print(f"  - Encoder depth: {depth} layers")
    print(f"  - Attention heads: {num_heads}")
    print(f"  - Patch size: {patch_size}×{patch_size}")
    print(f"  - Image size: {image_size}×{image_size}")
    print(f"  - Channel embedding: {channel_embed_mode}")
    print(f"  - PMA seeds: {pma_num_seeds}")
    
    return total_params

if __name__ == "__main__":
    print("Model Parameter Counts for README")
    print("=" * 60)
    
    # ViT-Tiny
    print_model_stats("tiny", image_size=128, channel_embed_mode="attn_pool", pma_num_seeds=4)
    
    # ViT-Small
    print_model_stats("small", image_size=128, channel_embed_mode="attn_pool", pma_num_seeds=4)
    
    print("\n" + "=" * 60)

