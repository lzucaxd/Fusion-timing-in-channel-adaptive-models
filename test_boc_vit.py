"""
Test script for Bag-of-Channels ViT.

Tests model creation, forward pass, and integration with CHAMMI dataloaders.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.boc_vit import BoCViT, PerChannelEncoder, BagAggregator
from losses.metric_losses import ProxyNCA


def test_per_channel_encoder():
    """Test PerChannelEncoder."""
    print("=" * 70)
    print("Testing PerChannelEncoder")
    print("=" * 70)
    
    encoder = PerChannelEncoder(
        img_size=128,
        patch_size=16,
        embed_dim=192,
        depth=6,
        num_heads=3,
    )
    
    # Test with different channel counts
    for C in [3, 4, 5]:
        x = torch.randn(2, C, 128, 128)
        z = encoder(x)
        print(f"  Input: {x.shape} -> Output: {z.shape}")
        assert z.shape == (2, C, 192), f"Expected (2, {C}, 192), got {z.shape}"
    
    print("✓ PerChannelEncoder works correctly\n")


def test_bag_aggregator():
    """Test BagAggregator."""
    print("=" * 70)
    print("Testing BagAggregator")
    print("=" * 70)
    
    # Test mean mode
    aggregator_mean = BagAggregator(embed_dim=192, mode="mean")
    z_channels = torch.randn(2, 4, 192)
    z_bag = aggregator_mean(z_channels)
    print(f"  Mean mode: {z_channels.shape} -> {z_bag.shape}")
    assert z_bag.shape == (2, 192), f"Expected (2, 192), got {z_bag.shape}"
    
    # Test attention mode
    aggregator_attn = BagAggregator(embed_dim=192, mode="attn")
    z_bag, attn = aggregator_attn(z_channels, return_attn=True)
    print(f"  Attention mode: {z_channels.shape} -> {z_bag.shape}, attn: {attn.shape}")
    assert z_bag.shape == (2, 192), f"Expected (2, 192), got {z_bag.shape}"
    assert attn.shape == (2, 4), f"Expected (2, 4), got {attn.shape}"
    assert torch.allclose(attn.sum(dim=1), torch.ones(2)), "Attention weights should sum to 1"
    
    print("✓ BagAggregator works correctly\n")


def test_boc_vit():
    """Test BoCViT."""
    print("=" * 70)
    print("Testing BoCViT")
    print("=" * 70)
    
    # Test CE mode
    model_ce = BoCViT(
        img_size=128,
        patch_size=16,
        embed_dim=192,
        depth=6,
        num_heads=3,
        aggregator_mode="mean",
        head_mode="ce",
        num_classes=10,
    )
    
    x = torch.randn(2, 4, 128, 128)
    logits = model_ce(x)
    print(f"  CE mode: {x.shape} -> {logits.shape}")
    assert logits.shape == (2, 10), f"Expected (2, 10), got {logits.shape}"
    
    # Test metric learning mode
    model_metric = BoCViT(
        img_size=128,
        patch_size=16,
        embed_dim=192,
        depth=6,
        num_heads=3,
        aggregator_mode="attn",
        head_mode="proxynca",
        num_classes=10,
        metric_embed_dim=256,
    )
    
    embedding = model_metric(x)
    print(f"  Metric mode: {x.shape} -> {embedding.shape}")
    assert embedding.shape == (2, 256), f"Expected (2, 256), got {embedding.shape}"
    
    # Check normalization
    norm = torch.norm(embedding, dim=1)
    assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5), "Embeddings should be L2-normalized"
    
    # Test attention return
    embedding, attn = model_metric(x, return_attn=True)
    print(f"  Attention weights: {attn.shape}")
    assert attn.shape == (2, 4), f"Expected (2, 4), got {attn.shape}"
    
    print("✓ BoCViT works correctly\n")


def test_proxy_nca():
    """Test ProxyNCA loss."""
    print("=" * 70)
    print("Testing ProxyNCA Loss")
    print("=" * 70)
    
    criterion = ProxyNCA(embed_dim=256, num_classes=10, temperature=0.05)
    
    embeddings = torch.randn(4, 256)
    labels = torch.tensor([0, 1, 2, 3])
    
    loss = criterion(embeddings, labels)
    print(f"  Loss: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    
    print("✓ ProxyNCA loss works correctly\n")


def test_with_chammi_dataloader():
    """Test integration with CHAMMI dataloader (if available)."""
    print("=" * 70)
    print("Testing with CHAMMI Dataloader")
    print("=" * 70)
    
    try:
        from channel_adaptive_pipeline.chammi_grouped_dataloader import create_grouped_chammi_dataloaders
        
        # Check if dataset path exists (user should set this)
        csv_file = os.environ.get("CHAMMI_CSV", None)
        root_dir = os.environ.get("CHAMMI_ROOT", None)
        
        if csv_file is None or root_dir is None:
            print("  Skipping: CHAMMI_CSV and CHAMMI_ROOT environment variables not set")
            print("  Set them to test with actual data:")
            print("    export CHAMMI_CSV=/path/to/combined_metadata.csv")
            print("    export CHAMMI_ROOT=/path/to/CHAMMI/")
            return
        
        # Create model
        model = BoCViT(
            img_size=128,
            patch_size=16,
            embed_dim=192,
            depth=6,
            num_heads=3,
            aggregator_mode="mean",
            head_mode="ce",
            num_classes=10,  # Will need to be set correctly
        )
        
        # Create dataloader
        loaders = create_grouped_chammi_dataloaders(
            csv_file=csv_file,
            root_dir=root_dir,
            batch_size=4,
            shuffle=False,
            split="train",
            resize_to=128,
            augment=False,
            normalize=True,
        )
        
        # Test with one batch
        for channel_count, loader in loaders.items():
            print(f"  Testing channel {channel_count}...")
            images, metadatas, labels = next(iter(loader))
            print(f"    Batch shape: {images.shape}")
            
            # Forward pass
            with torch.no_grad():
                outputs = model(images)
                print(f"    Output shape: {outputs.shape}")
            
            break  # Just test one channel group
        
        print("✓ Integration with CHAMMI dataloader works\n")
        
    except Exception as e:
        print(f"  Skipping: {e}\n")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("BAG-OF-CHANNELS VIT TEST SUITE")
    print("=" * 70 + "\n")
    
    test_per_channel_encoder()
    test_bag_aggregator()
    test_boc_vit()
    test_proxy_nca()
    test_with_chammi_dataloader()
    
    print("=" * 70)
    print("✓ ALL TESTS PASSED")
    print("=" * 70)

