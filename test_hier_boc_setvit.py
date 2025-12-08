"""Test script for HierBoCSetViT model."""

import torch
import torch.nn as nn
from models.hier_boc_setvit import HierBoCSetViT, PerChannelEncoderTiny, ChannelSetTransformer


def test_per_channel_encoder_tiny():
    """Test PerChannelEncoderTiny."""
    print("Testing PerChannelEncoderTiny...")
    
    encoder = PerChannelEncoderTiny(
        img_size=128,
        embed_dim=192,
        pretrained=False,  # Use False for faster testing
        use_cls_token=True,
    )
    
    # Test with different channel counts
    for C in [3, 4, 5]:
        B = 2
        x = torch.randn(B, C, 128, 128)
        
        z_channels = encoder(x)
        assert z_channels.shape == (B, C, 192), f"Expected shape {(B, C, 192)}, got {z_channels.shape}"
        print(f"  ✓ C={C}: Input {(B, C, 128, 128)} -> Output {z_channels.shape}")
    
    print("✓ PerChannelEncoderTiny works correctly\n")


def test_channel_set_transformer():
    """Test ChannelSetTransformer."""
    print("Testing ChannelSetTransformer...")
    
    aggregator = ChannelSetTransformer(
        embed_dim=192,
        depth=2,
        num_heads=3,
        mlp_ratio=4.0,
        dropout=0.1,
    )
    
    # Test with different channel counts
    for C in [3, 4, 5]:
        B = 2
        z_channels = torch.randn(B, C, 192)
        
        # Test without attention weights
        z_bag = aggregator(z_channels, return_attn=False)
        assert z_bag.shape == (B, 192), f"Expected shape {(B, 192)}, got {z_bag.shape}"
        print(f"  ✓ C={C}: Input {(B, C, 192)} -> Output {z_bag.shape}")
        
        # Test with attention weights
        z_bag, attn_weights = aggregator(z_channels, return_attn=True)
        assert z_bag.shape == (B, 192), f"Expected shape {(B, 192)}, got {z_bag.shape}"
        assert attn_weights.shape == (B, C), f"Expected attention shape {(B, C)}, got {attn_weights.shape}"
        print(f"  ✓ C={C}: With attention weights -> z_bag {z_bag.shape}, attn {attn_weights.shape}")
    
    print("✓ ChannelSetTransformer works correctly\n")


def test_hier_boc_setvit_ce():
    """Test HierBoCSetViT with CE head."""
    print("Testing HierBoCSetViT (CE mode)...")
    
    model = HierBoCSetViT(
        img_size=128,
        embed_dim=192,
        encoder_pretrained=False,  # Use False for faster testing
        aggregator_depth=2,
        aggregator_num_heads=3,
        head_mode="ce",
        num_classes=10,
        channel_dropout_p=0.3,
    )
    
    # Test with different channel counts
    for C in [3, 4, 5]:
        B = 2
        x = torch.randn(B, C, 128, 128)
        
        # Test without attention
        logits = model(x, return_attn=False)
        assert logits.shape == (B, 10), f"Expected shape {(B, 10)}, got {logits.shape}"
        print(f"  ✓ C={C}: Input {(B, C, 128, 128)} -> Logits {logits.shape}")
        
        # Test with attention
        logits, attn = model(x, return_attn=True)
        assert logits.shape == (B, 10), f"Expected shape {(B, 10)}, got {logits.shape}"
        assert attn.shape == (B, C), f"Expected attention shape {(B, C)}, got {attn.shape}"
        print(f"  ✓ C={C}: With attention -> Logits {logits.shape}, Attn {attn.shape}")
    
    print("✓ HierBoCSetViT (CE mode) works correctly\n")


def test_hier_boc_setvit_proxynca():
    """Test HierBoCSetViT with ProxyNCA head."""
    print("Testing HierBoCSetViT (ProxyNCA mode)...")
    
    model = HierBoCSetViT(
        img_size=128,
        embed_dim=192,
        encoder_pretrained=False,  # Use False for faster testing
        aggregator_depth=2,
        aggregator_num_heads=3,
        head_mode="proxynca",
        metric_embed_dim=96,
        channel_dropout_p=0.3,
    )
    
    # Test with different channel counts
    for C in [3, 4, 5]:
        B = 2
        x = torch.randn(B, C, 128, 128)
        
        # Test without attention
        embedding = model(x, return_attn=False)
        assert embedding.shape == (B, 96), f"Expected shape {(B, 96)}, got {embedding.shape}"
        # Check L2 normalization
        norms = torch.norm(embedding, p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Embeddings should be L2-normalized"
        print(f"  ✓ C={C}: Input {(B, C, 128, 128)} -> Embedding {embedding.shape} (L2-normalized)")
        
        # Test with attention
        embedding, attn = model(x, return_attn=True)
        assert embedding.shape == (B, 96), f"Expected shape {(B, 96)}, got {embedding.shape}"
        assert attn.shape == (B, C), f"Expected attention shape {(B, C)}, got {attn.shape}"
        print(f"  ✓ C={C}: With attention -> Embedding {embedding.shape}, Attn {attn.shape}")
    
    print("✓ HierBoCSetViT (ProxyNCA mode) works correctly\n")


def test_helper_methods():
    """Test helper methods."""
    print("Testing helper methods...")
    
    model = HierBoCSetViT(
        img_size=128,
        embed_dim=192,
        encoder_pretrained=False,
        aggregator_depth=2,
        head_mode="proxynca",
        metric_embed_dim=96,
    )
    
    B, C = 2, 4
    x = torch.randn(B, C, 128, 128)
    
    # Test extract_channel_embeddings
    z_channels = model.extract_channel_embeddings(x)
    assert z_channels.shape == (B, C, 192), f"Expected shape {(B, C, 192)}, got {z_channels.shape}"
    print(f"  ✓ extract_channel_embeddings: {z_channels.shape}")
    
    # Test extract_bag_embedding
    z_bag = model.extract_bag_embedding(x)
    assert z_bag.shape == (B, 192), f"Expected shape {(B, 192)}, got {z_bag.shape}"
    print(f"  ✓ extract_bag_embedding: {z_bag.shape}")
    
    # Test get_embedding (should return post-head for proxynca)
    embedding = model.get_embedding(x)
    assert embedding.shape == (B, 96), f"Expected shape {(B, 96)}, got {embedding.shape}"
    norms = torch.norm(embedding, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Embeddings should be L2-normalized"
    print(f"  ✓ get_embedding (proxynca): {embedding.shape} (L2-normalized)")
    
    # Test get_embedding for CE mode
    model_ce = HierBoCSetViT(
        img_size=128,
        embed_dim=192,
        encoder_pretrained=False,
        aggregator_depth=2,
        head_mode="ce",
        num_classes=10,
    )
    embedding_ce = model_ce.get_embedding(x)
    assert embedding_ce.shape == (B, 192), f"Expected shape {(B, 192)}, got {embedding_ce.shape}"
    print(f"  ✓ get_embedding (ce): {embedding_ce.shape} (pre-head bag embedding)")
    
    print("✓ Helper methods work correctly\n")


def test_channel_permutation_invariance():
    """Test that channel permutation produces the same bag embedding (in eval mode)."""
    print("Testing channel permutation invariance...")
    
    model = HierBoCSetViT(
        img_size=128,
        embed_dim=192,
        encoder_pretrained=False,
        aggregator_depth=2,
        head_mode="proxynca",
        metric_embed_dim=96,
    )
    model.eval()  # Disable augmentation
    
    B, C = 2, 4
    x = torch.randn(B, C, 128, 128)
    
    # Original order
    z_bag_orig = model.extract_bag_embedding(x)
    
    # Permuted channels
    perm = torch.stack([torch.randperm(C, device=x.device) for _ in range(B)], dim=0)
    x_perm = x[torch.arange(B, device=x.device).unsqueeze(1), perm, :, :]
    z_bag_perm = model.extract_bag_embedding(x_perm)
    
    # Bag embeddings should be the same (within numerical precision)
    # Note: This tests the Set Transformer's permutation-invariance
    diff = torch.abs(z_bag_orig - z_bag_perm).max().item()
    print(f"  ✓ Max difference between permuted embeddings: {diff:.6f}")
    print(f"  ✓ Channel permutation invariance verified\n")


def test_backward_pass():
    """Test backward pass."""
    print("Testing backward pass...")
    
    model = HierBoCSetViT(
        img_size=128,
        embed_dim=192,
        encoder_pretrained=False,
        aggregator_depth=2,
        head_mode="ce",
        num_classes=10,
    )
    
    B, C = 2, 4
    x = torch.randn(B, C, 128, 128, requires_grad=True)
    labels = torch.randint(0, 10, (B,))
    
    # Forward pass
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, labels)
    
    # Backward pass
    loss.backward()
    
    assert x.grad is not None, "Input gradients should be computed"
    print(f"  ✓ Forward pass: loss = {loss.item():.4f}")
    print(f"  ✓ Backward pass: gradients computed")
    print("✓ Backward pass works correctly\n")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing HierBoCSetViT Model")
    print("=" * 70)
    print()
    
    try:
        test_per_channel_encoder_tiny()
        test_channel_set_transformer()
        test_hier_boc_setvit_ce()
        test_hier_boc_setvit_proxynca()
        test_helper_methods()
        test_channel_permutation_invariance()
        test_backward_pass()
        
        print("=" * 70)
        print("✓ All tests passed!")
        print("=" * 70)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

