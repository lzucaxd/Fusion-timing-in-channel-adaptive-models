# HierBoCSetViT: Hierarchical Bag-of-Channels Set Transformer Vision Transformer

## Overview

The `HierBoCSetViT` model implements a hierarchical Bag-of-Channels architecture for multi-channel microscopy images with variable channel counts (e.g., CHAMMI dataset with C ∈ {3,4,5}).

## Key Features

### 1. **Hierarchical Architecture**
   - **Level 1 (Patch-level)**: Pretrained timm ViT-Tiny processes each channel independently with patch-level self-attention
   - **Level 2 (Channel-level)**: Set Transformer processes channel embeddings as an unordered set with permutation-equivariant self-attention + PMA pooling

### 2. **Set-like Processing**
   - No positional encodings over channels (channels are treated as an unordered set)
   - Permutation-equivariant intermediate layers
   - Permutation-invariant bag embedding via Pooling-by-Multihead-Attention (PMA)

### 3. **Robustness Augmentation**
   - Random channel permutation during training
   - Random channel dropout (with probability `channel_dropout_p`)
   - Enforces robustness to dropped or reordered channels

### 4. **Training Modes**
   - **CE mode**: Standard cross-entropy classification
   - **ProxyNCA mode**: Metric learning with ProxyNCA++ loss

## Architecture Components

### `PerChannelEncoderTiny`
- Wraps pretrained timm ViT-Tiny (`vit_tiny_patch16_224`)
- Adapts 3-channel patch embedding to 1-channel by averaging RGB weights
- Processes each channel independently: `(B, C, H, W) → (B, C, embed_dim)`
- Auto-detects `embed_dim=192` from timm model

### `ChannelSetTransformer`
- Set Transformer over channel embeddings
- Stack of permutation-equivariant self-attention blocks
- PMA pooling with learned bag query → permutation-invariant bag embedding
- Outputs: `(B, C, embed_dim) → (B, embed_dim)`

### `HierBoCSetViT`
- Combines encoder + aggregator + head
- Supports both CE and ProxyNCA++ heads
- Optional attention weight visualization

## Usage

### Basic Usage (CE Mode)

```python
from models.hier_boc_setvit import HierBoCSetViT

model = HierBoCSetViT(
    img_size=128,
    embed_dim=192,  # Auto-detected from timm ViT-Tiny if None
    encoder_pretrained=True,
    aggregator_depth=2,
    aggregator_num_heads=3,
    head_mode="ce",
    num_classes=10,
    channel_dropout_p=0.3,
)

# Forward pass
x = torch.randn(2, 4, 128, 128)  # (B, C, H, W)
logits = model(x)  # (B, num_classes)

# With attention weights
logits, attn_weights = model(x, return_attn=True)
# logits: (B, num_classes)
# attn_weights: (B, C) channel attention weights
```

### Metric Learning Mode (ProxyNCA++)

```python
model = HierBoCSetViT(
    img_size=128,
    embed_dim=192,
    encoder_pretrained=True,
    aggregator_depth=2,
    head_mode="proxynca",
    metric_embed_dim=96,  # Default: embed_dim // 2
    channel_dropout_p=0.3,
)

# Forward pass
x = torch.randn(2, 4, 128, 128)
embedding = model(x)  # (B, metric_embed_dim) L2-normalized
```

### Helper Methods

```python
# Extract per-channel embeddings (before aggregation)
z_channels = model.extract_channel_embeddings(x)  # (B, C, embed_dim)

# Extract bag embedding (after aggregation, before head)
z_bag = model.extract_bag_embedding(x)  # (B, embed_dim)

# Get embedding space optimized by training loss
# - For proxynca: post-head, L2-normalized embedding
# - For ce: pre-head bag embedding
embedding = model.get_embedding(x)  # (B, D)
```

## Model Specifications

### ViT-Tiny Scale (Memory Efficient)
- **embed_dim**: 192 (from timm ViT-Tiny)
- **num_heads**: 3
- **depth**: 12 (ViT-Tiny default)
- **patch_size**: 16
- **Channel Set Transformer depth**: 2 (configurable)

### Input/Output Shapes

| Component | Input | Output |
|-----------|-------|--------|
| `PerChannelEncoderTiny` | `(B, C, 128, 128)` | `(B, C, 192)` |
| `ChannelSetTransformer` | `(B, C, 192)` | `(B, 192)` |
| `HierBoCSetViT` (CE) | `(B, C, 128, 128)` | `(B, num_classes)` |
| `HierBoCSetViT` (ProxyNCA) | `(B, C, 128, 128)` | `(B, metric_embed_dim)` |

## Key Design Principles

1. **Permutation-Invariance**: Channels are treated as an unordered set, making the model robust to channel reordering
2. **Hierarchical Attention**: 
   - Patch-level attention within each channel (spatial structure)
   - Channel-level attention across channels (channel relationships)
3. **Robustness**: Channel permutation and dropout during training enforce robustness to missing or reordered channels
4. **Efficiency**: ViT-Tiny scale keeps memory usage low for laptop training

## Integration with Training Loop

The model is compatible with existing training loops:

```python
# Training
model.train()
for images, metadata, labels in dataloader:
    # images: (B, C, 128, 128)
    logits = model(images)  # or embedding for ProxyNCA
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

# Evaluation
model.eval()
with torch.no_grad():
    embedding = model.get_embedding(images)
    # Use for k-NN evaluation or UMAP visualization
```

## Requirements

- `timm` (for pretrained ViT-Tiny)
- `torch >= 2.0.0`
- See `setup.py` for full dependencies

## Testing

Run the test script to verify the model:

```bash
python test_hier_boc_setvit.py
```

**Note**: Requires `timm` to be installed. If not installed:
```bash
pip install timm
```

## Differences from BoC-ViT

1. **Pretrained Backbone**: Uses pretrained timm ViT-Tiny instead of training from scratch
2. **Set Transformer Aggregation**: Uses Set Transformer (permutation-equivariant blocks + PMA) instead of simple mean/attention pooling
3. **Channel Augmentation**: Built-in channel permutation and dropout for robustness
4. **Hierarchical Design**: Explicit two-level hierarchy (patch-level → channel-level)

