# Bag-of-Channels Vision Transformer (BoC-ViT)

Supervised Bag-of-Channels ViT implementation for CHAMMI, inspired by:
- "SCALING CHANNEL-INVARIANT SELF-SUPERVISED LEARNING FOR MICROSCOPY IMAGES" (DINO Bag-of-Channels, ICLR 2025)
- ChannelAdaptive-DINO Bag-of-Channels approach

## Architecture

### Key Components

1. **PerChannelEncoder**: Shared ViT-Tiny encoder that processes each channel independently
   - No channel embeddings/IDs (channels treated as unordered set)
   - Input: `(B, C, 128, 128)` → Output: `(B, C, embed_dim)`

2. **BagAggregator**: Permutation-invariant aggregation over channel embeddings
   - **Mean mode**: DeepSets-style mean pooling + optional MLP
   - **Attention mode**: Learnable query attention over channels (returns attention weights for visualization)

3. **Supervised Head**: 
   - **CE mode**: Cross-entropy classifier
   - **ProxyNCA mode**: Metric learning with learnable proxies

## Quick Start

### 1. Test the Model

```bash
python test_boc_vit.py
```

### 2. Train a Model

```bash
python -m training.train_boc_supervised \
    --csv-file /path/to/CHAMMI/combined_metadata.csv \
    --root-dir /path/to/CHAMMI/ \
    --batch-size 32 \
    --epochs 50 \
    --lr 1e-4 \
    --aggregator-mode mean \
    --head-mode ce \
    --num-classes 10 \
    --use-amp
```

### 3. Key Arguments

**Model Architecture:**
- `--embed-dim`: Embedding dimension (default: 192)
- `--depth`: Number of transformer blocks (default: 6)
- `--num-heads`: Number of attention heads (default: 3)
- `--aggregator-mode`: `mean` or `attn` (default: `mean`)
- `--head-mode`: `ce` or `proxynca` (default: `ce`)

**Training:**
- `--batch-size`: Batch size per channel group (default: 32)
- `--lr`: Learning rate (default: 1e-4)
- `--weight-decay`: Weight decay (default: 0.01)
- `--warmup-epochs`: Warmup epochs (default: 5)
- `--grad-accum-steps`: Gradient accumulation steps (default: 1)
- `--use-amp`: Enable mixed precision training

**Loss (for ProxyNCA):**
- `--temperature`: Temperature parameter (default: 0.05)

## Model Usage

### Basic Usage

```python
from models.boc_vit import BoCViT
import torch

# Create model
model = BoCViT(
    img_size=128,
    patch_size=16,
    embed_dim=192,
    depth=6,
    num_heads=3,
    aggregator_mode="mean",  # or "attn"
    head_mode="ce",  # or "proxynca"
    num_classes=10,
)

# Forward pass
x = torch.randn(2, 4, 128, 128)  # (B, C, H, W)
logits = model(x)  # (B, num_classes)
```

### Attention Visualization

```python
# Use attention aggregator
model = BoCViT(
    aggregator_mode="attn",
    head_mode="ce",
    num_classes=10,
)

# Get attention weights
logits, attn_weights = model(x, return_attn=True)
# attn_weights: (B, C) attention weights over channels
```

### Metric Learning

```python
from losses.metric_losses import ProxyNCA

# Create model with metric learning head
model = BoCViT(
    head_mode="proxynca",
    num_classes=10,
    metric_embed_dim=256,
)

# Create loss
criterion = ProxyNCA(
    embed_dim=256,
    num_classes=10,
    temperature=0.05,
)

# Training
embeddings = model(x)  # (B, 256) L2-normalized
loss = criterion(embeddings, labels)
```

## Evaluation

```python
from training.eval_boc import evaluate_model, visualize_channel_attention

# Evaluate across all channel groups
results = evaluate_model(
    model=model,
    train_loaders=train_loaders,
    test_loaders=test_loaders,
    device=device,
)

# Visualize channel attention
visualize_channel_attention(
    model=model,
    images=images,
    metadatas=metadatas,
    device=device,
    save_path="attention_vis.png",
)
```

## Model Details

### PerChannelEncoder

- **Input**: `(B, C, 128, 128)` multi-channel images
- **Process**: 
  1. Reshape to `(B*C, 1, 128, 128)` (treat each channel as independent grayscale image)
  2. Patch embedding: `(B*C, 1, 128, 128)` → `(B*C, 64, 192)` (64 patches per channel)
  3. Transformer encoding: `(B*C, 64, 192)` → `(B*C, 192)` (CLS token or mean pooling)
  4. Reshape: `(B*C, 192)` → `(B, C, 192)`
- **Output**: `(B, C, embed_dim)` per-channel embeddings

### BagAggregator

**Mean Mode:**
- Average over channels: `(B, C, D)` → `(B, D)`
- Optional MLP: `MLP(mean(z_channels))`

**Attention Mode:**
- Learnable query vector `q ∈ ℝ^D`
- Attention scores: `scores = softmax(z_channels @ q)`
- Weighted sum: `z_bag = Σ_i attn_i * z_i`
- Returns: `(z_bag, attn_weights)` for visualization

### Permutation Invariance

Both aggregation modes are **permutation-invariant**:
- Mean: Average is invariant to order
- Attention: Attention over channels is invariant to order (no positional encoding on channels)

## Training Details

### Data Loading

The training script uses `create_dataset_ordered_training_iterator` which:
- Creates one DataLoader per dataset (Allen, HPA, CP)
- Randomly interleaves batches from different datasets
- Prevents model from learning fixed dataset ordering

### Mixed Precision

- Uses `torch.amp.autocast` for MPS/CPU compatibility
- Gradient scaling for stability
- Enabled by default with `--use-amp`

### Learning Rate Schedule

- Cosine annealing with warmup
- Warmup: linear increase for first `warmup_epochs`
- After warmup: cosine decay

## File Structure

```
models/
  └── boc_vit.py          # PerChannelEncoder, BagAggregator, BoCViT

losses/
  └── metric_losses.py    # ProxyNCA loss

training/
  ├── train_boc_supervised.py  # Training script
  └── eval_boc.py              # Evaluation utilities

test_boc_vit.py           # Test suite
```

## Citation

If you use this implementation, please cite:

1. The original DINO Bag-of-Channels paper (ICLR 2025)
2. The CHAMMI benchmark paper

## License

See main LICENSE file.

