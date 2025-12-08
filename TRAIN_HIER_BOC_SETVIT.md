# Training HierBoCSetViT

## Quick Start

### Basic Training Command (ProxyNCA mode, 50 epochs)

```bash
python training/train_hier_boc_setvit.py \
    --csv-file /Users/zamfiraluca/Downloads/CHAMMI/combined_metadata.csv \
    --root-dir /Users/zamfiraluca/Downloads/CHAMMI \
    --batch-size 32 \
    --epochs 50 \
    --head-mode proxynca \
    --output-dir ./checkpoints/hier_boc_setvit_proxynca
```

### Training with Cross-Entropy (CE mode)

```bash
python training/train_hier_boc_setvit.py \
    --csv-file /Users/zamfiraluca/Downloads/CHAMMI/combined_metadata.csv \
    --root-dir /Users/zamfiraluca/Downloads/CHAMMI \
    --batch-size 32 \
    --epochs 50 \
    --head-mode ce \
    --output-dir ./checkpoints/hier_boc_setvit_ce
```

### Custom Training Parameters

```bash
python training/train_hier_boc_setvit.py \
    --csv-file /Users/zamfiraluca/Downloads/CHAMMI/combined_metadata.csv \
    --root-dir /Users/zamfiraluca/Downloads/CHAMMI \
    --batch-size 16 \
    --epochs 100 \
    --lr 5e-5 \
    --weight-decay 0.01 \
    --warmup-epochs 10 \
    --head-mode proxynca \
    --metric-embed-dim 128 \
    --aggregator-depth 3 \
    --channel-dropout-p 0.3 \
    --output-dir ./checkpoints/hier_boc_setvit_custom \
    --num-workers 4
```

### Resume Training from Checkpoint

```bash
python training/train_hier_boc_setvit.py \
    --csv-file /Users/zamfiraluca/Downloads/CHAMMI/combined_metadata.csv \
    --root-dir /Users/zamfiraluca/Downloads/CHAMMI \
    --resume ./checkpoints/hier_boc_setvit_proxynca/checkpoint_latest.pth \
    --output-dir ./checkpoints/hier_boc_setvit_proxynca
```

## Key Arguments

### Model Arguments
- `--embed-dim`: Embedding dimension (default: 192, auto-detected from timm ViT-Tiny)
- `--encoder-pretrained`: Use pretrained timm ViT-Tiny (default: True)
- `--aggregator-depth`: Depth of Set Transformer aggregator (default: 2)
- `--aggregator-num-heads`: Number of heads in Set Transformer (default: 3)
- `--head-mode`: "ce" or "proxynca" (default: "proxynca")
- `--metric-embed-dim`: Embedding dimension for ProxyNCA (default: embed_dim // 2)
- `--channel-dropout-p`: Channel dropout probability during training (default: 0.3)

### Training Arguments
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 1e-4)
- `--weight-decay`: Weight decay (default: 0.01)
- `--warmup-epochs`: Warmup epochs (default: 5)
- `--temperature`: Temperature for ProxyNCA loss (default: 0.05)
- `--batch-size`: Batch size (default: 32)

### Data Arguments
- `--csv-file`: Path to combined_metadata.csv (required)
- `--root-dir`: Path to CHAMMI root directory (required)
- `--target-labels`: Label column name (default: "Label")
- `--num-workers`: Number of data loading workers (default: 4)

## Output

Checkpoints are saved to the specified `--output-dir`:
- `checkpoint_epoch_{N}.pth`: Checkpoint for each epoch
- `checkpoint_latest.pth`: Latest checkpoint (updated each epoch)

Each checkpoint contains:
- Model state dict
- Optimizer state dict
- Scheduler state dict
- Training loss
- Epoch number
- Model configuration

## Training Features

- **Random Batch Interleaving**: Batches are randomly sampled from all datasets (Allen, HPA, CP)
- **Channel Permutation & Dropout**: Applied during training for robustness (controlled by `--channel-dropout-p`)
- **Mixed Precision**: Handled automatically by MPS device
- **Label Encoding**: Automatically discovers all unique labels across datasets
- **Warmup + Cosine Annealing**: LR schedule with warmup

## Monitoring Training

Training progress is shown with a progress bar showing:
- Current loss
- Dataset source for each batch
- Average loss
- Learning rate

