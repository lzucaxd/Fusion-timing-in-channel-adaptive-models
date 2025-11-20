# CHAMMI Dataset Implementation Summary

## Overview

This implementation provides a complete dataset loader and data processing pipeline for the **CHAMMI (Channel-Adaptive Models in Microscopy Imaging)** benchmark dataset. CHAMMI consists of three fluorescence microscopy sub-datasets (WTC-11/Allen, HPA, Cell Painting) with variable channel configurations (3, 4, 5 channels) designed to evaluate channel-adaptive vision models.

## What Has Been Implemented

### 1. Core Dataset Infrastructure

#### **CHAMMIDataset** (`channel_adaptive_pipeline/chammi_dataset.py`)
- Unified dataset class handling all three sub-datasets (Allen, HPA, CP)
- Automatic channel folding from tape format `(H, W*C)` to tensor format `(C, H, W)`
- Variable channel support (3, 4, or 5 channels) without padding
- Label extraction from enriched metadata files
- Train/test split filtering
- Flexible transform pipeline

**Key Features:**
- Handles different file formats (.ome.tiff, .png)
- Different directory structures (flat vs nested)
- Automatic metadata loading from enriched CSV files
- Returns `(image, metadata, label)` tuples with rich metadata

#### **CHAMMITransform** (`channel_adaptive_pipeline/chammi_dataset.py`)
- Resizes all images to 128×128 (configurable)
- Augmentations: RandomResizedCrop, horizontal/vertical flips
- **No color/brightness jitter** (preserves fluorescence signal relationships)
- Per-channel normalization using CHAMMI-specific statistics (not ImageNet)
- Handles variable channels (3, 4, 5) with channel-appropriate normalization

**Normalization Stats (computed from 5k train samples):**
- 3 channels: mean=[0.1107, 0.1345, 0.0425], std=[0.2593, 0.2815, 0.1218]
- 4 channels: mean=[0.0827, 0.0407, 0.0642, 0.0848], std=[0.1527, 0.0963, 0.1742, 0.1552]
- 5 channels: mean=[0.0998, 0.1934, 0.1625, 0.1810, 0.1479], std=[0.1718, 0.1664, 0.1510, 0.1466, 0.1501]

### 2. Efficient Batching Solutions

#### **Standard DataLoader** (`channel_adaptive_pipeline/chammi_dataset.py`)
- Custom collate function with three modes:
  - `'auto'`: Stacks if same channels, returns list if mixed (default)
  - `'list'`: Always returns list of tensors
  - `'pad'`: Pads to max channels in batch (not recommended)

#### **Grouped DataLoaders** (`channel_adaptive_pipeline/chammi_grouped_dataloader.py`) ⭐ **Recommended for Training**
- **Separate DataLoaders** for each channel count (3, 4, 5)
- Each batch has consistent channels → can stack efficiently
- True batching with GPU-optimized processing
- Two options:
  1. `create_grouped_chammi_dataloaders()`: Returns dict of DataLoaders by channel count
  2. `create_interleaved_chammi_dataloader()`: Interleaves batches from all channel counts

**Why Grouped?**
- Mixed batches (3, 4, 5 channels together) cannot be stacked → must process one-by-one (inefficient)
- Grouped batches (all same channels) can be stacked → process entire batch at once (efficient)

### 3. Utility Scripts

#### **compute_dataset_stats.py** (`channel_adaptive_pipeline/compute_dataset_stats.py`)
- Computes per-channel mean/std statistics for normalization
- Groups by channel count (3, 4, 5)
- Can be run to recompute statistics with more samples

#### **Bug Fix in folded_dataset.py**
- Fixed bug in `fold_channels()` function (line 40 - incorrect mask indexing)

### 4. Dataset Statistics

**Original Image Dimensions (before resize):**
- **Allen**: `(238, 374, 3)` - Height 238, width 374 per channel, 3 channels
- **HPA**: `(512, 512, 4)` - Height 512, width 512 per channel, 4 channels
- **CP**: `(160, 160, 5)` - Height 160, width 160 per channel, 5 channels

**After Transform (resize to 128×128):**
- All datasets: `(C, 128, 128)` where C is 3, 4, or 5

**Train Split Distribution:**
- 3 channels (Allen): ~31,060 samples
- 4 channels (HPA): ~32,725 samples
- 5 channels (CP): ~36,360 samples
- **Total**: ~100,145 training samples

## Usage Examples

### Basic Usage

```python
from channel_adaptive_pipeline.chammi_dataset import CHAMMIDataset, CHAMMITransform

# Create dataset
transform = CHAMMITransform(size=128, augment=False, normalize=True)
dataset = CHAMMIDataset(
    csv_file="path/to/combined_metadata.csv",
    root_dir="path/to/CHAMMI/",
    target_labels='Label',  # Extract from enriched_meta.csv
    transform=transform,
    split='train',
)

# Access samples
image, metadata, label = dataset[0]
# image: (C, 128, 128) tensor where C is 3, 4, or 5
# metadata: dict with 'num_channels', 'dataset_source', etc.
# label: extracted label from enriched metadata
```

### Efficient Training (Recommended)

```python
from channel_adaptive_pipeline.chammi_grouped_dataloader import create_grouped_chammi_dataloaders

# Create separate DataLoaders for each channel count
dataloaders = create_grouped_chammi_dataloaders(
    csv_file="path/to/combined_metadata.csv",
    root_dir="path/to/CHAMMI/",
    batch_size=32,
    shuffle=True,
    target_labels='Label',
    split='train',
    augment=True,  # RandomResizedCrop, flips (no color jitter)
    normalize=True,  # Uses CHAMMI-specific stats
)

# Training loop - process each channel count separately
for channel_count in [3, 4, 5]:
    dataloader = dataloaders[channel_count]
    for batch_images, batch_metadatas, batch_labels in dataloader:
        # batch_images shape: (batch_size, channel_count, 128, 128)
        # All samples have SAME channels → efficient batching!
        
        outputs = model(batch_images, num_channels=channel_count)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### Interleaved Training (Alternative)

```python
from channel_adaptive_pipeline.chammi_grouped_dataloader import create_interleaved_chammi_dataloader

# Interleave batches from all channel counts
interleaved = create_interleaved_chammi_dataloader(
    csv_file="path/to/combined_metadata.csv",
    root_dir="path/to/CHAMMI/",
    batch_size_per_channel=32,
    split='train',
    augment=True,
    normalize=True,
)

for batch_images, batch_metadatas, batch_labels, channel_count in interleaved:
    # Each batch has consistent channel count
    outputs = model(batch_images, num_channels=channel_count)
    loss = criterion(outputs, batch_labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## Files Structure

```
MorphEm/
├── channel_adaptive_pipeline/
│   ├── chammi_dataset.py              # Core dataset and transforms
│   ├── chammi_grouped_dataloader.py   # Efficient grouped dataloaders
│   ├── compute_dataset_stats.py       # Utility to compute normalization stats
│   ├── folded_dataset.py              # Channel folding utilities (fixed)
│   └── CHAMMI_DATASET_README.md       # Detailed API documentation
│
├── examples/
│   ├── demos/
│   │   ├── demo_dataloader_no_padding.py     # Demo of list-based batches
│   │   └── demo_efficient_batching.py        # Demo of grouped dataloaders
│   └── tests/
│       └── test_chammi_dataset.py            # Test script
│
└── CHAMMI_IMPLEMENTATION_SUMMARY.md   # This file
```

## Key Design Decisions

1. **No Padding by Default**: Preserves actual channel counts for channel-adaptive models
2. **Grouped Batching**: Separate DataLoaders per channel count for efficiency
3. **Dataset-Specific Normalization**: Computed from CHAMMI data, not ImageNet
4. **No Color Jitter**: Preserves fluorescence signal relationships
5. **Flexible Metadata**: Returns rich metadata for channel-adaptive processing

## Testing

Run the test script to verify everything works:

```bash
cd examples/tests
python test_chammi_dataset.py
```

This will:
- Test dataset loading for all three datasets
- Test label extraction
- Visualize samples from all datasets
- Test DataLoader functionality
- Save visualizations to `./chammi_visualizations/`

## Next Steps for Training

1. **Dataset is ready** - Use grouped dataloaders for efficient training
2. **Normalization configured** - Per-channel stats computed and applied
3. **Augmentations set** - Spatial augmentations only (no color jitter)
4. **Ready for channel-adaptive ViT models**:
   - Early Fusion: channels mixed immediately
   - Late Fusion: per-channel encoders, fusion at end
   - Hybrid Fusion: channels separate initially, fused mid-network

## Notes

- All images resized to 128×128 as specified in project requirements
- Train split: ~100k samples (31k + 33k + 36k)
- Supports both supervised classification and ProxyNCA++ loss
- Metadata includes channel counts, dataset source, cell types, etc.
- Compatible with ViT-Tiny/Small architectures

## Compatibility

- Python 3.8+
- PyTorch 1.8+
- skimage, pandas, numpy
- torchvision

