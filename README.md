# CHAMMI Dataset Pipeline

This repository provides a complete dataset pipeline for the **CHAMMI (Channel-Adaptive Models in Microscopy Imaging)** benchmark dataset. The implementation handles all three sub-datasets with variable channel configurations and provides efficient batching for training any channel-adaptive model.

## Overview

CHAMMI consists of three fluorescence microscopy sub-datasets:
- **Allen/WTC-11**: 65,103 images, 3 channels, .ome.tiff format
- **HPA** (Human Protein Atlas): 66,936 images, 4 channels, .png format  
- **CP** (Cell Painting): 88,245 images, 5 channels, .png format

## Quick Start

### Installation

```bash
git clone https://github.com/lzucaxd/Fusion-timing-in-channel-adaptive-models.git
cd Fusion-timing-in-channel-adaptive-models
pip install -e .
```

### Dataset Setup

1. Download CHAMMI dataset from [Zenodo](https://zenodo.org/record/7988357)
2. Extract to a directory (e.g., `/path/to/CHAMMI/`)
3. Structure should be:
   ```
   CHAMMI/
     ├── combined_metadata.csv
     ├── Allen/
     │   ├── enriched_meta.csv
     │   └── crops/
     ├── HPA/
     │   ├── enriched_meta.csv
     │   └── crops/
     └── CP/
         ├── enriched_meta.csv
         └── crops/
   ```

## Key Features

### Dataset Handling
- ✅ **Unified Dataset Class**: Handles all three sub-datasets (Allen, HPA, CP)
- ✅ **Variable Channel Support**: 3, 4, 5 channels without padding
- ✅ **Automatic Channel Folding**: Converts tape format `(H, W*C)` to `(C, H, W)`
- ✅ **Label Extraction**: Automatically loads labels from enriched metadata
- ✅ **128×128 Images**: All images resized to match project requirements

### Efficient Batching
- ✅ **Dataset-Specific DataLoaders**: One DataLoader per dataset (Allen, HPA, CP) with natural channel configuration
  - Allen: 3 channels, ~31k samples
  - HPA: 4 channels, ~33k samples
  - CP: 5 channels, ~36k samples
- ✅ **Random Batch Interleaving**: Randomly sample which dataset to get next batch from
  - Prevents model from learning fixed dataset ordering
  - Makes model more robust to dataset sequence patterns
  - Example: CP → Allen → HPA → CP → Allen → ...
- ✅ **Frequent Shuffling**: Shuffles samples within each dataset every epoch (enabled by default)
- ✅ **True Batching**: All samples in batch have same channels → GPU-optimized processing
- ✅ **No Padding**: Preserves actual channel counts for channel-adaptive models

### Data Preprocessing
- ✅ **CHAMMI-Specific Normalization**: Per-channel stats computed from dataset (not ImageNet)
- ✅ **Spatial Augmentations**: RandomResizedCrop, horizontal/vertical flips
- ✅ **No Color Jitter**: Preserves fluorescence signal relationships

### Normalization Statistics

Per-channel normalization stats (computed from 5k train samples):
- **3 channels**: mean=[0.1107, 0.1345, 0.0425], std=[0.2593, 0.2815, 0.1218]
- **4 channels**: mean=[0.0827, 0.0407, 0.0642, 0.0848], std=[0.1527, 0.0963, 0.1742, 0.1552]
- **5 channels**: mean=[0.0998, 0.1934, 0.1625, 0.1810, 0.1479], std=[0.1718, 0.1664, 0.1510, 0.1466, 0.1501]

## Dataset Statistics

**Train Split (~100k samples):**
- 3 channels (Allen): ~31,060 samples
- 4 channels (HPA): ~32,725 samples
- 5 channels (CP): ~36,360 samples

**Image Dimensions:**
- Original: Allen `(238, 374, 3)`, HPA `(512, 512, 4)`, CP `(160, 160, 5)`
- After transform: All `(C, 128, 128)` where C is 3, 4, or 5

## Usage Examples

### Basic Dataset Usage

```python
from channel_adaptive_pipeline.chammi_dataset import CHAMMIDataset, CHAMMITransform

# Create dataset
transform = CHAMMITransform(size=128, augment=False, normalize=True)
dataset = CHAMMIDataset(
    csv_file="path/to/combined_metadata.csv",
    root_dir="path/to/CHAMMI/",
    target_labels='Label',
    transform=transform,
    split='train',
)

# Access samples
image, metadata, label = dataset[0]
# image: (C, 128, 128) tensor where C is 3, 4, or 5
# metadata: dict with 'num_channels', 'dataset_source', 'cell_type', etc.
# label: extracted label from enriched_meta.csv
```

### Recommended: Randomly Interleave Batches from All Datasets

**Main Training Approach**: Randomly sample which dataset to get the next batch from. This prevents the model from learning a fixed dataset ordering and makes it more robust.

```python
from channel_adaptive_pipeline.chammi_grouped_dataloader import create_dataset_ordered_training_iterator

# Training loop - batches randomly interleaved from all datasets
for epoch in range(num_epochs):
    # Create iterator for this epoch (will randomly interleave datasets)
    iterator = create_dataset_ordered_training_iterator(
        csv_file="path/to/CHAMMI/combined_metadata.csv",
        root_dir="path/to/CHAMMI/",
        batch_size=32,
        shuffle=True,  # Shuffles samples within each dataset
        target_labels='Label',  # From enriched_meta.csv
        split='train',  # ~100k training samples
        augment=True,  # RandomResizedCrop, flips (no color jitter)
        normalize=True,  # Uses CHAMMI-specific per-channel stats
        shuffle_dataset_order=True,  # Randomly samples which dataset per batch
    )
    
    # Process all batches (randomly interleaved from all datasets)
    for batch_images, batch_metadatas, batch_labels, dataset_source in iterator:
        # Each batch is from one dataset with consistent channels
        # Batches are randomly interleaved: CP → Allen → HPA → CP → ...
        
        channel_count = 3 if dataset_source == 'Allen' else 4 if dataset_source == 'HPA' else 5
        
        # Your model forward pass
        outputs = model(batch_images, num_channels=channel_count)
        
        # Loss computation
        loss = criterion(outputs, batch_labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Iterator automatically stops after one epoch
    # Next epoch will create a new iterator with different random interleaving
```

### Alternative: Separate DataLoaders per Dataset

```python
from channel_adaptive_pipeline.chammi_grouped_dataloader import create_dataset_specific_dataloaders

# Create one DataLoader per dataset
dataloaders = create_dataset_specific_dataloaders(
    csv_file="path/to/combined_metadata.csv",
    root_dir="path/to/CHAMMI/",
    batch_size=32,
    split='train',
    augment=True,
    normalize=True,
)

# Access individual dataset DataLoaders
allen_loader = dataloaders['Allen']  # 3 channels
hpa_loader = dataloaders['HPA']      # 4 channels
cp_loader = dataloaders['CP']         # 5 channels

# Use in training loop
for batch_images, batch_metadatas, batch_labels in allen_loader:
    # batch_images.shape = (32, 3, 128, 128)
    pass
```

### Batch Shapes

**Dataset-Specific DataLoaders:**
```python
# Each dataset has its natural channel configuration

# Allen DataLoader: all batches have 3 channels
batch_images.shape = (batch_size, 3, 128, 128)  # Allen

# HPA DataLoader: all batches have 4 channels
batch_images.shape = (batch_size, 4, 128, 128)  # HPA

# CP DataLoader: all batches have 5 channels
batch_images.shape = (batch_size, 5, 128, 128)  # CP

# Examples with batch_size=32:
#   Allen: (32, 3, 128, 128)
#   HPA:   (32, 4, 128, 128)
#   CP:    (32, 5, 128, 128)
```

## Testing

### Run Comprehensive Tests

```bash
# Test all dataloaders and normalization
python test_all_dataloaders.py

# Check normalization per batch
python check_normalization_per_batch.py

# Test dataset functionality
cd examples/tests
python test_chammi_dataset.py
```

## Project Structure

```
Fusion-timing-in-channel-adaptive-models/
├── channel_adaptive_pipeline/
│   ├── chammi_dataset.py              # Core dataset and transforms
│   ├── chammi_grouped_dataloader.py   # Efficient grouped dataloaders ⭐
│   ├── compute_dataset_stats.py       # Normalization stats utility
│   ├── folded_dataset.py              # Channel folding utilities
│   └── ...
├── examples/
│   ├── demos/                         # Demo scripts
│   └── tests/                         # Test scripts
├── FOR_TEAMMATES.md                   # Quick start guide
├── CHAMMI_IMPLEMENTATION_SUMMARY.md   # Full documentation
└── README.md                          # This file
```

## Models

### HierBoCSetViT (Hierarchical Bag-of-Channels Set Transformer Vision Transformer)

A state-of-the-art channel-adaptive model for multi-channel microscopy images that treats channels as an unordered set.

**Key Features:**
- ✅ **Hierarchical Architecture**: Patch-level ViT encoder + Set Transformer channel aggregator
- ✅ **Permutation Invariant**: Robust to channel ordering
- ✅ **Multiple Embedding Modes**: CLS token, mean pooling, or attention pooling (default: `attn_pool`)
- ✅ **Optional Channel Gating**: Learnable gating mechanism for channel importance
- ✅ **Multi-Seed PMA**: Multiple bag queries for richer aggregation
- ✅ **Pretrained Encoders**: ViT-Tiny or ViT-Small from timm

**Training Features:**
- ✅ **ProxyNCA Optimization**: Separate parameter group for metric learning proxies
- ✅ **Two-Tier Learning Rates**: Different LRs for encoder vs. rest of model
- ✅ **Encoder Freeze/Unfreeze**: Optional schedule for gradual fine-tuning
- ✅ **Hard Label Encoding**: Robust filtering of invalid labels
- ✅ **Random Dataset Sampling**: Ensures all data is processed each epoch
- ✅ **Gradient Clipping**: Prevents exploding gradients
- ✅ **Reproducibility**: Random seed control

**Quick Start:**
```bash
# Train with default settings (attention pooling, multi-seed PMA)
python training/train_hier_boc_setvit.py \
    --csv-file path/to/CHAMMI/combined_metadata.csv \
    --root-dir path/to/CHAMMI/ \
    --encoder-type tiny \
    --channel-embed-mode attn_pool \
    --pma-num-seeds 4 \
    --encoder-lr-mult 0.2 \
    --epochs 20

# Evaluate trained model
python training/evaluate_hier_boc.py \
    --checkpoint checkpoints/hier_boc_setvit_tiny/best_model.pth \
    --csv-file path/to/CHAMMI/combined_metadata.csv \
    --root-dir path/to/CHAMMI/
```

**Model Architecture:**
- **Per-Channel Encoder**: Pretrained ViT-Tiny/Small processes each channel independently
- **Channel Embedding**: Three modes available:
  - `"cls"`: Uses CLS token (original)
  - `"mean_patches"`: Mean pooling over patch tokens
  - `"attn_pool"`: Attention pooling with learnable query (default, recommended)
- **Channel Aggregator**: Set Transformer with Pooling-by-Multihead-Attention (PMA)
- **Head**: Cross-entropy or ProxyNCA++ for metric learning

For detailed model documentation, see:
- `HIER_BOC_SETVIT_README.md` - Model architecture details
- `NEW_FEATURES_SUMMARY.md` - Complete list of new features
- `COMPREHENSIVE_RESULTS_SUMMARY.md` - Experimental results

## Documentation

- **Quick Start**: See `FOR_TEAMMATES.md`
- **Full Documentation**: See `CHAMMI_IMPLEMENTATION_SUMMARY.md`
- **Model Documentation**: See `HIER_BOC_SETVIT_README.md` and `NEW_FEATURES_SUMMARY.md`
- **Results**: See `COMPREHENSIVE_RESULTS_SUMMARY.md`
- **API Reference**: See `channel_adaptive_pipeline/CHAMMI_DATASET_README.md`
- **Test Results**: See `DATALOADER_TEST_RESULTS.md`

## Requirements

- Python 3.8+
- PyTorch 1.8+
- torchvision
- scikit-image, pandas, numpy
- See `requirements.txt` for full list

## Citation

If you use CHAMMI dataset, please cite:
```
@article{chen2023chammi,
  title={CHAMMI: A benchmark for channel-adaptive models in microscopy imaging},
  author={Chen, Zitong and Pham, Minh and others},
  journal={arXiv preprint arXiv:2310.19224},
  year={2023}
}
```

## License

See `LICENSE` file.

## Original CHAMMI Repository

The original CHAMMI benchmark code and data can be found at:
- Dataset: [Zenodo](https://zenodo.org/record/7988357)
- Code: [GitHub](https://github.com/chaudatascience/channel_adaptive_models)

---

**Status**: ✅ Dataset pipeline complete and ready for training any channel-adaptive model

---

## Recent Updates (December 2024)

### Model Enhancements
- **Attention Pooling**: New default mode for patch token aggregation with learnable query
- **Channel Gating**: Optional learnable gating mechanism for channel importance
- **Multi-Seed PMA**: Support for multiple bag queries (K queries) in channel aggregation
- **Performance**: Vectorized channel dropout for faster training

### Training Improvements
- **ProxyNCA Optimization**: Proxies now properly optimized as separate parameter group
- **Two-Tier Learning Rates**: Different learning rates for encoder vs. rest of model
- **Encoder Freeze/Unfreeze**: Optional schedule for gradual fine-tuning
- **Hard Label Encoding**: Robust filtering of invalid/unknown labels
- **Random Dataset Sampling**: Ensures all batches processed each epoch with random dataset selection
- **Gradient Clipping**: Prevents exploding gradients
- **Improved LR Scheduler**: Pure Python cosine decay with warmup
- **Reproducibility**: Random seed control for reproducible experiments

See `NEW_FEATURES_SUMMARY.md` for complete details.
