# Fusion Timing in Channel-Adaptive Models for CHAMMI

This repository implements a complete dataset pipeline and training framework for investigating channel-adaptive vision transformers on the **CHAMMI (Channel-Adaptive Models in Microscopy Imaging)** benchmark dataset.

## Overview

CHAMMI consists of three fluorescence microscopy sub-datasets:
- **Allen/WTC-11**: 65,103 images, 3 channels, .ome.tiff format
- **HPA** (Human Protein Atlas): 66,936 images, 4 channels, .png format  
- **CP** (Cell Painting): 88,245 images, 5 channels, .png format

**Project Goal**: Determine where fusion should happen in channel-adaptive vision transformers:
- **Early Fusion**: channels mixed immediately
- **Late Fusion**: per-channel encoders and fusion only at the end
- **Hybrid Fusion**: channels separate initially, then fused mid-network

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

### Training Setup (Main Approach)

**Recommended: One DataLoader per Dataset with Frequent Shuffling**

Each dataset (Allen, HPA, CP) has its natural channel configuration:
- **Allen**: 3 channels (~31k samples)
- **HPA**: 4 channels (~33k samples)
- **CP**: 5 channels (~36k samples)

```python
from channel_adaptive_pipeline.chammi_grouped_dataloader import create_dataset_specific_dataloaders

# Create one DataLoader per dataset (Allen, HPA, CP)
dataloaders = create_dataset_specific_dataloaders(
    csv_file="path/to/CHAMMI/combined_metadata.csv",
    root_dir="path/to/CHAMMI/",
    batch_size=32,
    shuffle=True,  # Shuffles every epoch for variety
    target_labels='Label',  # From enriched_meta.csv
    split='train',  # ~100k training samples
    augment=True,  # RandomResizedCrop, flips (no color jitter)
    normalize=True,  # Uses CHAMMI-specific per-channel stats
)

# Training loop - process each dataset separately
for epoch in range(num_epochs):
    # Process each dataset
    for dataset_name in ['Allen', 'HPA', 'CP']:
        dataloader = dataloaders[dataset_name]
        
        # Each dataset has its natural channel configuration
        channel_count = 3 if dataset_name == 'Allen' else 4 if dataset_name == 'HPA' else 5
        
        for batch_images, batch_metadatas, batch_labels in dataloader:
            # batch_images shape: (batch_size, channel_count, 128, 128)
            # All samples from same dataset with same channels → efficient batching!
            
            # Your channel-adaptive ViT forward pass
            outputs = model(batch_images, num_channels=channel_count)
            
            # Loss (supervised classification or ProxyNCA++)
            loss = criterion(outputs, batch_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
```

### Alternative: Interleaved with Shuffled Dataset Order

For better generalization, interleave datasets with shuffled order each epoch:

```python
from channel_adaptive_pipeline.chammi_grouped_dataloader import create_interleaved_dataset_dataloader

# Interleaves datasets in shuffled order each epoch
interleaved = create_interleaved_dataset_dataloader(
    csv_file="path/to/CHAMMI/combined_metadata.csv",
    root_dir="path/to/CHAMMI/",
    batch_size=32,
    split='train',
    augment=True,
    normalize=True,
    shuffle_dataset_order=True,  # Different dataset order each epoch
)

for epoch in range(num_epochs):
    for batch_images, batch_metadatas, batch_labels, dataset_source in interleaved:
        # Each batch from one dataset with consistent channels
        # Dataset order changes each epoch (HPA→Allen→CP, then Allen→CP→HPA, etc.)
        channel_count = 3 if dataset_source == 'Allen' else 4 if dataset_source == 'HPA' else 5
        outputs = model(batch_images, num_channels=channel_count)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

## Key Features

### Dataset Handling
- ✅ **Unified Dataset Class**: Handles all three sub-datasets (Allen, HPA, CP)
- ✅ **Variable Channel Support**: 3, 4, 5 channels without padding
- ✅ **Automatic Channel Folding**: Converts tape format `(H, W*C)` to `(C, H, W)`
- ✅ **Label Extraction**: Automatically loads labels from enriched metadata
- ✅ **128×128 Images**: All images resized to match project requirements

### Efficient Batching (Main Approach)
- ✅ **Dataset-Specific DataLoaders**: One DataLoader per dataset (Allen, HPA, CP) with natural channel configuration
  - Allen: 3 channels, ~31k samples
  - HPA: 4 channels, ~33k samples
  - CP: 5 channels, ~36k samples
- ✅ **Frequent Shuffling**: Shuffles every epoch for variety (enabled by default)
- ✅ **True Batching**: All samples in batch have same channels → GPU-optimized processing
- ✅ **No Padding**: Preserves actual channel counts for channel-adaptive models
- ✅ **Interleaved Option**: Can shuffle dataset order each epoch for better generalization

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

### Batch Shapes

**Dataset-Specific DataLoaders (Main Approach):**
```python
# Each dataset has its natural channel configuration
dataloaders = create_dataset_specific_dataloaders(...)

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

## Evaluation

Models will be evaluated on:
- **SD vs OOD accuracy and macro-F1**: In-distribution vs out-of-distribution performance
- **Channel robustness**: Dropping or shuffling channels
- **Channel importance**: Which channels are most influential for each task

This allows identification of which fusion strategy (Early/Late/Hybrid) is most robust and which biological signals each task depends on.

## Model Architecture

Train channel-adaptive ViT models at ViT-Tiny/Small capacity with different fusion strategies:

1. **Early Fusion**: Channels mixed immediately → standard ViT with variable input channels
2. **Late Fusion**: Per-channel encoders, fusion at end → separate encoders per channel
3. **Hybrid Fusion**: Channels separate initially, fused mid-network → flexible fusion point

## Loss Functions

- **Supervised Classification**: Standard cross-entropy loss
- **ProxyNCA++**: Supervised metric learning loss (used in CHAMMI benchmark)

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

## Documentation

- **Quick Start**: See `FOR_TEAMMATES.md`
- **Full Documentation**: See `CHAMMI_IMPLEMENTATION_SUMMARY.md`
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

**Status**: ✅ Dataset pipeline complete and ready for channel-adaptive ViT training
