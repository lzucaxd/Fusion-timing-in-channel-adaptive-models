# CHAMMI Dataset Loader

This module provides a unified dataset class and DataLoader for the CHAMMI (Channel-Adaptive Models in Microscopy Imaging) benchmark dataset.

## Overview

CHAMMI consists of three fluorescence microscopy sub-datasets:
- **Allen/WTC-11**: 65,103 images, 3 channels, .ome.tiff format
- **HPA** (Human Protein Atlas): 66,936 images, 4 channels, .png format  
- **CP** (Cell Painting): 88,245 images, 5 channels, .png format

All images are single-cell crops with channels stored in "tape format" (flattened channels in width dimension).

## Key Features

1. **Unified interface**: Single dataset class handles all three sub-datasets
2. **Variable channel support**: Handles 3, 4, and 5 channel images without padding (models can use metadata)
3. **Automatic channel folding**: Converts tape format (H, W*C) to tensor format (C, H, W)
4. **Label extraction**: Automatically loads labels from enriched metadata files
5. **Flexible transforms**: Built-in transforms for resizing to 128×128 (or custom size)
6. **Train/test split filtering**: Easy filtering by dataset split

## Usage

### Basic Usage

```python
from channel_adaptive_pipeline.chammi_dataset import CHAMMIDataset, CHAMMITransform

# Create dataset
transform = CHAMMITransform(size=128, augment=False)
dataset = CHAMMIDataset(
    csv_file="path/to/combined_metadata.csv",
    root_dir="path/to/CHAMMI/",
    target_labels='Label',  # Extract Label column from enriched_meta.csv
    transform=transform,
    split='train',  # or 'test', or None for all
)

# Access samples
image, metadata, label = dataset[0]
# image: torch.Tensor of shape (C, 128, 128) where C is num_channels (3, 4, or 5)
# metadata: dict with 'num_channels', 'dataset_source', 'cell_type', etc.
# label: extracted label from enriched_meta.csv
```

### Using DataLoader

```python
from channel_adaptive_pipeline.chammi_dataset import create_chammi_dataloader

dataloader = create_chammi_dataloader(
    csv_file="path/to/combined_metadata.csv",
    root_dir="path/to/CHAMMI/",
    batch_size=32,
    shuffle=True,
    target_labels='Label',
    split='train',
    resize_to=128,
    augment=True,  # Use RandomResizedCrop for training
    num_workers=4,
)

# Iterate over batches
for batch_images, batch_metadatas, batch_labels in dataloader:
    # batch_images: (B, max_channels, 128, 128) - padded to max channels in batch
    # batch_metadatas: list of dicts with metadata for each sample
    # batch_labels: list of labels
    
    # Access channel count for each sample
    for i, meta in enumerate(batch_metadatas):
        num_channels = meta['num_channels']  # 3, 4, or 5
        dataset_source = meta['dataset_source']  # 'Allen', 'HPA', or 'CP'
        
        # Use only actual channels (ignore padding)
        actual_image = batch_images[i, :num_channels, :, :]
        # ... use in channel-adaptive model ...
```

### Dataset Return Format

Each sample returns a tuple `(image, metadata, label)`:

- **image**: `torch.Tensor` of shape `(num_channels, height, width)`
  - Channels vary: 3 (Allen), 4 (HPA), or 5 (CP)
  - Already transformed (resized to target size if transform provided)

- **metadata**: `dict` containing:
  - `'num_channels'`: int (3, 4, or 5)
  - `'dataset_source'`: str ('Allen', 'HPA', or 'CP')
  - `'cell_type'`: str (e.g., 'hiPSC', 'A-431', 'A549')
  - `'file_path'`: str (relative path to image file)
  - `'ID'`: str (unique identifier)
  - `'channel_width'`: int (width before channel folding)
  - `'channels_content'`: str (description of each channel)

- **label**: Extracted from enriched metadata files
  - `None` if `target_labels=None`
  - Single value if `target_labels='Label'` (or other column name)
  - Dict if `target_labels=['Label', 'cell_type', ...]`

### Label Columns by Dataset

Different enriched metadata files have different label columns:

- **Allen/enriched_meta.csv**: `Label` (cell cycle stage), `Structure` (subcellular localization)
- **HPA/enriched_meta.csv**: `Label` (subcellular protein localization), `cell_type`
- **CP/enriched_meta.csv**: `Label` (same as `Treatment`), `Treatment`, `source`

You can extract multiple columns:

```python
dataset = CHAMMIDataset(
    csv_file="...",
    root_dir="...",
    target_labels=['Label', 'cell_type'],  # Extract both
    ...
)

image, metadata, labels = dataset[0]
# labels: {'Label': 'golgi apparatus', 'cell_type': 'A-431'}
```

### Transforms

#### CHAMMITransform

Resizes images to target size with optional augmentation:

```python
# Centered resize (no augmentation)
transform = CHAMMITransform(size=128, augment=False, mode='center')

# Random crop and resize (for training)
transform = CHAMMITransform(size=128, augment=True, mode='random')
```

#### Custom Transforms

You can also use custom transforms:

```python
from torchvision import transforms

custom_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

dataset = CHAMMIDataset(..., transform=custom_transform)
```

## Testing and Visualization

Run the test script to verify the dataset works correctly:

```bash
python test_chammi_dataset.py
```

This will:
1. Test basic dataset functionality
2. Test label extraction
3. Visualize samples from all three datasets
4. Test DataLoader functionality
5. Save visualization images to `./chammi_visualizations/`

## Important Notes

### Channel Handling

The dataset preserves variable channel counts (3, 4, 5) without padding. This allows models to:
- Process channels adaptively
- Use metadata to determine actual channel count
- Handle variable channel architectures

The DataLoader's collate function pads images to the max channels in each batch for efficient batching. Models should use `metadata['num_channels']` to determine which channels to use.

### File Paths

All file paths in `combined_metadata.csv` are relative to the CHAMMI root directory:
- `Allen/crops/filename.ome.tiff`
- `HPA/crops/filename.png`
- `CP/crops/LINCS/plate/well/site/filename.png`

Ensure `root_dir` points to the parent directory containing these subdirectories.

### Memory Considerations

The dataset loads images on-demand (not cached in memory). For faster iteration, you can:
- Increase `num_workers` in DataLoader
- Pre-process images to a common format
- Use a faster storage backend (SSD, NVMe)

## Example: Training Loop

```python
from channel_adaptive_pipeline.chammi_dataset import create_chammi_dataloader
import torch
import torch.nn as nn

# Create dataloaders
train_loader = create_chammi_dataloader(
    csv_file="combined_metadata.csv",
    root_dir="CHAMMI/",
    batch_size=32,
    shuffle=True,
    target_labels='Label',
    split='train',
    augment=True,
)

val_loader = create_chammi_dataloader(
    csv_file="combined_metadata.csv",
    root_dir="CHAMMI/",
    batch_size=32,
    shuffle=False,
    target_labels='Label',
    split='test',
    augment=False,
)

# Training loop
model = YourChannelAdaptiveModel()  # Your model here
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    for batch_images, batch_metadatas, batch_labels in train_loader:
        # Get actual channel counts
        num_channels_list = [meta['num_channels'] for meta in batch_metadatas]
        
        # Process batch (your model should handle variable channels)
        outputs = model(batch_images, num_channels_list)
        
        # Convert labels to tensor if needed
        labels_tensor = torch.tensor([hash(label) % num_classes for label in batch_labels])
        
        loss = criterion(outputs, labels_tensor)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

## Troubleshooting

### Labels are None
- Check that `target_labels` is specified
- Verify enriched metadata files exist in subdirectories
- Check that ID matching works (some datasets use 'Key' instead of 'ID')

### Images not loading
- Verify `root_dir` points to CHAMMI parent directory
- Check file paths in `combined_metadata.csv` are relative to root_dir
- Ensure image files exist at specified paths

### Transform errors
- Images are already tensors after `fold_channels()`
- Transforms should work on `(C, H, W)` tensors
- If using PIL transforms, convert to PIL first: `transforms.ToPILImage()(tensor)`

