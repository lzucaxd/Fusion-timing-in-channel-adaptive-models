"""
Demo script showing exactly how DataLoader works WITHOUT padding.
This shows the exact structure when collate_mode='auto' or 'list'.
"""

from channel_adaptive_pipeline.chammi_dataset import create_chammi_dataloader
import torch
import os

chammi_root = '/Users/zamfiraluca/Downloads/CHAMMI'
csv_file = os.path.join(chammi_root, 'combined_metadata.csv')

print('=' * 80)
print('DataLoader WITHOUT padding (collate_mode="auto" - returns list when mixed)')
print('=' * 80)

# Create dataloader with auto mode (no padding when channels differ)
dataloader = create_chammi_dataloader(
    csv_file=csv_file,
    root_dir=chammi_root,
    batch_size=6,  # Small batch to show mixed channels
    shuffle=True,
    split='train',
    num_workers=0,
    collate_mode='auto',  # This returns list when channels differ
)

# Get a batch
batch_images, batch_metadatas, batch_labels = next(iter(dataloader))

print(f'\nBatch Type: {type(batch_images)}')
print(f'Batch Size (number of samples): {len(batch_images)}')
print()

print('=' * 80)
print('Detailed Structure:')
print('=' * 80)

for i in range(len(batch_images)):
    img = batch_images[i]
    meta = batch_metadatas[i]
    label = batch_labels[i]
    
    print(f'\nSample {i}:')
    print(f'  Type: {type(img)}')
    print(f'  Image shape: {img.shape}  # (channels, height, width)')
    print(f'  Dataset: {meta["dataset_source"]}')
    print(f'  Channels: {meta["num_channels"]}')
    print(f'  Cell type: {meta.get("cell_type", "N/A")}')
    print(f'  File: {meta["file_path"].split("/")[-1][:50]}...')
    print(f'  Label: {label}')
    print(f'  Image dtype: {img.dtype}')
    print(f'  Image range: [{img.min().item():.4f}, {img.max().item():.4f}]')
    
    # Per-channel means
    ch_means = [img[c].mean().item() for c in range(img.shape[0])]
    print(f'  Per-channel means: {[f"{m:.4f}" for m in ch_means]}')

print('\n' + '=' * 80)
print('How to use this in training (example):')
print('=' * 80)
print('''
# In your training loop:
for batch_images, batch_metadatas, batch_labels in dataloader:
    # batch_images is a LIST of tensors, not a single tensor!
    # Each tensor has shape (C, H, W) where C varies (3, 4, or 5)
    
    losses = []
    for i, (img, meta) in enumerate(zip(batch_images, batch_metadatas)):
        num_channels = meta['num_channels']  # 3, 4, or 5
        
        # Your model should handle variable channels
        # img shape: (num_channels, 128, 128)
        img_batch = img.unsqueeze(0)  # Add batch dim: (1, C, H, W)
        output = model(img_batch, num_channels=num_channels)
        
        # Compute loss
        loss = criterion(output, labels[i])
        losses.append(loss)
    
    # Average losses
    total_loss = sum(losses) / len(losses)
    total_loss.backward()
    
# OR: Process in groups by channel count for efficiency
for batch_images, batch_metadatas, batch_labels in dataloader:
    # Group by channel count
    groups = {3: [], 4: [], 5: []}
    for img, meta in zip(batch_images, batch_metadatas):
        groups[meta['num_channels']].append((img, meta))
    
    # Process each group separately (can batch if same channels)
    for num_channels in [3, 4, 5]:
        if groups[num_channels]:
            imgs = [x[0] for x in groups[num_channels]]
            # Stack if all same channels
            imgs_tensor = torch.stack(imgs)  # (N, C, H, W)
            outputs = model(imgs_tensor, num_channels=num_channels)
            # ... compute loss ...
''')

print('\n' + '=' * 80)
print('Comparing with padded mode (for reference):')
print('=' * 80)

# Get a padded batch for comparison
dataloader_padded = create_chammi_dataloader(
    csv_file=csv_file,
    root_dir=chammi_root,
    batch_size=6,
    shuffle=True,
    split='train',
    num_workers=0,
    collate_mode='pad',  # This pads to max channels
)

batch_images_padded, batch_metadatas_padded, batch_labels_padded = next(iter(dataloader_padded))

print(f'\nPadded mode (for comparison):')
print(f'  Type: {type(batch_images_padded)}')
print(f'  Shape: {batch_images_padded.shape}  # (batch_size, max_channels, H, W)')
print(f'  Channel counts in batch: {[m["num_channels"] for m in batch_metadatas_padded]}')
print(f'  Max channels: {batch_images_padded.shape[1]}')
print(f'  Note: Samples with fewer channels are padded with zeros')
print(f'  Example: If max is 5 channels, a 3-channel image will have zeros in channels 3 and 4')
print(f'  Memory usage: Higher (padded tensors take up more space)')

print('\n' + '=' * 80)
print('Key Differences:')
print('=' * 80)
print('''
WITHOUT padding (collate_mode="auto" or "list"):
  - Returns: list of tensors
  - Each tensor: (C, H, W) where C is actual channel count (3, 4, or 5)
  - No wasted memory (no zero-padding)
  - Model must handle variable channels per sample
  - More flexible, better for channel-adaptive models

WITH padding (collate_mode="pad"):
  - Returns: single tensor
  - Shape: (batch_size, max_channels, H, W)
  - All samples padded to same number of channels
  - Uses more memory (zero-padded channels)
  - Model processes fixed-size tensors
  - Less flexible, wastes memory
''')

print('\n' + '=' * 80)
print('Visual representation:')
print('=' * 80)
print('''
Without padding (list):
  batch_images = [
    tensor([3, 128, 128]),  # Allen sample - 3 channels
    tensor([4, 128, 128]),  # HPA sample - 4 channels
    tensor([5, 128, 128]),  # CP sample - 5 channels
    tensor([3, 128, 128]),  # Allen sample - 3 channels
    tensor([4, 128, 128]),  # HPA sample - 4 channels
    tensor([5, 128, 128]),  # CP sample - 5 channels
  ]

With padding (tensor):
  batch_images = tensor([
    [3 channels, 0, 0],      # Allen - padded to 5
    [4 channels, 0],         # HPA - padded to 5
    [5 channels],            # CP - no padding
    [3 channels, 0, 0],      # Allen - padded to 5
    [4 channels, 0],         # HPA - padded to 5
    [5 channels],            # CP - no padding
  ])
  Shape: (6, 5, 128, 128)
''')

