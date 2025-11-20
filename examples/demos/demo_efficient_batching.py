"""
Demo: Efficient batching with separate DataLoaders per channel count.
This solves the efficiency problem by ensuring all samples in a batch have the same channels.
"""

import sys
sys.path.insert(0, '.')
from channel_adaptive_pipeline.chammi_grouped_dataloader import create_grouped_chammi_dataloaders, create_interleaved_chammi_dataloader
import os
import torch

chammi_root = '/Users/zamfiraluca/Downloads/CHAMMI'
csv_file = os.path.join(chammi_root, 'combined_metadata.csv')

print('=' * 80)
print('EFFICIENT BATCHING SOLUTION')
print('=' * 80)
print()
print('Problem: Mixed batches (3, 4, 5 channels together) can\'t be stacked efficiently')
print('Solution: Separate DataLoaders for each channel count')
print()

print('=' * 80)
print('Option 1: Separate DataLoaders (process each dataset type separately)')
print('=' * 80)

dataloaders = create_grouped_chammi_dataloaders(
    csv_file=csv_file,
    root_dir=chammi_root,
    batch_size=32,
    shuffle=True,
    split='train',
    augment=True,
    num_workers=0,
)

print('\nBatch shapes with separate DataLoaders:')
print('-' * 80)
for channel_count in sorted(dataloaders.keys()):
    dataloader = dataloaders[channel_count]
    batch_images, batch_metadatas, batch_labels = next(iter(dataloader))
    print(f'{channel_count} channels: batch_images.shape = {batch_images.shape}')
    print(f'  → Can process {batch_images.shape[0]} samples at once!')
    print(f'  → All have {channel_count} channels → efficient batching ✓')

print('\n' + '=' * 80)
print('Training Loop Example (Option 1):')
print('=' * 80)
print('''
# Process each channel count separately
for channel_count in [3, 4, 5]:
    dataloader = dataloaders[channel_count]
    
    for epoch in range(num_epochs):
        for batch_images, batch_metadatas, batch_labels in dataloader:
            # batch_images shape: (batch_size, channel_count, 128, 128)
            # All samples have SAME channels → can process efficiently!
            
            # Forward pass on entire batch at once
            outputs = model(batch_images, num_channels=channel_count)
            
            # Loss computation
            loss = criterion(outputs, batch_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
''')

print('=' * 80)
print('Option 2: Interleaved DataLoader (mix batches from all channel counts)')
print('=' * 80)

interleaved = create_interleaved_chammi_dataloader(
    csv_file=csv_file,
    root_dir=chammi_root,
    batch_size_per_channel=32,
    split='train',
    augment=True,
    num_workers=0,
)

print('\nFirst 3 batches from interleaved DataLoader:')
print('-' * 80)
for i, (batch_images, batch_metadatas, batch_labels, channel_count) in enumerate(interleaved):
    if i >= 3:
        break
    print(f'Batch {i+1}: channel_count={channel_count}, shape={batch_images.shape}')
    print(f'  → All {batch_images.shape[0]} samples have {channel_count} channels ✓')

print('\n' + '=' * 80)
print('Training Loop Example (Option 2):')
print('=' * 80)
print('''
# Interleave batches from all channel counts
for epoch in range(num_epochs):
    interleaved = create_interleaved_chammi_dataloader(...)
    
    for batch_images, batch_metadatas, batch_labels, channel_count in interleaved:
        # Each batch has consistent channel count
        # batch_images shape: (batch_size, channel_count, 128, 128)
        
        # Forward pass on entire batch
        outputs = model(batch_images, num_channels=channel_count)
        
        # Loss and backward
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
''')

print('=' * 80)
print('Comparison:')
print('=' * 80)
print('''
WITHOUT grouping (mixed batches):
  batch_images = list of tensors with shapes:
    - (3, 128, 128)
    - (4, 128, 128)
    - (5, 128, 128)
    - ...
  → Must process ONE BY ONE (slow)
  → No true batching

WITH grouping (separate DataLoaders):
  batch_images = tensor with shape:
    - (32, 3, 128, 128) for 3-channel batch
    - (32, 4, 128, 128) for 4-channel batch
    - (32, 5, 128, 128) for 5-channel batch
  → Can process ENTIRE BATCH at once (fast!)
  → True batching, efficient GPU utilization
''')

