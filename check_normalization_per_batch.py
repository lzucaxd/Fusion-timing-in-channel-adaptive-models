"""
Visual check of normalization: shows mean and std for each batch per channel.
This allows visual verification that normalization is working correctly.
"""

import sys
sys.path.insert(0, '.')
import torch
import os
from channel_adaptive_pipeline.chammi_grouped_dataloader import create_grouped_chammi_dataloaders

chammi_root = '/Users/zamfiraluca/Downloads/CHAMMI'
csv_file = os.path.join(chammi_root, 'combined_metadata.csv')

print('=' * 100)
print('NORMALIZATION CHECK: Mean and Std per Batch per Channel')
print('=' * 100)
print('\nAfter normalization, we expect:')
print('  - Mean ≈ 0 (close to zero)')
print('  - Std ≈ 1 (close to one)')
print()

# Create grouped dataloaders
dataloaders = create_grouped_chammi_dataloaders(
    csv_file=csv_file,
    root_dir=chammi_root,
    batch_size=16,  # Larger batch for better stats
    shuffle=True,
    split='train',
    augment=False,
    normalize=True,
    num_workers=0,
)

# Check multiple batches for each channel count
num_batches_to_check = 3

for channel_count in sorted(dataloaders.keys()):
    dataloader = dataloaders[channel_count]
    
    print('\n' + '=' * 100)
    print(f'{channel_count} CHANNELS DataLoader')
    print('=' * 100)
    
    # Get dataset name
    first_batch = next(iter(dataloader))
    dataset_name = first_batch[1][0]['dataset_source']
    print(f'Dataset: {dataset_name}')
    print(f'Batch shape: (batch_size, {channel_count}, 128, 128)')
    print()
    
    # Reset iterator and check multiple batches
    dataloader_iter = iter(dataloader)
    
    for batch_idx in range(num_batches_to_check):
        batch_images, batch_metadatas, batch_labels = next(dataloader_iter)
        
        # Compute per-channel statistics over the batch
        # Shape: (batch_size, channels, H, W)
        per_channel_means = batch_images.mean(dim=(0, 2, 3)).tolist()  # Mean over batch and spatial dims
        per_channel_stds = batch_images.std(dim=(0, 2, 3)).tolist()    # Std over batch and spatial dims
        
        print(f'Batch {batch_idx + 1} (batch_size={batch_images.shape[0]}):')
        print('-' * 100)
        print(f'{"Channel":<10} {"Mean":<15} {"Std":<15} {"|Mean|":<15} {"Status":<20}')
        print('-' * 100)
        
        for ch in range(channel_count):
            mean_val = per_channel_means[ch]
            std_val = per_channel_stds[ch]
            mean_abs = abs(mean_val)
            
            # Determine status
            if mean_abs < 0.1 and 0.8 < std_val < 1.2:
                status = "✓ Excellent"
            elif mean_abs < 0.3 and 0.6 < std_val < 1.5:
                status = "✓ Good"
            elif mean_abs < 0.5 and 0.5 < std_val < 2.0:
                status = "✓ OK"
            else:
                status = "⚠ Check"
            
            print(f'Channel {ch:<3}  {mean_val:>10.4f}     {std_val:>10.4f}     {mean_abs:>10.4f}     {status}')
        
        # Overall batch statistics
        mean_abs_avg = sum(abs(m) for m in per_channel_means) / channel_count
        mean_abs_max = max(abs(m) for m in per_channel_means)
        std_avg = sum(per_channel_stds) / channel_count
        std_min = min(per_channel_stds)
        std_max = max(per_channel_stds)
        
        print('-' * 100)
        print(f'Batch Stats:')
        print(f'  |Mean|_avg (across channels) = {mean_abs_avg:.4f}  (should be < 0.5)')
        print(f'  |Mean|_max (across channels) = {mean_abs_max:.4f}  (should be < 0.5)')
        print(f'  Std_avg (across channels)    = {std_avg:.4f}  (should be ≈ 1.0)')
        print(f'  Std_range                    = [{std_min:.4f}, {std_max:.4f}]  (should be [0.5, 2.0])')
        
        # Overall status
        if mean_abs_avg < 0.3 and 0.7 < std_min and std_max < 1.5:
            print(f'  Status: ✓✓ Normalization looks excellent!')
        elif mean_abs_avg < 0.5 and 0.5 < std_min and std_max < 2.0:
            print(f'  Status: ✓ Normalization looks good')
        else:
            print(f'  Status: ⚠ Normalization may need adjustment')
        print()

print('=' * 100)
print('SUMMARY')
print('=' * 100)
print('\nFor properly normalized data:')
print('  ✓ Mean should be close to 0 (ideally < 0.3, acceptable < 0.5)')
print('  ✓ Std should be close to 1 (ideally [0.8, 1.2], acceptable [0.5, 2.0])')
print('\nIf means are consistently far from 0 or stds far from 1,')
print('the normalization statistics may need to be recomputed.')
print()

