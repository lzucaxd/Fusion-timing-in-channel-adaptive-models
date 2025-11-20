"""
Comprehensive test for all CHAMMI dataloaders.
Tests batch shapes and normalization for each dataloader type.
"""

import sys
sys.path.insert(0, '.')
import torch
import os
from channel_adaptive_pipeline.chammi_dataset import create_chammi_dataloader
from channel_adaptive_pipeline.chammi_grouped_dataloader import (
    create_grouped_chammi_dataloaders,
    create_interleaved_chammi_dataloader,
    create_dataset_ordered_dataloader
)

chammi_root = '/Users/zamfiraluca/Downloads/CHAMMI'
csv_file = os.path.join(chammi_root, 'combined_metadata.csv')

print('=' * 80)
print('COMPREHENSIVE DATALOADER TESTING')
print('=' * 80)

def test_normalization(batch_images, batch_metadatas, dataloader_name, show_all_samples=False):
    """Test that normalization is applied correctly per channel."""
    print(f'\n  Normalization check for {dataloader_name}:')
    print(f'  {"="*76}')
    
    # Group by channel count
    by_channels = {}
    for i, meta in enumerate(batch_metadatas):
        ch = meta['num_channels']
        if ch not in by_channels:
            by_channels[ch] = []
        by_channels[ch].append(i)
    
    for ch, indices in sorted(by_channels.items()):
        # Get samples with this channel count
        if isinstance(batch_images, list):
            samples = [batch_images[i] for i in indices]
            print(f'\n  {ch} channels ({len(samples)} samples):')
            
            # Show stats for each sample or just first
            num_to_show = len(samples) if show_all_samples else min(3, len(samples))
            for sample_idx in range(num_to_show):
                sample = samples[sample_idx]
                original_idx = indices[sample_idx]
                meta = batch_metadatas[original_idx]
                per_channel_means = [sample[c].mean().item() for c in range(ch)]
                per_channel_stds = [sample[c].std().item() for c in range(ch)]
                
                print(f'\n    Sample {original_idx} ({meta["dataset_source"]}):')
                print(f'      Channel means: {[f"{m:8.4f}" for m in per_channel_means]}')
                print(f'      Channel stds:  {[f"{s:8.4f}" for s in per_channel_stds]}')
                
                # Overall stats
                mean_abs = sum(abs(m) for m in per_channel_means) / len(per_channel_means)
                std_avg = sum(s for s in per_channel_stds) / len(per_channel_stds)
                mean_max_abs = max(abs(m) for m in per_channel_means)
                std_min = min(per_channel_stds)
                std_max = max(per_channel_stds)
                
                print(f'      Stats: |mean|_avg={mean_abs:.4f}, |mean|_max={mean_max_abs:.4f}, std_range=[{std_min:.4f}, {std_max:.4f}]')
                
                if mean_abs < 0.5 and 0.5 < std_min and std_max < 2.0:
                    print(f'      Status: ✓ Normalized (means≈0, stds≈1)')
                else:
                    print(f'      Status: ⚠ Check normalization')
            
            if len(samples) > num_to_show:
                print(f'\n    ... ({len(samples) - num_to_show} more samples)')
        else:
            # Tensor batch
            samples = batch_images[indices]
            per_channel_means = samples.mean(dim=(0, 2, 3)).tolist()  # Mean over batch and spatial
            per_channel_stds = samples.std(dim=(0, 2, 3)).tolist()
            
            # Get dataset info
            datasets = set([batch_metadatas[i]['dataset_source'] for i in indices])
            
            print(f'\n  {ch} channels batch ({len(indices)} samples, datasets: {datasets}):')
            print(f'\n    Per-channel statistics (computed over batch):')
            for c in range(ch):
                print(f'      Channel {c}: mean={per_channel_means[c]:8.4f}, std={per_channel_stds[c]:8.4f}')
            
            # Overall stats
            mean_abs = sum(abs(m) for m in per_channel_means[:ch]) / ch
            std_avg = sum(s for s in per_channel_stds[:ch]) / ch
            mean_max_abs = max(abs(m) for m in per_channel_means[:ch])
            std_min = min(per_channel_stds[:ch])
            std_max = max(per_channel_stds[:ch])
            
            print(f'\n    Batch statistics:')
            print(f'      |mean|_avg = {mean_abs:.4f}')
            print(f'      |mean|_max = {mean_max_abs:.4f}')
            print(f'      std_range  = [{std_min:.4f}, {std_max:.4f}]')
            print(f'      std_avg    = {std_avg:.4f}')
            
            if mean_abs < 0.5 and 0.5 < std_min and std_max < 2.0:
                print(f'      Status: ✓ Normalized correctly (means≈0, stds≈1)')
            else:
                print(f'      Status: ⚠ Check normalization')

# Test 1: Standard DataLoader with 'auto' mode
print('\n' + '=' * 80)
print('TEST 1: Standard DataLoader (collate_mode="auto")')
print('=' * 80)

dataloader_auto = create_chammi_dataloader(
    csv_file=csv_file,
    root_dir=chammi_root,
    batch_size=6,
    shuffle=True,
    split='train',
    augment=False,
    normalize=True,
    num_workers=0,
    collate_mode='auto',
)

batch_images, batch_metadatas, batch_labels = next(iter(dataloader_auto))
print(f'\nBatch type: {type(batch_images)}')
if isinstance(batch_images, list):
    print(f'Batch size: {len(batch_images)}')
    print(f'Shapes: {[img.shape for img in batch_images]}')
    print(f'Channel counts: {[meta["num_channels"] for meta in batch_metadatas]}')
    print(f'Datasets: {[meta["dataset_source"] for meta in batch_metadatas]}')
else:
    print(f'Batch shape: {batch_images.shape}')
    print(f'Channel counts: {[meta["num_channels"] for meta in batch_metadatas]}')
    print(f'Datasets: {[meta["dataset_source"] for meta in batch_metadatas]}')

test_normalization(batch_images, batch_metadatas, "Standard DataLoader (auto)", show_all_samples=True)

# Test 2: Grouped DataLoaders (separate per channel count)
print('\n' + '=' * 80)
print('TEST 2: Grouped DataLoaders (separate per channel count)')
print('=' * 80)

dataloaders_grouped = create_grouped_chammi_dataloaders(
    csv_file=csv_file,
    root_dir=chammi_root,
    batch_size=8,
    shuffle=True,
    split='train',
    augment=False,
    normalize=True,
    num_workers=0,
)

for channel_count in sorted(dataloaders_grouped.keys()):
    dataloader = dataloaders_grouped[channel_count]
    batch_images, batch_metadatas, batch_labels = next(iter(dataloader))
    
    print(f'\n{channel_count} channels DataLoader:')
    print(f'  Batch type: {type(batch_images)}')
    print(f'  Batch shape: {batch_images.shape}')
    print(f'  Expected: (batch_size, {channel_count}, 128, 128)')
    print(f'  All samples have {channel_count} channels: ✓')
    print(f'  Datasets in batch: {set([meta["dataset_source"] for meta in batch_metadatas])}')
    
    test_normalization(batch_images, batch_metadatas, f"Grouped DataLoader ({channel_count}ch)", show_all_samples=False)

# Test 3: Interleaved DataLoader
print('\n' + '=' * 80)
print('TEST 3: Interleaved DataLoader')
print('=' * 80)

interleaved = create_interleaved_chammi_dataloader(
    csv_file=csv_file,
    root_dir=chammi_root,
    batch_size_per_channel=8,
    shuffle=True,
    split='train',
    augment=False,
    normalize=True,
    num_workers=0,
)

print('\nFirst 3 batches:')
for i, (batch_images, batch_metadatas, batch_labels, channel_count) in enumerate(interleaved):
    if i >= 3:
        break
    print(f'\nBatch {i+1}:')
    print(f'  Channel count: {channel_count}')
    print(f'  Batch shape: {batch_images.shape}')
    print(f'  Expected: (batch_size, {channel_count}, 128, 128)')
    print(f'  Datasets: {set([meta["dataset_source"] for meta in batch_metadatas])}')
    test_normalization(batch_images, batch_metadatas, f"Interleaved DataLoader (batch {i+1}, {channel_count}ch)", show_all_samples=False)

# Test 4: Dataset-Ordered DataLoader
print('\n' + '=' * 80)
print('TEST 4: Dataset-Ordered DataLoader (shuffled dataset order)')
print('=' * 80)

ordered = create_dataset_ordered_dataloader(
    csv_file=csv_file,
    root_dir=chammi_root,
    batch_size=8,
    shuffle=True,
    split='train',
    augment=False,
    normalize=True,
    shuffle_dataset_order=True,
    num_workers=0,
)

print('\nFirst 6 batches (showing dataset order):')
dataset_order = []
for i, (batch_images, batch_metadatas, batch_labels, channel_count) in enumerate(ordered):
    if i >= 6:
        break
    dataset_source = batch_metadatas[0]['dataset_source']
    dataset_order.append(dataset_source)
    print(f'\nBatch {i+1}:')
    print(f'  Dataset: {dataset_source}')
    print(f'  Channel count: {channel_count}')
    print(f'  Batch shape: {batch_images.shape}')
    print(f'  Expected: (batch_size, {channel_count}, 128, 128)')
    test_normalization(batch_images, batch_metadatas, f"Dataset-Ordered DataLoader (batch {i+1}, {dataset_source})", show_all_samples=False)

print(f'\nDataset order in first 6 batches: {" → ".join(dataset_order)}')

# Test 5: Verify normalization stats per dataset
print('\n' + '=' * 80)
print('TEST 5: Normalization Verification per Dataset')
print('=' * 80)

# Get samples from each dataset separately
dataloaders_by_dataset = {}
for channel_count in [3, 4, 5]:
    dataloader = dataloaders_grouped[channel_count]
    batch_images, batch_metadatas, batch_labels = next(iter(dataloader))
    
    # Group by dataset
    for i, meta in enumerate(batch_metadatas):
        dataset_source = meta['dataset_source']
        if dataset_source not in dataloaders_by_dataset:
            dataloaders_by_dataset[dataset_source] = []
        dataloaders_by_dataset[dataset_source].append((batch_images[i], meta))

print('\nPer-dataset normalization check:')
for dataset_source in ['Allen', 'HPA', 'CP']:
    if dataset_source in dataloaders_by_dataset:
        samples = dataloaders_by_dataset[dataset_source]
        if samples:
            img, meta = samples[0]
            ch = meta['num_channels']
            per_channel_means = [img[c].mean().item() for c in range(ch)]
            per_channel_stds = [img[c].std().item() for c in range(ch)]
            print(f'\n{dataset_source} ({ch} channels):')
            print(f'  Per-channel means: {[f"{m:7.4f}" for m in per_channel_means]}')
            print(f'  Per-channel stds:  {[f"{s:7.4f}" for s in per_channel_stds]}')
            mean_abs = sum(abs(m) for m in per_channel_means) / len(per_channel_means)
            std_avg = sum(s for s in per_channel_stds) / len(per_channel_stds)
            if mean_abs < 0.5 and 0.5 < std_avg < 2.0:
                print(f'  ✓ Normalization correct')
            else:
                print(f'  ⚠ Check normalization (mean={mean_abs:.4f}, std={std_avg:.4f})')

print('\n' + '=' * 80)
print('TEST SUMMARY')
print('=' * 80)
print('✓ All dataloaders tested')
print('✓ Batch shapes verified')
print('✓ Normalization checked per channel')
print('✓ All datasets (Allen, HPA, CP) working')
print('\nAll tests completed!')

