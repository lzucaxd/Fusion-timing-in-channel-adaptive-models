"""
Compute dataset statistics (mean, std) for CHAMMI dataset normalization.
This helps determine proper normalization values instead of using ImageNet stats.
"""

import os
import torch
import numpy as np
from typing import Dict, Optional, List
from tqdm import tqdm
from channel_adaptive_pipeline.chammi_dataset import CHAMMIDataset, CHAMMITransform


def compute_dataset_statistics(
    csv_file: str,
    root_dir: str,
    split: str = 'train',
    num_samples: Optional[int] = None,
    resize_to: int = 128,
    max_channels: int = 5,
) -> Dict[int, Dict[str, List[float]]]:
    """
    Compute mean and std statistics for each channel count separately.
    
    Args:
        csv_file: Path to combined_metadata.csv
        root_dir: Root directory of CHAMMI dataset
        split: Dataset split to use ('train', 'test', or None for all)
        num_samples: Number of samples to use (None for all)
        resize_to: Target image size
        max_channels: Maximum number of channels to compute stats for
    
    Returns:
        Dict mapping channel count to {'mean': [...], 'std': [...]}
    """
    # Create dataset without normalization
    transform = CHAMMITransform(size=resize_to, augment=False, normalize=False)
    dataset = CHAMMIDataset(
        csv_file=csv_file,
        root_dir=root_dir,
        transform=transform,
        split=split,
    )
    
    # Group samples by channel count
    samples_by_channels = {c: [] for c in range(3, max_channels + 1)}
    
    num_samples_to_use = num_samples if num_samples else len(dataset)
    indices = np.random.choice(len(dataset), min(num_samples_to_use, len(dataset)), replace=False)
    
    print(f"Computing statistics from {len(indices)} samples...")
    
    for idx in tqdm(indices):
        image, metadata, _ = dataset[idx]
        num_channels = metadata['num_channels']
        
        if num_channels in samples_by_channels:
            samples_by_channels[num_channels].append(image)
    
    # Compute statistics per channel count
    stats = {}
    
    for num_channels, images in samples_by_channels.items():
        if len(images) == 0:
            continue
        
        print(f"\nComputing stats for {num_channels} channels ({len(images)} samples)...")
        
        # Stack all images
        stacked = torch.stack(images)  # (N, C, H, W)
        
        # Compute mean and std per channel
        # Shape: (C,)
        mean = stacked.mean(dim=(0, 2, 3)).tolist()  # Mean over batch and spatial dims
        std = stacked.std(dim=(0, 2, 3)).tolist()  # Std over batch and spatial dims
        
        stats[num_channels] = {
            'mean': mean,
            'std': std,
        }
        
        print(f"  Mean: {[f'{m:.4f}' for m in mean]}")
        print(f"  Std:  {[f'{s:.4f}' for s in std]}")
    
    return stats


def main():
    """Compute and print dataset statistics."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute CHAMMI dataset statistics')
    parser.add_argument('--csv', type=str, required=True, help='Path to combined_metadata.csv')
    parser.add_argument('--root', type=str, required=True, help='Path to CHAMMI root directory')
    parser.add_argument('--split', type=str, default='train', help='Dataset split to use')
    parser.add_argument('--num-samples', type=int, default=None, help='Number of samples to use')
    parser.add_argument('--resize', type=int, default=128, help='Target image size')
    
    args = parser.parse_args()
    
    stats = compute_dataset_statistics(
        csv_file=args.csv,
        root_dir=args.root,
        split=args.split,
        num_samples=args.num_samples,
        resize_to=args.resize,
    )
    
    print("\n" + "=" * 70)
    print("Dataset Statistics Summary")
    print("=" * 70)
    print("\nUse these values for normalization:")
    print("\n# Example usage:")
    print("from channel_adaptive_pipeline.chammi_dataset import create_chammi_dataloader")
    print()
    
    for num_channels in sorted(stats.keys()):
        mean_str = str(stats[num_channels]['mean'])
        std_str = str(stats[num_channels]['std'])
        print(f"# {num_channels} channels:")
        print(f"#   mean={mean_str}")
        print(f"#   std={std_str}")
        print()


if __name__ == "__main__":
    main()

