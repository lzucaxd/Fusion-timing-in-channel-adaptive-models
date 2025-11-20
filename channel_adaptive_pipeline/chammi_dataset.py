import os
import torch
import skimage.io

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
from typing import Tuple, List, Optional, Dict, Union
from collections.abc import Sequence

from channel_adaptive_pipeline.folded_dataset import fold_channels, Single_cell_centered, RandomResizedCrop

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class CHAMMIDataset(Dataset):
    """
    Unified dataset class for CHAMMI benchmark.
    Handles all three sub-datasets (Allen/WTC-11, HPA, CP) with variable channel numbers (3, 4, 5).
    
    Returns images with their channel count and metadata to enable channel-adaptive models.
    """
    
    def __init__(
        self, 
        csv_file: str,
        root_dir: str,
        target_labels: Optional[Union[str, List[str]]] = None,
        transform: Optional[callable] = None,
        split: Optional[str] = None,
        resize_to: int = 128,
    ):
        """
        Args:
            csv_file: Path to combined_metadata.csv
            root_dir: Root directory of CHAMMI dataset (parent of Allen/, HPA/, CP/)
            target_labels: Column name(s) in enriched_meta.csv files to use as labels.
                          Can be 'Label' (default classification label) or list of columns.
            transform: Optional transform to be applied on a sample
            split: Filter by train_test_split ('train', 'test', or None for all)
            resize_to: Target size for images (default 128x128 as per benchmark)
        """
        self.metadata = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.target_labels = target_labels
        self.resize_to = resize_to
        
        # Filter by split if specified
        if split is not None:
            split_lower = split.lower()
            if split_lower == 'train':
                self.metadata = self.metadata[self.metadata['train_test_split'] == 'train']
            elif split_lower == 'test':
                self.metadata = self.metadata[self.metadata['train_test_split'] == 'test']
        
        # Reset index after filtering
        self.metadata = self.metadata.reset_index(drop=True)
        
        # Load enriched metadata files for labels
        self._load_enriched_metadata()
        
    def _load_enriched_metadata(self):
        """Load enriched metadata from each sub-dataset for label information."""
        self.enriched_meta = {}
        
        # HPA enriched metadata
        hpa_meta_path = os.path.join(self.root_dir, 'HPA', 'enriched_meta.csv')
        if os.path.exists(hpa_meta_path):
            self.enriched_meta['HPA'] = pd.read_csv(hpa_meta_path)
        
        # Allen enriched metadata
        allen_meta_path = os.path.join(self.root_dir, 'Allen', 'enriched_meta.csv')
        if os.path.exists(allen_meta_path):
            self.enriched_meta['Allen'] = pd.read_csv(allen_meta_path)
        
        # CP enriched metadata
        cp_meta_path = os.path.join(self.root_dir, 'CP', 'enriched_meta.csv')
        if os.path.exists(cp_meta_path):
            self.enriched_meta['CP'] = pd.read_csv(cp_meta_path)
    
    def __len__(self):
        return len(self.metadata)
    
    def _get_dataset_source(self, file_path: str) -> str:
        """Determine which sub-dataset a file belongs to based on path."""
        if file_path.startswith('HPA/'):
            return 'HPA'
        elif file_path.startswith('Allen/'):
            return 'Allen'
        elif file_path.startswith('CP/'):
            return 'CP'
        else:
            raise ValueError(f"Unknown dataset source for path: {file_path}")
    
    def _get_label_from_enriched_meta(self, idx: int, dataset_source: str) -> Optional[Union[str, Dict]]:
        """Get label(s) from enriched metadata files."""
        if self.target_labels is None:
            return None
        
        row = self.metadata.iloc[idx]
        file_path = row['file_path']
        
        # Get the appropriate enriched metadata
        if dataset_source not in self.enriched_meta:
            return None
        
        enriched_df = self.enriched_meta[dataset_source]
        
        # Match by ID or Key (CP uses 'Key', others use 'ID')
        if dataset_source == 'CP':
            match_col = 'Key'
            match_val = row['ID']
        else:
            match_col = 'ID'
            match_val = row['ID']
        
        matched_rows = enriched_df[enriched_df[match_col] == match_val]
        
        if len(matched_rows) == 0:
            # Fallback: try matching by file_path
            matched_rows = enriched_df[enriched_df['file_path'] == file_path]
        
        if len(matched_rows) == 0:
            return None
        
        enriched_row = matched_rows.iloc[0]
        
        # Extract labels
        if isinstance(self.target_labels, str):
            if self.target_labels in enriched_row:
                return enriched_row[self.target_labels]
            elif self.target_labels == 'Label' and 'Label' in enriched_row:
                return enriched_row['Label']
            else:
                return None
        elif isinstance(self.target_labels, list):
            labels = {}
            for label_col in self.target_labels:
                if label_col in enriched_row:
                    labels[label_col] = enriched_row[label_col]
            return labels if labels else None
        else:
            return None
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        row = self.metadata.iloc[idx]
        
        # Load image
        img_path = os.path.join(self.root_dir, row['file_path'])
        channel_width = int(row['channel_width'])
        
        # Load image using skimage (handles both .png and .ome.tiff)
        image = skimage.io.imread(img_path)
        
        # Fold channels from tape format to (H, W, C)
        image = fold_channels(image, channel_width, mode="ignore")
        
        # Convert to tensor if not already (fold_channels returns tensor)
        if not isinstance(image, torch.Tensor):
            image = torchvision.transforms.ToTensor()(image)
        
        # Get metadata
        num_channels = int(row['num_channels'])
        dataset_source = self._get_dataset_source(row['file_path'])
        cell_type = row.get('cell_type', None)
        
        # Get labels if requested
        labels = self._get_label_from_enriched_meta(idx, dataset_source)
        
        # Apply transform (should handle resizing to resize_to)
        if self.transform:
            image = self.transform(image)
        
        # Return image, metadata dict, and labels
        metadata = {
            'num_channels': num_channels,
            'dataset_source': dataset_source,
            'cell_type': cell_type,
            'file_path': row['file_path'],
            'ID': row['ID'],
            'channel_width': channel_width,
            'channels_content': row.get('channels_content', ''),
        }
        
        return image, metadata, labels


class CHAMMITransform(torch.nn.Module):
    """
    Transform for CHAMMI images that resizes to target size (default 128x128).
    Includes proper augmentations for training and optional normalization.
    """
    
    def __init__(
        self, 
        size: int = 128,
        augment: bool = False,
        mode: str = 'center',
        normalize: bool = True,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ):
        """
        Args:
            size: Target size for images (will be size x size)
            augment: If True, apply random augmentations (RandomResizedCrop, horizontal flip, etc.)
            mode: 'center' for centered crop/resize, 'random' for random crop
            normalize: If True, apply normalization using mean/std
            mean: Custom mean values for normalization (per-channel). If None, uses ImageNet stats
            std: Custom std values for normalization (per-channel). If None, uses ImageNet stats
        """
        super().__init__()
        self.size = size
        self.augment = augment
        self.mode = mode
        self.normalize = normalize
        
        # CHAMMI dataset statistics (computed from 5k train samples, per-channel)
        # These are fluorescence microscopy images, not natural images, so we use dataset-specific stats
        self.chammi_stats = {
            3: {
                'mean': [0.1107, 0.1345, 0.0425],
                'std': [0.2593, 0.2815, 0.1218]
            },
            4: {
                'mean': [0.0827, 0.0407, 0.0642, 0.0848],
                'std': [0.1527, 0.0963, 0.1742, 0.1552]
            },
            5: {
                'mean': [0.0998, 0.1934, 0.1625, 0.1810, 0.1479],
                'std': [0.1718, 0.1664, 0.1510, 0.1466, 0.1501]
            }
        }
        
        # If mean/std provided, use them (will be padded/repeated for variable channels)
        # Otherwise, we'll use dataset-specific stats per-channel-count in forward()
        self.use_provided_stats = mean is not None and std is not None
        self.mean = mean
        self.std = std
        
        if augment or mode == 'random':
            self.resize_transform = RandomResizedCrop(
                size=(size, size),
                scale=(0.8, 1.0),
                ratio=(3.0/4.0, 4.0/3.0)
            )
        else:
            # Use centered resize
            self.resize_transform = Single_cell_centered(size=size)
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: Input image tensor of shape (C, H, W) in range [0, 1]
        Returns:
            Transformed image tensor of shape (C, size, size)
        """
        # Apply resize/crop
        img = self.resize_transform(img)
        
        # Apply augmentations if training
        if self.augment:
            # Random horizontal flip
            if torch.rand(1) < 0.5:
                img = transforms.functional.hflip(img)
            
            # Random vertical flip (common for microscopy)
            if torch.rand(1) < 0.5:
                img = transforms.functional.vflip(img)
            
            # Note: No color/brightness jitter for fluorescence microscopy images
            # These images have specific intensity relationships that should be preserved
        
        # Normalize using per-channel statistics
        if self.normalize:
            num_channels = img.shape[0]
            
            # Use dataset-specific stats if not explicitly provided
            if not self.use_provided_stats:
                # Use CHAMMI statistics for this channel count
                if num_channels in self.chammi_stats:
                    mean = self.chammi_stats[num_channels]['mean']
                    std = self.chammi_stats[num_channels]['std']
                else:
                    # Fallback: use stats from closest channel count or pad
                    if num_channels < 3:
                        # Use 3-channel stats, take first N
                        mean = self.chammi_stats[3]['mean'][:num_channels]
                        std = self.chammi_stats[3]['std'][:num_channels]
                    else:
                        # Use 5-channel stats, pad if needed
                        base_mean = self.chammi_stats[5]['mean']
                        base_std = self.chammi_stats[5]['std']
                        mean = base_mean[:num_channels] if num_channels <= 5 else base_mean + [base_mean[-1]] * (num_channels - 5)
                        std = base_std[:num_channels] if num_channels <= 5 else base_std + [base_std[-1]] * (num_channels - 5)
            else:
                # Use provided mean/std, pad/repeat if needed
                if num_channels > len(self.mean):
                    mean = list(self.mean) + [self.mean[-1]] * (num_channels - len(self.mean))
                    std = list(self.std) + [self.std[-1]] * (num_channels - len(self.std))
                else:
                    mean = self.mean[:num_channels]
                    std = self.std[:num_channels]
            
            mean_tensor = torch.tensor(mean, dtype=img.dtype, device=img.device).view(-1, 1, 1)
            std_tensor = torch.tensor(std, dtype=img.dtype, device=img.device).view(-1, 1, 1)
            img = (img - mean_tensor) / std_tensor
        
        return img


def create_chammi_dataloader(
    csv_file: str,
    root_dir: str,
    batch_size: int = 32,
    shuffle: bool = True,
    target_labels: Optional[Union[str, List[str]]] = None,
    split: Optional[str] = None,
    resize_to: int = 128,
    augment: bool = False,
    normalize: bool = True,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    collate_mode: str = 'auto',
) -> DataLoader:
    """
    Convenience function to create a CHAMMI DataLoader.
    
    Args:
        csv_file: Path to combined_metadata.csv
        root_dir: Root directory of CHAMMI dataset
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle the data
        target_labels: Label column(s) to extract from enriched_meta.csv
        split: Filter by 'train' or 'test' split
        resize_to: Target image size (default 128x128)
        augment: Whether to apply data augmentation
        normalize: Whether to apply normalization (default True)
        mean: Custom mean values for normalization (per-channel)
        std: Custom std values for normalization (per-channel)
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        collate_mode: How to handle variable channels ('auto', 'pad', 'list')
            - 'auto': Stack if same channels, return list if mixed (default, recommended)
            - 'pad': Always pad to max channels in batch (uses more memory)
            - 'list': Always return as list of tensors (flexible but slower)
    
    Returns:
        DataLoader instance
    """
    transform = CHAMMITransform(
        size=resize_to, 
        augment=augment,
        normalize=normalize,
        mean=mean,
        std=std
    )
    
    dataset = CHAMMIDataset(
        csv_file=csv_file,
        root_dir=root_dir,
        target_labels=target_labels,
        transform=transform,
        split=split,
        resize_to=resize_to,
    )
    
    # Custom collate function to handle variable channel numbers
    def collate_fn(batch):
        images, metadatas, labels = zip(*batch)
        
        channel_counts = [meta['num_channels'] for meta in metadatas]
        unique_channels = set(channel_counts)
        
        if collate_mode == 'list':
            # Always return as list
            return list(images), list(metadatas), list(labels)
        
        elif collate_mode == 'pad':
            # Always pad to max channels
            max_channels = max(channel_counts)
            padded_images = []
            for img, meta in zip(images, metadatas):
                if img.shape[0] < max_channels:
                    padding = torch.zeros(
                        max_channels - img.shape[0],
                        img.shape[1],
                        img.shape[2],
                        dtype=img.dtype,
                        device=img.device
                    )
                    img = torch.cat([img, padding], dim=0)
                padded_images.append(img)
            batch_images = torch.stack(padded_images)
            return batch_images, list(metadatas), list(labels)
        
        else:  # 'auto' mode (default)
            if len(unique_channels) == 1:
                # All same channels - stack normally for efficiency
                batch_images = torch.stack(images)
                return batch_images, list(metadatas), list(labels)
            else:
                # Mixed channels - return as list for flexible handling
                # Models can use metadata['num_channels'] to process each sample
                return list(images), list(metadatas), list(labels)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

