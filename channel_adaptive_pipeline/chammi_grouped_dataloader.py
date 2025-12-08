"""
Grouped DataLoaders for CHAMMI dataset.

Creates separate DataLoaders for each channel count (3, 4, 5) and provides
utilities for random batch interleaving during training.
"""

import torch
import random
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Dict, List, Optional, Iterator, Tuple
from channel_adaptive_pipeline.chammi_dataset import CHAMMIDataset, CHAMMITransform


# Module-level collate function for multiprocessing support
def _chammi_collate_fn(batch):
    """
    Collate function for CHAMMI batches.
    Defined at module level for pickling with multiprocessing.
    """
    images, metadatas, labels = zip(*batch)
    # Stack images (all have same channels within a batch)
    batch_images = torch.stack(images)
    batch_metadatas = list(metadatas)
    batch_labels = list(labels)  # Keep as list (may be strings or mixed types)
    return batch_images, batch_metadatas, batch_labels


# Module-level transform wrapper for multiprocessing support
class TransformWrapper(Dataset):
    """Wrapper for applying transforms to dataset subsets."""
    def __init__(self, subset: Subset, transform):
        self.subset = subset
        self.transform = transform
    
    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx):
        image, metadata, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, metadata, label


def create_grouped_chammi_dataloaders(
    csv_file: str,
    root_dir: str,
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    target_labels: Optional[str] = None,
    augment: bool = True,
    normalize: bool = True,
    resize_to: int = 128,
    drop_last: bool = True,
) -> Dict[int, DataLoader]:
    """
    Create separate DataLoaders for each channel count (3, 4, 5).
    
    Args:
        csv_file: Path to combined_metadata.csv
        root_dir: Root directory of CHAMMI dataset
        split: Dataset split ('train', 'test', etc.)
        batch_size: Batch size per DataLoader
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle samples
        target_labels: Label column name (e.g., 'Label')
        augment: Whether to apply augmentations
        normalize: Whether to normalize images
        resize_to: Target image size
        drop_last: Whether to drop last incomplete batch
    
    Returns:
        Dict mapping channel_count -> DataLoader
        Example: {3: DataLoader(...), 4: DataLoader(...), 5: DataLoader(...)}
        
        Each DataLoader yields batches of:
        - images: (B, C, 128, 128) where C is fixed for that loader
        - metadatas: list[dict]
        - labels: list (may contain strings or other types)
    """
    # Create base dataset (no transform yet - will apply in wrapper)
    base_dataset = CHAMMIDataset(
        csv_file=csv_file,
        root_dir=root_dir,
        target_labels=target_labels,
        transform=None,  # Transform applied in wrapper
        split=split,
        resize_to=resize_to,
    )
    
    # Group indices by channel count
    indices_by_channels: Dict[int, List[int]] = {3: [], 4: [], 5: []}
    
    for i in range(len(base_dataset)):
        row = base_dataset.metadata.iloc[i]
        num_channels = int(row['num_channels'])
        if num_channels in indices_by_channels:
            indices_by_channels[num_channels].append(i)
    
    # Create transform
    transform = CHAMMITransform(
        size=resize_to,
        augment=augment,
        normalize=normalize,
    )
    
    # Create DataLoaders for each channel count
    dataloaders = {}
    for num_channels, indices in indices_by_channels.items():
        if len(indices) == 0:
            continue
        
        # Create subset
        subset = Subset(base_dataset, indices)
        
        # Apply transform wrapper
        transformed_dataset = TransformWrapper(subset, transform)
        
        # Create DataLoader
        dataloader = DataLoader(
            transformed_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=drop_last,
            collate_fn=_chammi_collate_fn,  # Module-level function
        )
        
        dataloaders[num_channels] = dataloader
        print(f"Created DataLoader for {num_channels} channels: {len(indices)} samples, {len(dataloader)} batches")
    
    return dataloaders


def create_random_interleaved_iterator(
    dataloaders: Dict[int, DataLoader],
    random_seed: Optional[int] = None,
) -> Iterator[Tuple[torch.Tensor, List[Dict], List, int]]:
    """
    Create an iterator that randomly interleaves batches from different channel groups.
    
    Args:
        dataloaders: Dict mapping channel_count -> DataLoader (from create_grouped_chammi_dataloaders)
        random_seed: Optional random seed for reproducibility
    
    Yields:
        (images, metadatas, labels, channel_count)
        - images: (B, C, 128, 128) tensor
        - metadatas: list of metadata dicts
        - labels: list of labels
        - channel_count: int (3, 4, or 5)
    
    Example:
        dataloaders = create_grouped_chammi_dataloaders(...)
        iterator = create_random_interleaved_iterator(dataloaders)
        
        for images, metadatas, labels, channel_count in iterator:
            # Process batch - channel_count tells you which dataset this came from
            ...
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    # Create iterators for each DataLoader
    iterators = {ch: iter(dl) for ch, dl in dataloaders.items()}
    
    # Count total batches per channel group
    batches_per_channel = {ch: len(dl) for ch, dl in dataloaders.items()}
    total_batches = sum(batches_per_channel.values())
    
    # Track batches processed per channel
    batches_processed = {ch: 0 for ch in dataloaders.keys()}
    total_processed = 0
    
    while total_processed < total_batches:
        # Get available channel groups that still have batches
        available_channels = [
            ch for ch in dataloaders.keys()
            if batches_processed[ch] < batches_per_channel[ch]
        ]
        
        if not available_channels:
            break
        
        # Randomly choose a channel group
        chosen_channel = random.choice(available_channels)
        
        try:
            # Get batch from chosen channel group
            images, metadatas, labels = next(iterators[chosen_channel])
            batches_processed[chosen_channel] += 1
            total_processed += 1
            
            yield images, metadatas, labels, chosen_channel
            
        except StopIteration:
            # This channel group is exhausted
            continue


def create_dataset_specific_dataloaders(
    csv_file: str,
    root_dir: str,
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    target_labels: Optional[str] = None,
    augment: bool = True,
    normalize: bool = True,
    resize_to: int = 128,
    drop_last: bool = True,
) -> Dict[str, DataLoader]:
    """
    Create separate DataLoaders for each dataset (Allen, HPA, CP).
    
    Args:
        Same as create_grouped_chammi_dataloaders
    
    Returns:
        Dict mapping dataset_source -> DataLoader
        Example: {'Allen': DataLoader(...), 'HPA': DataLoader(...), 'CP': DataLoader(...)}
    """
    # Create base dataset
    base_dataset = CHAMMIDataset(
        csv_file=csv_file,
        root_dir=root_dir,
        target_labels=target_labels,
        transform=None,
        split=split,
        resize_to=resize_to,
    )
    
    # Group indices by dataset source
    indices_by_dataset: Dict[str, List[int]] = {'Allen': [], 'HPA': [], 'CP': []}
    
    for i in range(len(base_dataset)):
        row = base_dataset.metadata.iloc[i]
        file_path = row['file_path']
        dataset_source = file_path.split('/')[0]
        
        if dataset_source in indices_by_dataset:
            indices_by_dataset[dataset_source].append(i)
    
    # Create transform
    transform = CHAMMITransform(
        size=resize_to,
        augment=augment,
        normalize=normalize,
    )
    
    # Create DataLoaders for each dataset
    dataloaders = {}
    for dataset_source, indices in indices_by_dataset.items():
        if len(indices) == 0:
            continue
        
        # Create subset
        subset = Subset(base_dataset, indices)
        
        # Apply transform wrapper
        transformed_dataset = TransformWrapper(subset, transform)
        
        # Create DataLoader
        dataloader = DataLoader(
            transformed_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=drop_last,
            collate_fn=_chammi_collate_fn,  # Module-level function
        )
        
        dataloaders[dataset_source] = dataloader
        print(f"Created DataLoader for {dataset_source}: {len(indices)} samples, {len(dataloader)} batches")
    
    return dataloaders


def create_random_dataset_interleaved_iterator(
    dataloaders: Dict[str, DataLoader],
    random_seed: Optional[int] = None,
) -> Iterator[Tuple[torch.Tensor, List[Dict], List, str]]:
    """
    Create an iterator that randomly interleaves batches from different datasets.
    
    Args:
        dataloaders: Dict mapping dataset_source -> DataLoader (from create_dataset_specific_dataloaders)
        random_seed: Optional random seed for reproducibility
    
    Yields:
        (images, metadatas, labels, dataset_source)
        - images: (B, C, 128, 128) tensor where C depends on dataset (3 for Allen, 4 for HPA, 5 for CP)
        - metadatas: list of metadata dicts
        - labels: list of labels
        - dataset_source: str ('Allen', 'HPA', or 'CP')
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    # Create iterators for each DataLoader
    iterators = {ds: iter(dl) for ds, dl in dataloaders.items()}
    
    # Count total batches per dataset
    batches_per_dataset = {ds: len(dl) for ds, dl in dataloaders.items()}
    total_batches = sum(batches_per_dataset.values())
    
    # Track batches processed per dataset
    batches_processed = {ds: 0 for ds in dataloaders.keys()}
    total_processed = 0
    
    while total_processed < total_batches:
        # Get available datasets that still have batches
        available_datasets = [
            ds for ds in dataloaders.keys()
            if batches_processed[ds] < batches_per_dataset[ds]
        ]
        
        if not available_datasets:
            break
        
        # Randomly choose a dataset
        chosen_dataset = random.choice(available_datasets)
        
        try:
            # Get batch from chosen dataset
            images, metadatas, labels = next(iterators[chosen_dataset])
            batches_processed[chosen_dataset] += 1
            total_processed += 1
            
            yield images, metadatas, labels, chosen_dataset
            
        except StopIteration:
            # This dataset is exhausted
            continue
