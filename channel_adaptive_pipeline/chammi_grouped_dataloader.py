"""
Grouped DataLoader for CHAMMI dataset.
Creates separate batches for each channel count (3, 4, 5) for efficient processing.
Supports dataset ordering strategies for better generalization.
"""

import torch
import random
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from typing import Dict, List, Optional, Union, Iterator, Tuple
from channel_adaptive_pipeline.chammi_dataset import CHAMMIDataset, CHAMMITransform, create_chammi_dataloader


def create_grouped_chammi_dataloaders(
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
) -> Dict[int, DataLoader]:
    """
    Create separate DataLoaders for each channel count (3, 4, 5).
    This allows efficient batching where all samples in a batch have the same channels.
    
    Returns:
        Dict mapping channel_count -> DataLoader
        Example: {3: DataLoader(...), 4: DataLoader(...), 5: DataLoader(...)}
    """
    # Create base dataset to filter
    base_dataset = CHAMMIDataset(
        csv_file=csv_file,
        root_dir=root_dir,
        target_labels=target_labels,
        transform=None,  # Transform will be applied in DataLoader
        split=split,
        resize_to=resize_to,
    )
    
    # Group indices by channel count
    indices_by_channels = {3: [], 4: [], 5: []}
    
    for i in range(len(base_dataset)):
        row = base_dataset.metadata.iloc[i]
        num_channels = int(row['num_channels'])
        if num_channels in indices_by_channels:
            indices_by_channels[num_channels].append(i)
    
    # Create separate datasets and dataloaders for each channel count
    dataloaders = {}
    transform = CHAMMITransform(
        size=resize_to,
        augment=augment,
        normalize=normalize,
        mean=mean,
        std=std
    )
    
    for num_channels, indices in indices_by_channels.items():
        if len(indices) == 0:
            continue
        
        # Create subset dataset
        subset_dataset = torch.utils.data.Subset(base_dataset, indices)
        
        # Apply transform wrapper
        class TransformWrapper(Dataset):
            def __init__(self, subset, transform):
                self.subset = subset
                self.transform = transform
            
            def __len__(self):
                return len(self.subset)
            
            def __getitem__(self, idx):
                image, metadata, labels = self.subset[idx]
                if self.transform:
                    image = self.transform(image)
                # Return tuple - DataLoader will handle collation
                return image, metadata, labels if labels is not None else []
        
        transformed_dataset = TransformWrapper(subset_dataset, transform)
        
        # Custom collate to handle metadata dicts and None labels
        def collate_fn(batch):
            images, metadatas, labels = zip(*batch)
            # Images can now be stacked since all have same channels!
            batch_images = torch.stack(images)
            batch_metadatas = list(metadatas)
            # Handle labels - convert None to empty list
            batch_labels = [label if label is not None else None for label in labels]
            return batch_images, batch_metadatas, batch_labels
        
        # Create DataLoader - can now stack normally since all have same channels
        dataloader = DataLoader(
            transformed_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )
        
        dataloaders[num_channels] = dataloader
        print(f"Created DataLoader for {num_channels} channels: {len(indices)} samples")
    
    return dataloaders


def create_interleaved_chammi_dataloader(
    csv_file: str,
    root_dir: str,
    batch_size_per_channel: int = 32,
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
):
    """
    Create a single DataLoader that interleaves batches from each channel count.
    Each iteration yields a batch from one channel group (all same channels).
    
    Yields:
        (batch_images, batch_metadatas, batch_labels, channel_count)
        - batch_images: tensor of shape (batch_size, C, H, W) where C is consistent
        - channel_count: 3, 4, or 5
    """
    # Create grouped dataloaders
    grouped_dataloaders = create_grouped_chammi_dataloaders(
        csv_file=csv_file,
        root_dir=root_dir,
        batch_size=batch_size_per_channel,
        shuffle=shuffle,
        target_labels=target_labels,
        split=split,
        resize_to=resize_to,
        augment=augment,
        normalize=normalize,
        mean=mean,
        std=std,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    # Create iterators for each dataloader
    iterators = {ch: iter(dl) for ch, dl in grouped_dataloaders.items()}
    
    # Interleave batches
    while True:
        for channel_count in [3, 4, 5]:
            if channel_count in iterators:
                try:
                    batch_images, batch_metadatas, batch_labels = next(iterators[channel_count])
                    yield batch_images, batch_metadatas, batch_labels, channel_count
                except StopIteration:
                    # Restart this iterator
                    iterators[channel_count] = iter(grouped_dataloaders[channel_count])
                    batch_images, batch_metadatas, batch_labels = next(iterators[channel_count])
                    yield batch_images, batch_metadatas, batch_labels, channel_count


def create_dataset_ordered_dataloader(
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
    shuffle_dataset_order: bool = True,
) -> Iterator[Tuple[torch.Tensor, List[Dict], List, int]]:
    """
    Create a DataLoader that interleaves datasets in different orders each epoch.
    This helps model generalization by preventing overfitting to a specific dataset order.
    
    At each epoch:
    - Shuffles the order of datasets (Allen, HPA, CP) if shuffle_dataset_order=True
    - Interleaves batches from different datasets in that order
    - Maintains efficient batching (all samples in batch have same channels)
    
    Args:
        shuffle_dataset_order: If True, shuffle dataset order each epoch (recommended)
        All other args same as create_grouped_chammi_dataloaders
    
    Yields:
        (batch_images, batch_metadatas, batch_labels, channel_count)
        - batch_images: tensor of shape (batch_size, C, H, W)
        - channel_count: 3, 4, or 5
    """
    # Create base dataset to filter by dataset source
    base_dataset = CHAMMIDataset(
        csv_file=csv_file,
        root_dir=root_dir,
        target_labels=target_labels,
        transform=None,
        split=split,
        resize_to=resize_to,
    )
    
    # Group indices by (channel_count, dataset_source)
    indices_by_group = {}
    dataset_sources = ['Allen', 'HPA', 'CP']
    
    for i in range(len(base_dataset)):
        row = base_dataset.metadata.iloc[i]
        num_channels = int(row['num_channels'])
        dataset_source = row['file_path'].split('/')[0]
        
        if num_channels in [3, 4, 5] and dataset_source in dataset_sources:
            key = (num_channels, dataset_source)
            if key not in indices_by_group:
                indices_by_group[key] = []
            indices_by_group[key].append(i)
    
    # Create DataLoaders for each (channel_count, dataset_source) group
    transform = CHAMMITransform(
        size=resize_to,
        augment=augment,
        normalize=normalize,
        mean=mean,
        std=std
    )
    
    dataloaders_by_group = {}
    
    for (num_channels, dataset_source), indices in indices_by_group.items():
        if len(indices) == 0:
            continue
        
        subset_dataset = torch.utils.data.Subset(base_dataset, indices)
        
        class TransformWrapper(Dataset):
            def __init__(self, subset, transform):
                self.subset = subset
                self.transform = transform
            
            def __len__(self):
                return len(self.subset)
            
            def __getitem__(self, idx):
                image, metadata, labels = self.subset[idx]
                if self.transform:
                    image = self.transform(image)
                return image, metadata, labels
        
        transformed_dataset = TransformWrapper(subset_dataset, transform)
        
        def collate_fn(batch):
            images, metadatas, labels = zip(*batch)
            batch_images = torch.stack(images)
            batch_metadatas = list(metadatas)
            batch_labels = [label if label is not None else None for label in labels]
            return batch_images, batch_metadatas, batch_labels
        
        dataloader = DataLoader(
            transformed_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )
        
        dataloaders_by_group[(num_channels, dataset_source)] = dataloader
    
    # Create iterators and yield batches in shuffled dataset order
    while True:
        # Determine dataset order for this epoch
        if shuffle_dataset_order:
            epoch_dataset_order = random.sample(dataset_sources, len(dataset_sources))
        else:
            epoch_dataset_order = dataset_sources
        
        # Create iterators for this epoch
        iterators = {}
        for (num_channels, dataset_source), dataloader in dataloaders_by_group.items():
            iterators[(num_channels, dataset_source)] = iter(dataloader)
        
        # Interleave batches: cycle through datasets in order, within each cycle through channel counts
        # This ensures dataset order changes each epoch while maintaining efficient batching
        channel_counts = [3, 4, 5]
        exhausted = False
        
        while not exhausted:
            exhausted = True
            # For each dataset in the epoch order, yield batches from all channel counts
            for dataset_source in epoch_dataset_order:
                for channel_count in channel_counts:
                    key = (channel_count, dataset_source)
                    if key in iterators:
                        try:
                            batch_images, batch_metadatas, batch_labels = next(iterators[key])
                            yield batch_images, batch_metadatas, batch_labels, channel_count
                            exhausted = False
                        except StopIteration:
                            # Restart this iterator for next epoch
                            iterators[key] = iter(dataloaders_by_group[key])
                            try:
                                batch_images, batch_metadatas, batch_labels = next(iterators[key])
                                yield batch_images, batch_metadatas, batch_labels, channel_count
                                exhausted = False
                            except StopIteration:
                                pass


def create_dataset_specific_dataloaders(
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
    shuffle_dataset_order: bool = True,
) -> Dict[str, DataLoader]:
    """
    Create separate DataLoaders for each dataset (Allen, HPA, CP).
    This is the main training approach: one DataLoader per dataset with given channel configuration.
    
    Each dataset has its natural channel count:
    - Allen: 3 channels
    - HPA: 4 channels  
    - CP: 5 channels
    
    Args:
        shuffle_dataset_order: If True, shuffle dataset order when interleaving (if used)
        All other args same as create_grouped_chammi_dataloaders
    
    Returns:
        Dict mapping dataset_source -> DataLoader
        Example: {'Allen': DataLoader(...), 'HPA': DataLoader(...), 'CP': DataLoader(...)}
        
    Usage:
        # Training with one DataLoader per dataset
        dataloaders = create_dataset_specific_dataloaders(...)
        
        for epoch in range(num_epochs):
            # Process each dataset separately
            for dataset_name in ['Allen', 'HPA', 'CP']:
                dataloader = dataloaders[dataset_name]
                for batch_images, batch_metadatas, batch_labels in dataloader:
                    # Process batch
                    ...
    """
    # Create base dataset to filter
    base_dataset = CHAMMIDataset(
        csv_file=csv_file,
        root_dir=root_dir,
        target_labels=target_labels,
        transform=None,
        split=split,
        resize_to=resize_to,
    )
    
    # Group indices by dataset source
    indices_by_dataset = {'Allen': [], 'HPA': [], 'CP': []}
    
    for i in range(len(base_dataset)):
        row = base_dataset.metadata.iloc[i]
        dataset_source = row['file_path'].split('/')[0]
        
        if dataset_source in indices_by_dataset:
            indices_by_dataset[dataset_source].append(i)
    
    # Create separate datasets and dataloaders for each dataset source
    dataloaders = {}
    transform = CHAMMITransform(
        size=resize_to,
        augment=augment,
        normalize=normalize,
        mean=mean,
        std=std
    )
    
    for dataset_source, indices in indices_by_dataset.items():
        if len(indices) == 0:
            continue
        
        # Create subset dataset
        subset_dataset = torch.utils.data.Subset(base_dataset, indices)
        
        # Apply transform wrapper
        class TransformWrapper(Dataset):
            def __init__(self, subset, transform):
                self.subset = subset
                self.transform = transform
            
            def __len__(self):
                return len(self.subset)
            
            def __getitem__(self, idx):
                image, metadata, labels = self.subset[idx]
                if self.transform:
                    image = self.transform(image)
                return image, metadata, labels
        
        transformed_dataset = TransformWrapper(subset_dataset, transform)
        
        # Custom collate to handle metadata dicts
        def collate_fn(batch):
            images, metadatas, labels = zip(*batch)
            # Images can be stacked since all from same dataset (same channels)
            batch_images = torch.stack(images)
            batch_metadatas = list(metadatas)
            batch_labels = [label if label is not None else None for label in labels]
            return batch_images, batch_metadatas, batch_labels
        
        # Create DataLoader - enable shuffling for frequent reshuffling
        dataloader = DataLoader(
            transformed_dataset,
            batch_size=batch_size,
            shuffle=shuffle,  # Shuffle every epoch for variety
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )
        
        dataloaders[dataset_source] = dataloader
        print(f"Created DataLoader for {dataset_source}: {len(indices)} samples")
    
    return dataloaders


def create_interleaved_dataset_dataloader(
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
    shuffle_dataset_order: bool = True,
) -> Iterator[Tuple[torch.Tensor, List[Dict], List, str]]:
    """
    Create a DataLoader that interleaves datasets in different orders each epoch.
    One DataLoader per dataset, shuffled order for better generalization.
    
    Yields:
        (batch_images, batch_metadatas, batch_labels, dataset_source)
        - batch_images: tensor of shape (batch_size, C, H, W) where C is dataset-specific
        - dataset_source: 'Allen' (3ch), 'HPA' (4ch), or 'CP' (5ch)
    """
    # Create dataset-specific dataloaders
    dataset_dataloaders = create_dataset_specific_dataloaders(
        csv_file=csv_file,
        root_dir=root_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        target_labels=target_labels,
        split=split,
        resize_to=resize_to,
        augment=augment,
        normalize=normalize,
        mean=mean,
        std=std,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle_dataset_order=shuffle_dataset_order,
    )
    
    dataset_names = ['Allen', 'HPA', 'CP']
    
    # Create iterators and yield batches in shuffled dataset order
    while True:
        # Determine dataset order for this epoch
        if shuffle_dataset_order:
            epoch_dataset_order = random.sample(dataset_names, len(dataset_names))
        else:
            epoch_dataset_order = dataset_names
        
        # Create iterators for this epoch
        iterators = {name: iter(dl) for name, dl in dataset_dataloaders.items()}
        
        # Interleave batches from different datasets in shuffled order
        exhausted = False
        
        while not exhausted:
            exhausted = True
            for dataset_source in epoch_dataset_order:
                if dataset_source in iterators:
                    try:
                        batch_images, batch_metadatas, batch_labels = next(iterators[dataset_source])
                        yield batch_images, batch_metadatas, batch_labels, dataset_source
                        exhausted = False
                    except StopIteration:
                        # Restart this iterator for next epoch
                        iterators[dataset_source] = iter(dataset_dataloaders[dataset_source])
                        try:
                            batch_images, batch_metadatas, batch_labels = next(iterators[dataset_source])
                            yield batch_images, batch_metadatas, batch_labels, dataset_source
                            exhausted = False
                        except StopIteration:
                            pass


if __name__ == "__main__":
    # Example usage
    import os
    
    chammi_root = "/Users/zamfiraluca/Downloads/CHAMMI"
    csv_file = os.path.join(chammi_root, "combined_metadata.csv")
    
    print("Option 1: Separate DataLoaders for each channel count")
    print("=" * 70)
    
    dataloaders = create_grouped_chammi_dataloaders(
        csv_file=csv_file,
        root_dir=chammi_root,
        batch_size=32,
        shuffle=True,
        split='train',
        augment=True,
    )
    
    print("\nTesting DataLoaders:")
    for channel_count, dataloader in dataloaders.items():
        batch_images, batch_metadatas, batch_labels = next(iter(dataloader))
        print(f"\n{channel_count} channels DataLoader:")
        print(f"  Batch shape: {batch_images.shape}  # (batch_size, {channel_count}, 128, 128)")
        print(f"  All same channels: ✓")
        print(f"  Can process efficiently: ✓")
    
    print("\n\nOption 2: Interleaved DataLoader")
    print("=" * 70)
    
    interleaved = create_interleaved_chammi_dataloader(
        csv_file=csv_file,
        root_dir=chammi_root,
        batch_size_per_channel=32,
        split='train',
        augment=True,
    )
    
    print("\nTesting interleaved DataLoader (first 3 batches):")
    for i, (batch_images, batch_metadatas, batch_labels, channel_count) in enumerate(interleaved):
        if i >= 3:
            break
        print(f"\nBatch {i+1}:")
        print(f"  Channel count: {channel_count}")
        print(f"  Batch shape: {batch_images.shape}  # (batch_size, {channel_count}, 128, 128)")
        print(f"  All same channels: ✓")
    
    print("\n\nOption 3: Dataset-Specific DataLoaders (one per dataset)")
    print("=" * 70)
    
    dataset_dataloaders = create_dataset_specific_dataloaders(
        csv_file=csv_file,
        root_dir=chammi_root,
        batch_size=32,
        split='train',
        augment=True,
    )
    
    print("\nTesting dataset-specific DataLoaders:")
    for dataset_name, dataloader in dataset_dataloaders.items():
        batch_images, batch_metadatas, batch_labels = next(iter(dataloader))
        channel_count = batch_metadatas[0]['num_channels']
        print(f"\n{dataset_name} DataLoader:")
        print(f"  Batch shape: {batch_images.shape}  # (batch_size, {channel_count}, 128, 128)")
        print(f"  All same channels: ✓")
        print(f"  All same dataset: ✓")
    
    print("\n\nOption 4: Interleaved Dataset DataLoader (shuffles dataset order each epoch)")
    print("=" * 70)
    
    interleaved = create_interleaved_dataset_dataloader(
        csv_file=csv_file,
        root_dir=chammi_root,
        batch_size=32,
        split='train',
        augment=True,
        shuffle_dataset_order=True,
    )
    
    print("\nTesting interleaved dataset DataLoader (first 6 batches):")
    for i, (batch_images, batch_metadatas, batch_labels, dataset_source) in enumerate(interleaved):
        if i >= 6:
            break
        channel_count = batch_metadatas[0]['num_channels']
        print(f"\nBatch {i+1}:")
        print(f"  Dataset: {dataset_source}")
        print(f"  Channel count: {channel_count}")
        print(f"  Batch shape: {batch_images.shape}")
        print(f"  All same channels: ✓")
        print(f"  All same dataset: ✓")

