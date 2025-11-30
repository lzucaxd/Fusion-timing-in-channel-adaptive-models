"""
Model utilities for checkpointing and model management.
"""

import torch
import os
import pandas as pd
from typing import Optional, Dict, Any


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    metrics: Dict[str, float],
    save_path: str,
    is_best: bool = False,
):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer instance
        scheduler: Scheduler instance (optional)
        epoch: Current epoch number
        metrics: Dictionary of metrics to save
        save_path: Path to save checkpoint
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'is_best': is_best,
    }
    
    # Add scheduler state if provided
    if scheduler is not None:
        if hasattr(scheduler, 'state_dict'):
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Save checkpoint
    torch.save(checkpoint, save_path)
    
    if is_best:
        # Also save as best model
        best_path = save_path.replace('.pth', '_best.pth').replace('_latest.pth', '_best.pth')
        if '_best' not in save_path:
            torch.save(checkpoint, best_path)


def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: str = 'cuda',
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
    
    Returns:
        Dictionary containing checkpoint information (epoch, metrics, etc.)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        if hasattr(scheduler, 'load_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'metrics': checkpoint.get('metrics', {}),
        'is_best': checkpoint.get('is_best', False),
    }


def get_num_classes_from_metadata(csv_file: str, root_dir: str, target_labels: str = 'Label') -> int:
    """
    Auto-detect number of classes from enriched metadata files.
    
    Args:
        csv_file: Path to combined_metadata.csv
        root_dir: Root directory of CHAMMI dataset
        target_labels: Column name in enriched_meta.csv to use for labels
    
    Returns:
        Number of unique classes
    """
    import os
    
    # Load combined metadata to get dataset sources
    combined_meta = pd.read_csv(csv_file)
    
    all_labels = set()
    
    # Check each dataset's enriched metadata
    for dataset_source in ['Allen', 'HPA', 'CP']:
        enriched_meta_path = os.path.join(root_dir, dataset_source, 'enriched_meta.csv')
        
        if os.path.exists(enriched_meta_path):
            enriched_df = pd.read_csv(enriched_meta_path)
            
            if target_labels in enriched_df.columns:
                labels = enriched_df[target_labels].dropna().unique()
                all_labels.update(labels)
    
    num_classes = len(all_labels)
    
    if num_classes == 0:
        raise ValueError(f"No labels found in enriched metadata. Check target_labels='{target_labels}'")
    
    return num_classes


def get_class_to_idx_mapping(csv_file: str, root_dir: str, target_labels: str = 'Label') -> Dict[str, int]:
    """
    Get mapping from class names to indices.
    
    Args:
        csv_file: Path to combined_metadata.csv
        root_dir: Root directory of CHAMMI dataset
        target_labels: Column name in enriched_meta.csv to use for labels
    
    Returns:
        Dictionary mapping class name -> class index
    """
    import os
    
    all_labels = set()
    
    # Collect all unique labels
    for dataset_source in ['Allen', 'HPA', 'CP']:
        enriched_meta_path = os.path.join(root_dir, dataset_source, 'enriched_meta.csv')
        
        if os.path.exists(enriched_meta_path):
            enriched_df = pd.read_csv(enriched_meta_path)
            
            if target_labels in enriched_df.columns:
                labels = enriched_df[target_labels].dropna().unique()
                all_labels.update(labels)
    
    # Create sorted mapping
    sorted_labels = sorted(all_labels)
    class_to_idx = {label: idx for idx, label in enumerate(sorted_labels)}
    
    return class_to_idx

