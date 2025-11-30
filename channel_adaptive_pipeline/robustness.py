"""
Robustness evaluation for channel-adaptive models.
Tests model robustness to channel manipulations (missing channels, shuffled channels).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score


def evaluate_baseline(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    class_to_idx: Dict[str, int],
) -> Dict[str, float]:
    """
    Evaluate model baseline performance.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader
        device: Device to evaluate on
        num_classes: Number of classes
        class_to_idx: Mapping from class names to indices
    
    Returns:
        Dictionary with accuracy and macro_f1
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_images, batch_metadatas, batch_labels in dataloader:
            batch_images = batch_images.to(device)
            
            # Convert labels to indices
            if isinstance(batch_labels[0], str):
                batch_labels_tensor = torch.tensor([
                    class_to_idx[label] if label in class_to_idx else 0
                    for label in batch_labels
                ], dtype=torch.long, device=device)
            else:
                batch_labels_tensor = torch.tensor(batch_labels, dtype=torch.long, device=device)
            
            # Forward pass
            num_channels = batch_images.size(1)
            logits = model(batch_images, num_channels=num_channels)
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_labels_tensor.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_predictions)
    macro_f1 = f1_score(all_labels, all_predictions, average='macro')
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
    }


def evaluate_missing_channels(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    class_to_idx: Dict[str, int],
) -> Dict[int, Dict[str, float]]:
    """
    Evaluate model with missing channels (zeroed out one at a time).
    
    Args:
        model: PyTorch model
        dataloader: DataLoader
        device: Device to evaluate on
        num_classes: Number of classes
        class_to_idx: Mapping from class names to indices
    
    Returns:
        Dictionary mapping channel_idx -> metrics dict with performance drop
    """
    # Get baseline performance
    baseline = evaluate_baseline(model, dataloader, device, num_classes, class_to_idx)
    
    model.eval()
    results = {}
    
    # Get channel count from first batch
    first_batch = next(iter(dataloader))
    num_channels = first_batch[0].size(1)
    
    # Evaluate with each channel missing
    for channel_idx in range(num_channels):
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_images, batch_metadatas, batch_labels in dataloader:
                batch_images = batch_images.to(device).clone()
                
                # Zero out the specified channel
                batch_images[:, channel_idx, :, :] = 0.0
                
                # Convert labels to indices
                if isinstance(batch_labels[0], str):
                    batch_labels_tensor = torch.tensor([
                        class_to_idx[label] if label in class_to_idx else 0
                        for label in batch_labels
                    ], dtype=torch.long, device=device)
                else:
                    batch_labels_tensor = torch.tensor(batch_labels, dtype=torch.long, device=device)
                
                # Forward pass
                logits = model(batch_images, num_channels=num_channels)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch_labels_tensor.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        accuracy = accuracy_score(all_labels, all_predictions)
        macro_f1 = f1_score(all_labels, all_predictions, average='macro')
        
        # Compute performance drop
        accuracy_drop = baseline['accuracy'] - accuracy
        f1_drop = baseline['macro_f1'] - macro_f1
        
        results[channel_idx] = {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'accuracy_drop': accuracy_drop,
            'f1_drop': f1_drop,
        }
    
    return results


def evaluate_shuffled_channels(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    class_to_idx: Dict[str, int],
    num_permutations: int = 10,
) -> Dict[str, float]:
    """
    Evaluate model with shuffled channel order.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader
        device: Device to evaluate on
        num_classes: Number of classes
        class_to_idx: Mapping from class names to indices
        num_permutations: Number of random permutations to test
    
    Returns:
        Dictionary with average metrics across permutations
    """
    model.eval()
    
    # Get channel count from first batch
    first_batch = next(iter(dataloader))
    num_channels = first_batch[0].size(1)
    
    all_accuracies = []
    all_f1s = []
    
    # Test multiple random permutations
    for perm_idx in range(num_permutations):
        # Generate random permutation
        perm = torch.randperm(num_channels)
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_images, batch_metadatas, batch_labels in dataloader:
                batch_images = batch_images.to(device).clone()
                
                # Shuffle channels according to permutation
                batch_images = batch_images[:, perm, :, :]
                
                # Convert labels to indices
                if isinstance(batch_labels[0], str):
                    batch_labels_tensor = torch.tensor([
                        class_to_idx[label] if label in class_to_idx else 0
                        for label in batch_labels
                    ], dtype=torch.long, device=device)
                else:
                    batch_labels_tensor = torch.tensor(batch_labels, dtype=torch.long, device=device)
                
                # Forward pass
                logits = model(batch_images, num_channels=num_channels)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch_labels_tensor.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        accuracy = accuracy_score(all_labels, all_predictions)
        macro_f1 = f1_score(all_labels, all_predictions, average='macro')
        
        all_accuracies.append(accuracy)
        all_f1s.append(macro_f1)
    
    return {
        'accuracy': np.mean(all_accuracies),
        'accuracy_std': np.std(all_accuracies),
        'macro_f1': np.mean(all_f1s),
        'macro_f1_std': np.std(all_f1s),
    }

