"""
Channel importance analysis for channel-adaptive models.
Computes importance scores for each channel through ablation studies.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score


def compute_channel_importance(
    model: nn.Module,
    dataloaders_dict: Dict[int, DataLoader],
    device: torch.device,
    num_classes: int,
    class_to_idx: Dict[str, int],
    metric: str = 'accuracy',
) -> Dict[int, Dict[int, float]]:
    """
    Compute channel importance through per-channel ablation.
    
    For each channel count and each channel index:
    - Zero out that channel
    - Evaluate performance
    - Compute performance drop vs baseline
    
    Args:
        model: PyTorch model
        dataloaders_dict: Dictionary mapping channel_count -> DataLoader
        device: Device to evaluate on
        num_classes: Number of classes
        class_to_idx: Mapping from class names to indices
        metric: Metric to use for importance ('accuracy' or 'macro_f1')
    
    Returns:
        Dictionary mapping channel_count -> {channel_idx: importance_score}
        Importance score is the performance drop when channel is removed
    """
    model.eval()
    
    results = {}
    
    for channel_count, dataloader in dataloaders_dict.items():
        # Get baseline performance
        baseline_predictions = []
        baseline_labels = []
        
        with torch.no_grad():
            for batch_images, batch_metadatas, batch_labels in dataloader:
                batch_images = batch_images.to(device)
                
                if isinstance(batch_labels[0], str):
                    batch_labels_tensor = torch.tensor([
                        class_to_idx[label] if label in class_to_idx else 0
                        for label in batch_labels
                    ], dtype=torch.long, device=device)
                else:
                    batch_labels_tensor = torch.tensor(batch_labels, dtype=torch.long, device=device)
                
                logits = model(batch_images)
                predictions = torch.argmax(logits, dim=1)
                
                baseline_predictions.extend(predictions.cpu().numpy())
                baseline_labels.extend(batch_labels_tensor.cpu().numpy())
        
        baseline_predictions = np.array(baseline_predictions)
        baseline_labels = np.array(baseline_labels)
        
        if metric == 'accuracy':
            baseline_score = accuracy_score(baseline_labels, baseline_predictions)
        else:
            baseline_score = f1_score(baseline_labels, baseline_predictions, average='macro')
        
        # Ablate each channel
        channel_importance = {}
        
        for channel_idx in range(channel_count):
            ablated_predictions = []
            ablated_labels = []
            
            with torch.no_grad():
                for batch_images, batch_metadatas, batch_labels in dataloader:
                    batch_images = batch_images.to(device).clone()
                    
                    # Zero out the channel
                    batch_images[:, channel_idx, :, :] = 0.0
                    
                    if isinstance(batch_labels[0], str):
                        batch_labels_tensor = torch.tensor([
                            class_to_idx[label] if label in class_to_idx else 0
                            for label in batch_labels
                        ], dtype=torch.long, device=device)
                    else:
                        batch_labels_tensor = torch.tensor(batch_labels, dtype=torch.long, device=device)
                    
                    logits = model(batch_images)
                    predictions = torch.argmax(logits, dim=1)
                    
                    ablated_predictions.extend(predictions.cpu().numpy())
                    ablated_labels.extend(batch_labels_tensor.cpu().numpy())
            
            ablated_predictions = np.array(ablated_predictions)
            ablated_labels = np.array(ablated_labels)
            
            if metric == 'accuracy':
                ablated_score = accuracy_score(ablated_labels, ablated_predictions)
            else:
                ablated_score = f1_score(ablated_labels, ablated_predictions, average='macro')
            
            # Importance = performance drop
            importance = baseline_score - ablated_score
            channel_importance[channel_idx] = importance
        
        results[channel_count] = channel_importance
    
    return results


def visualize_channel_importance(
    importance_scores: Dict[int, Dict[int, float]],
    save_path: str,
    metric: str = 'accuracy',
):
    """
    Visualize channel importance as bar plots.
    
    Args:
        importance_scores: Results from compute_channel_importance
        save_path: Directory to save plots
        metric: Metric name for plot titles
    """
    import os
    import matplotlib.pyplot as plt
    
    os.makedirs(save_path, exist_ok=True)
    
    for channel_count, scores in importance_scores.items():
        channel_indices = sorted(scores.keys())
        importance_values = [scores[idx] for idx in channel_indices]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(channel_indices, importance_values, color='steelblue', alpha=0.7)
        
        # Add value labels on bars
        for bar, val in zip(bars, importance_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}',
                   ha='center', va='bottom' if val >= 0 else 'top')
        
        ax.set_xlabel('Channel Index')
        ax.set_ylabel(f'Importance ({metric} drop)')
        ax.set_title(f'Channel Importance - {channel_count} Channels')
        ax.set_xticks(channel_indices)
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        plot_path = os.path.join(save_path, f'channel_importance_ch{channel_count}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved channel importance plot to: {plot_path}")

