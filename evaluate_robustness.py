"""
Robustness evaluation script for channel-adaptive ViT models.
Tests model robustness to channel manipulations.
"""

import torch
import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np

from config import Config
from channel_adaptive_pipeline.models.early_fusion_vit import EarlyFusionViT
from channel_adaptive_pipeline.chammi_grouped_dataloader import create_grouped_chammi_dataloaders
from channel_adaptive_pipeline.model_utils import (
    load_model_checkpoint,
    get_num_classes_from_metadata,
    get_class_to_idx_mapping,
)
from channel_adaptive_pipeline.robustness import (
    evaluate_baseline,
    evaluate_missing_channels,
    evaluate_shuffled_channels,
)


def plot_channel_importance(missing_channel_results: dict, save_path: str, channel_count: int):
    """
    Plot channel importance (performance drop when channel is missing).
    
    Args:
        missing_channel_results: Results from evaluate_missing_channels
        save_path: Path to save plot
        channel_count: Number of channels
    """
    channel_indices = sorted(missing_channel_results.keys())
    accuracy_drops = [missing_channel_results[idx]['accuracy_drop'] for idx in channel_indices]
    f1_drops = [missing_channel_results[idx]['f1_drop'] for idx in channel_indices]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy drop plot
    ax1.bar(channel_indices, accuracy_drops, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Channel Index')
    ax1.set_ylabel('Accuracy Drop')
    ax1.set_title(f'Channel Importance (Accuracy Drop) - {channel_count} Channels')
    ax1.set_xticks(channel_indices)
    ax1.grid(axis='y', alpha=0.3)
    
    # F1 drop plot
    ax2.bar(channel_indices, f1_drops, color='coral', alpha=0.7)
    ax2.set_xlabel('Channel Index')
    ax2.set_ylabel('Macro-F1 Drop')
    ax2.set_title(f'Channel Importance (F1 Drop) - {channel_count} Channels')
    ax2.set_xticks(channel_indices)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved channel importance plot to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate model robustness')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None, help='Path to config file (optional)')
    parser.add_argument('--output_dir', type=str, default='./robustness_results', help='Output directory')
    parser.add_argument('--split', type=str, default='val', help='Split to evaluate on')
    args = parser.parse_args()
    
    # Load config
    config = Config()
    
    # Set device
    device = torch.device(config.device if config.device == 'cpu' else f'cuda:{config.gpu_id}')
    print(f"Using device: {device}")
    
    # Auto-detect number of classes
    num_classes = get_num_classes_from_metadata(
        csv_file=config.csv_file,
        root_dir=config.root_dir,
        target_labels=config.target_labels,
    )
    
    # Get class to index mapping
    class_to_idx = get_class_to_idx_mapping(
        csv_file=config.csv_file,
        root_dir=config.root_dir,
        target_labels=config.target_labels,
    )
    
    # Initialize model
    if config.model_type == 'early_fusion':
        model = EarlyFusionViT(
            num_classes=num_classes,
            img_size=config.img_size,
            patch_size=config.patch_size,
        )
    else:
        raise ValueError(f"Model type {config.model_type} not implemented yet")
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    load_model_checkpoint(model, args.checkpoint, device)
    
    # Create dataloaders
    print(f"Creating dataloaders for split: {args.split}")
    dataloaders = create_grouped_chammi_dataloaders(
        csv_file=config.csv_file,
        root_dir=config.root_dir,
        batch_size=config.batch_size,
        shuffle=False,
        target_labels=config.target_labels,
        split=args.split,
        resize_to=config.img_size,
        augment=False,
        normalize=config.normalize,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate robustness for each channel count
    all_results = {}
    
    for channel_count in sorted(dataloaders.keys()):
        print(f"\nEvaluating robustness for {channel_count} channels...")
        dataloader = dataloaders[channel_count]
        
        # Baseline
        print("  Computing baseline...")
        baseline = evaluate_baseline(model, dataloader, device, num_classes, class_to_idx)
        print(f"    Baseline Accuracy: {baseline['accuracy']:.4f}, F1: {baseline['macro_f1']:.4f}")
        
        # Missing channels
        print("  Testing missing channels...")
        missing_results = evaluate_missing_channels(
            model, dataloader, device, num_classes, class_to_idx
        )
        
        # Shuffled channels
        print("  Testing shuffled channels...")
        shuffled_results = evaluate_shuffled_channels(
            model, dataloader, device, num_classes, class_to_idx, num_permutations=10
        )
        print(f"    Shuffled Accuracy: {shuffled_results['accuracy']:.4f} ± {shuffled_results['accuracy_std']:.4f}")
        print(f"    Shuffled F1: {shuffled_results['macro_f1']:.4f} ± {shuffled_results['macro_f1_std']:.4f}")
        
        # Store results
        all_results[channel_count] = {
            'baseline': baseline,
            'missing_channels': {str(k): v for k, v in missing_results.items()},
            'shuffled_channels': shuffled_results,
        }
        
        # Plot channel importance
        plot_path = os.path.join(args.output_dir, f'channel_importance_ch{channel_count}.png')
        plot_channel_importance(missing_results, plot_path, channel_count)
    
    # Save results
    results_path = os.path.join(args.output_dir, 'robustness_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()

