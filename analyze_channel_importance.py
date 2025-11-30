"""
Channel importance analysis script.
Computes and visualizes channel importance through ablation studies.
"""

import torch
import argparse
import json
import os

from config import Config
from channel_adaptive_pipeline.models.early_fusion_vit import EarlyFusionViT
from channel_adaptive_pipeline.chammi_grouped_dataloader import create_grouped_chammi_dataloaders
from channel_adaptive_pipeline.model_utils import (
    load_model_checkpoint,
    get_num_classes_from_metadata,
    get_class_to_idx_mapping,
)
from channel_adaptive_pipeline.channel_importance import (
    compute_channel_importance,
    visualize_channel_importance,
)


def main():
    parser = argparse.ArgumentParser(description='Analyze channel importance')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None, help='Path to config file (optional)')
    parser.add_argument('--output_dir', type=str, default='./channel_importance', help='Output directory')
    parser.add_argument('--split', type=str, default='val', help='Split to evaluate on')
    parser.add_argument('--metric', type=str, default='accuracy', choices=['accuracy', 'macro_f1'],
                       help='Metric to use for importance computation')
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
    
    # Compute channel importance
    print(f"\nComputing channel importance (metric: {args.metric})...")
    importance_scores = compute_channel_importance(
        model=model,
        dataloaders_dict=dataloaders,
        device=device,
        num_classes=num_classes,
        class_to_idx=class_to_idx,
        metric=args.metric,
    )
    
    # Print results
    print("\nChannel Importance Scores:")
    for channel_count in sorted(importance_scores.keys()):
        print(f"\n{channel_count} channels:")
        scores = importance_scores[channel_count]
        for channel_idx in sorted(scores.keys()):
            print(f"  Channel {channel_idx}: {scores[channel_idx]:.4f}")
    
    # Visualize
    print(f"\nGenerating visualizations...")
    visualize_channel_importance(importance_scores, args.output_dir, metric=args.metric)
    
    # Save results
    results_path = os.path.join(args.output_dir, 'channel_importance.json')
    # Convert to JSON-serializable format
    json_results = {
        str(ch_count): {str(ch_idx): float(score) for ch_idx, score in scores.items()}
        for ch_count, scores in importance_scores.items()
    }
    
    os.makedirs(args.output_dir, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()

