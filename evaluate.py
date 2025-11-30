"""
Evaluation script for channel-adaptive ViT models.
Evaluates models on SD (in-distribution) and OOD (out-of-distribution) splits.
"""

import torch
import argparse
import json
import os
import pandas as pd
try:
    from tabulate import tabulate
except ImportError:
    # Fallback if tabulate is not installed
    def tabulate(data, headers, tablefmt="grid"):
        # Simple table printing without tabulate
        print(" | ".join(headers))
        print("-" * (sum(len(h) for h in headers) + 3 * len(headers)))
        for row in data:
            print(" | ".join(str(x) for x in row))

from config import Config
from channel_adaptive_pipeline.models.early_fusion_vit import EarlyFusionViT
from channel_adaptive_pipeline.model_utils import (
    load_model_checkpoint,
    get_num_classes_from_metadata,
    get_class_to_idx_mapping,
)
from channel_adaptive_pipeline.evaluation import (
    evaluate_sd_ood,
    evaluate_per_dataset,
)


def print_results_table(results: dict, split_name: str):
    """
    Print evaluation results in a formatted table.
    
    Args:
        results: Results dictionary from evaluate_sd_ood
        split_name: Name of the split ('sd' or 'ood')
    """
    table_data = []
    
    if split_name == 'sd':
        for channel_count in sorted(results.keys()):
            metrics = results[channel_count]
            table_data.append([
                channel_count,
                f"{metrics['accuracy']:.4f}",
                f"{metrics['macro_f1']:.4f}",
            ])
        
        headers = ["Channel Count", "Accuracy", "Macro-F1"]
        print(f"\n{split_name.upper()} Results:")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    else:  # OOD splits
        for ood_split_name in sorted(results.keys()):
            ood_results = results[ood_split_name]
            for channel_count in sorted(ood_results.keys()):
                metrics = ood_results[channel_count]
                table_data.append([
                    ood_split_name,
                    channel_count,
                    f"{metrics['accuracy']:.4f}",
                    f"{metrics['macro_f1']:.4f}",
                ])
        
        headers = ["OOD Split", "Channel Count", "Accuracy", "Macro-F1"]
        print(f"\n{split_name.upper()} Results:")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))


def print_per_dataset_results(dataset_results: dict):
    """Print per-dataset evaluation results."""
    table_data = []
    
    for dataset_source in sorted(dataset_results.keys()):
        metrics = dataset_results[dataset_source]
        table_data.append([
            dataset_source,
            metrics['num_samples'],
            f"{metrics['accuracy']:.4f}",
            f"{metrics['macro_f1']:.4f}",
        ])
    
    headers = ["Dataset", "Num Samples", "Accuracy", "Macro-F1"]
    print("\nPer-Dataset Results:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


def main():
    parser = argparse.ArgumentParser(description='Evaluate channel-adaptive ViT model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None, help='Path to config file (optional)')
    parser.add_argument('--output', type=str, default=None, help='Path to save results JSON')
    args = parser.parse_args()
    
    # Load config
    config = Config()
    
    # Set device
    device = torch.device(config.device if config.device == 'cpu' else f'cuda:{config.gpu_id}')
    print(f"Using device: {device}")
    
    # Auto-detect number of classes
    print("Auto-detecting number of classes from metadata...")
    num_classes = get_num_classes_from_metadata(
        csv_file=config.csv_file,
        root_dir=config.root_dir,
        target_labels=config.target_labels,
    )
    print(f"Found {num_classes} classes")
    
    # Get class to index mapping
    class_to_idx = get_class_to_idx_mapping(
        csv_file=config.csv_file,
        root_dir=config.root_dir,
        target_labels=config.target_labels,
    )
    
    # Initialize model
    print(f"Initializing {config.model_type} ViT-{config.vit_size}...")
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
    checkpoint_info = load_model_checkpoint(model, args.checkpoint, device)
    print(f"Loaded checkpoint from epoch {checkpoint_info['epoch']}")
    
    # Evaluate on SD and OOD splits
    print("\nEvaluating model...")
    results = evaluate_sd_ood(
        model=model,
        csv_file=config.csv_file,
        root_dir=config.root_dir,
        config=config,
        device=device,
        class_to_idx=class_to_idx,
        num_classes=num_classes,
    )
    
    # Print results
    print_results_table(results['sd'], 'sd')
    
    for ood_split_name in sorted(results['ood'].keys()):
        print_results_table(results['ood'][ood_split_name], f'ood_{ood_split_name}')
    
    # Evaluate per-dataset (on SD split)
    print("\nComputing per-dataset metrics...")
    from channel_adaptive_pipeline.chammi_grouped_dataloader import create_grouped_chammi_dataloaders
    
    sd_dataloaders = create_grouped_chammi_dataloaders(
        csv_file=config.csv_file,
        root_dir=config.root_dir,
        batch_size=config.batch_size,
        shuffle=False,
        target_labels=config.target_labels,
        split=config.sd_split,
        resize_to=config.img_size,
        augment=False,
        normalize=config.normalize,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    
    dataset_results = evaluate_per_dataset(
        model=model,
        dataloaders_dict=sd_dataloaders,
        device=device,
        num_classes=num_classes,
        class_to_idx=class_to_idx,
    )
    
    print_per_dataset_results(dataset_results)
    
    # Save results
    if args.output:
        output_dict = {
            'checkpoint': args.checkpoint,
            'num_classes': num_classes,
            'sd_results': {str(k): v for k, v in results['sd'].items()},
            'ood_results': {
                ood_name: {str(k): v for k, v in ood_results.items()}
                for ood_name, ood_results in results['ood'].items()
            },
            'per_dataset_results': dataset_results,
        }
        
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(output_dict, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()

