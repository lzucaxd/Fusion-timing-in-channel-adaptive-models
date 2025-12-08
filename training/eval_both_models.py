"""
Evaluate both mean pooling and attention pooling models.
"""

import torch
import torch.nn as nn
import argparse
from pathlib import Path
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.boc_vit import BoCViT
from channel_adaptive_pipeline.chammi_grouped_dataloader import create_dataset_specific_dataloaders
from training.eval_boc import extract_embeddings, evaluate_1nn


def create_label_encoder(root_dir, target_labels="Label"):
    """Create label encoder from enriched metadata files."""
    all_labels = set()
    for dataset in ['Allen', 'HPA', 'CP']:
        meta_file = os.path.join(root_dir, dataset, 'enriched_meta.csv')
        if os.path.exists(meta_file):
            try:
                df = pd.read_csv(meta_file, low_memory=False)
                if target_labels in df.columns:
                    labels = df[target_labels].dropna().unique()
                    all_labels.update([str(l) for l in labels])
            except Exception as e:
                print(f"  {dataset}: Error reading - {e}")
    
    unique_labels = sorted(list(all_labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    return label_to_idx, unique_labels


def encode_labels(labels, label_to_idx):
    """Encode string labels to integer indices."""
    if isinstance(labels, list):
        encoded = []
        for label in labels:
            if label is None:
                encoded.append(0)
            elif isinstance(label, dict):
                label = list(label.values())[0] if label else None
                encoded.append(label_to_idx.get(str(label), 0))
            else:
                label_str = str(label)
                encoded.append(label_to_idx.get(label_str, 0))
        return np.array(encoded)
    return labels


def evaluate_model_checkpoint(
    checkpoint_path,
    model_config,
    dataloaders_train,
    dataloaders_test,
    device,
    label_to_idx,
    model_name
):
    """Evaluate a model checkpoint."""
    print(f"\n{'='*70}")
    print(f"Evaluating {model_name}")
    print(f"{'='*70}")
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = BoCViT(**model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    results = {}
    
    # Evaluate on each dataset
    for dataset_name in ['Allen', 'HPA', 'CP']:
        if dataset_name not in dataloaders_train or dataset_name not in dataloaders_test:
            continue
        
        print(f"\n  Evaluating on {dataset_name} dataset...")
        
        # Extract train embeddings
        print("    Extracting training embeddings...")
        train_loader = dataloaders_train[dataset_name]
        train_embeddings_list = []
        train_labels_list = []
        
        with torch.no_grad():
            for images, metadatas, labels in train_loader:
                images = images.to(device)
                embeddings = model(images)
                train_embeddings_list.append(embeddings.cpu().numpy())
                train_labels_list.extend(labels)
        
        train_embeddings = np.concatenate(train_embeddings_list, axis=0)
        train_labels = encode_labels(train_labels_list, label_to_idx)
        
        # Extract test embeddings
        print("    Extracting test embeddings...")
        test_loader = dataloaders_test[dataset_name]
        test_embeddings_list = []
        test_labels_list = []
        
        with torch.no_grad():
            for images, metadatas, labels in test_loader:
                images = images.to(device)
                embeddings = model(images)
                test_embeddings_list.append(embeddings.cpu().numpy())
                test_labels_list.extend(labels)
        
        test_embeddings = np.concatenate(test_embeddings_list, axis=0)
        test_labels = encode_labels(test_labels_list, label_to_idx)
        
        # Evaluate with 1-NN
        print("    Computing 1-NN metrics...")
        metrics = evaluate_1nn(train_embeddings, train_labels, test_embeddings, test_labels, metric="cosine")
        
        results[dataset_name] = metrics
        print(f"    Accuracy: {metrics['accuracy']:.4f}")
        print(f"    Macro-F1: {metrics['macro_f1']:.4f}")
    
    # Overall metrics (weighted average)
    if len(results) > 0:
        total_samples = sum(len(dataloaders_test[name]) * dataloaders_test[name].batch_size 
                          for name in results.keys())
        weighted_acc = sum(results[name]['accuracy'] * len(dataloaders_test[name]) * dataloaders_test[name].batch_size 
                          for name in results.keys()) / total_samples if total_samples > 0 else 0
        weighted_f1 = sum(results[name]['macro_f1'] * len(dataloaders_test[name]) * dataloaders_test[name].batch_size 
                         for name in results.keys()) / total_samples if total_samples > 0 else 0
        
        results['Overall'] = {
            'accuracy': weighted_acc,
            'macro_f1': weighted_f1
        }
        print(f"\n  Overall Accuracy: {weighted_acc:.4f}")
        print(f"  Overall Macro-F1: {weighted_f1:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate both mean and attention models")
    
    parser.add_argument("--csv-file", type=str, required=True)
    parser.add_argument("--root-dir", type=str, required=True)
    parser.add_argument("--target-labels", type=str, default="Label")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    
    # Model config
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--metric-embed-dim", type=int, default=256)
    
    # Checkpoint paths
    parser.add_argument("--mean-checkpoint", type=str, default="./checkpoints/boc_mean_proxynca/checkpoint_latest.pth")
    parser.add_argument("--attn-checkpoint", type=str, default="./checkpoints/boc_attn_proxynca/checkpoint_latest.pth")
    
    args = parser.parse_args()
    
    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create label encoder
    print("\nCreating label encoder...")
    label_to_idx, unique_labels = create_label_encoder(args.root_dir, args.target_labels)
    num_classes = len(unique_labels)
    print(f"Total unique labels: {num_classes}")
    
    # Create data loaders
    print("\nCreating DataLoaders...")
    dataloaders_train = create_dataset_specific_dataloaders(
        csv_file=args.csv_file,
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        shuffle=False,
        target_labels=args.target_labels,
        split="train",
        resize_to=args.img_size,
        augment=False,
        normalize=True,
        num_workers=args.num_workers,
    )
    
    dataloaders_test = create_dataset_specific_dataloaders(
        csv_file=args.csv_file,
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        shuffle=False,
        target_labels=args.target_labels,
        split="test",
        resize_to=args.img_size,
        augment=False,
        normalize=True,
        num_workers=args.num_workers,
    )
    
    # Model config
    base_config = {
        "img_size": args.img_size,
        "patch_size": args.patch_size,
        "embed_dim": args.embed_dim,
        "depth": args.depth,
        "num_heads": args.num_heads,
        "head_mode": "proxynca",
        "num_classes": num_classes,
        "metric_embed_dim": args.metric_embed_dim,
    }
    
    # Evaluate mean pooling model
    if os.path.exists(args.mean_checkpoint):
        mean_config = {**base_config, "aggregator_mode": "mean"}
        mean_results = evaluate_model_checkpoint(
            args.mean_checkpoint,
            mean_config,
            dataloaders_train,
            dataloaders_test,
            device,
            label_to_idx,
            "Mean Pooling Model"
        )
    else:
        print(f"\nWarning: Mean checkpoint not found at {args.mean_checkpoint}")
        mean_results = None
    
    # Evaluate attention pooling model
    if os.path.exists(args.attn_checkpoint):
        attn_config = {**base_config, "aggregator_mode": "attn"}
        attn_results = evaluate_model_checkpoint(
            args.attn_checkpoint,
            attn_config,
            dataloaders_train,
            dataloaders_test,
            device,
            label_to_idx,
            "Attention Pooling Model"
        )
    else:
        print(f"\nWarning: Attention checkpoint not found at {args.attn_checkpoint}")
        attn_results = None
    
    # Print comparison
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")
    
    if mean_results and attn_results:
        print("\nDataset-wise Comparison:")
        for dataset in ['Allen', 'HPA', 'CP']:
            if dataset in mean_results and dataset in attn_results:
                print(f"\n  {dataset}:")
                print(f"    Mean Pooling:    Acc={mean_results[dataset]['accuracy']:.4f}, F1={mean_results[dataset]['macro_f1']:.4f}")
                print(f"    Attention Pool:  Acc={attn_results[dataset]['accuracy']:.4f}, F1={attn_results[dataset]['macro_f1']:.4f}")
        
        if 'Overall' in mean_results and 'Overall' in attn_results:
            print(f"\n  Overall:")
            print(f"    Mean Pooling:    Acc={mean_results['Overall']['accuracy']:.4f}, F1={mean_results['Overall']['macro_f1']:.4f}")
            print(f"    Attention Pool:  Acc={attn_results['Overall']['accuracy']:.4f}, F1={attn_results['Overall']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()

