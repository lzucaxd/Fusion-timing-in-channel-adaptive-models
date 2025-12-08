"""
Evaluate the latest checkpoints of both mean and attention pooling models.
"""

import torch
import torch.nn as nn
import argparse
from pathlib import Path
import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.boc_vit import BoCViT
from channel_adaptive_pipeline.chammi_grouped_dataloader import create_dataset_specific_dataloaders


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


def evaluate_1nn(train_embeddings, train_labels, test_embeddings, test_labels, metric="cosine"):
    """Evaluate using 1-NN classification."""
    # Normalize embeddings for cosine distance
    if metric == "cosine":
        train_embeddings = train_embeddings / (np.linalg.norm(train_embeddings, axis=1, keepdims=True) + 1e-8)
        test_embeddings = test_embeddings / (np.linalg.norm(test_embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Fit 1-NN
    nn_model = NearestNeighbors(n_neighbors=1, metric=metric)
    nn_model.fit(train_embeddings)
    
    # Find nearest neighbors
    distances, indices = nn_model.kneighbors(test_embeddings)
    
    # Predict labels
    pred_labels = train_labels[indices.flatten()]
    
    # Compute metrics
    accuracy = accuracy_score(test_labels, pred_labels)
    macro_f1 = f1_score(test_labels, pred_labels, average="macro")
    
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


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
    if not os.path.exists(checkpoint_path):
        print(f"  ERROR: Checkpoint not found!")
        return None
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = BoCViT(**model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"  Loaded model from epoch {epoch}")
    
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
                # Model outputs embeddings directly (proxynca mode)
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
    print(f"\n{'='*70}")
    print("EVALUATING MEAN POOLING MODEL")
    print(f"{'='*70}")
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
    
    # Evaluate attention pooling model
    print(f"\n{'='*70}")
    print("EVALUATING ATTENTION POOLING MODEL")
    print(f"{'='*70}")
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
    
    # Print comparison
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    if mean_results and attn_results:
        print("\nDataset-wise Comparison:")
        print(f"{'Dataset':<12} {'Mean Acc':<12} {'Mean F1':<12} {'Attn Acc':<12} {'Attn F1':<12}")
        print("-" * 60)
        
        for dataset in ['Allen', 'HPA', 'CP']:
            if dataset in mean_results and dataset in attn_results:
                mean_acc = mean_results[dataset]['accuracy']
                mean_f1 = mean_results[dataset]['macro_f1']
                attn_acc = attn_results[dataset]['accuracy']
                attn_f1 = attn_results[dataset]['macro_f1']
                print(f"{dataset:<12} {mean_acc:<12.4f} {mean_f1:<12.4f} {attn_acc:<12.4f} {attn_f1:<12.4f}")
        
        # Overall average
        if len(mean_results) > 0 and len(attn_results) > 0:
            mean_avg_acc = np.mean([mean_results[d]['accuracy'] for d in mean_results.keys() if d != 'Overall'])
            mean_avg_f1 = np.mean([mean_results[d]['macro_f1'] for d in mean_results.keys() if d != 'Overall'])
            attn_avg_acc = np.mean([attn_results[d]['accuracy'] for d in attn_results.keys() if d != 'Overall'])
            attn_avg_f1 = np.mean([attn_results[d]['macro_f1'] for d in attn_results.keys() if d != 'Overall'])
            
            print("-" * 60)
            print(f"{'Average':<12} {mean_avg_acc:<12.4f} {mean_avg_f1:<12.4f} {attn_avg_acc:<12.4f} {attn_avg_f1:<12.4f}")
    elif mean_results:
        print("\nOnly Mean Pooling results available:")
        for dataset, metrics in mean_results.items():
            if dataset != 'Overall':
                print(f"  {dataset}: Acc={metrics['accuracy']:.4f}, F1={metrics['macro_f1']:.4f}")
    elif attn_results:
        print("\nOnly Attention Pooling results available:")
        for dataset, metrics in attn_results.items():
            if dataset != 'Overall':
                print(f"  {dataset}: Acc={metrics['accuracy']:.4f}, F1={metrics['macro_f1']:.4f}")
    else:
        print("\nNo results available. Check checkpoint paths.")


if __name__ == "__main__":
    main()

