"""
Enhanced evaluation script for Early Fusion ViT models.
Evaluates per CHAMMI task splits with UMAP, t-SNE visualizations.
Based on evaluate_boc_enhanced.py but adapted for Early Fusion ViT.
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
from sklearn.manifold import TSNE
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: umap-learn not installed. UMAP visualizations will be skipped.")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from channel_adaptive_pipeline.models.early_fusion_vit import EarlyFusionViT
from channel_adaptive_pipeline.chammi_dataset import CHAMMIDataset, CHAMMITransform
from channel_adaptive_pipeline.chammi_grouped_dataloader import create_dataset_specific_dataloaders
from channel_adaptive_pipeline.model_utils import load_model_checkpoint
from torch.utils.data import DataLoader


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


def evaluate_1nn(train_embeddings, train_labels, test_embeddings, test_labels, metric="cosine", leave_one_out=False):
    """Evaluate using 1-NN classification."""
    # Normalize embeddings for cosine distance
    if metric == "cosine":
        train_embeddings = train_embeddings / (np.linalg.norm(train_embeddings, axis=1, keepdims=True) + 1e-8)
        test_embeddings = test_embeddings / (np.linalg.norm(test_embeddings, axis=1, keepdims=True) + 1e-8)
    
    if leave_one_out:
        print(f"        Using leave-one-out evaluation for {len(test_labels)} test samples...", flush=True)
        all_embeddings = np.concatenate([train_embeddings, test_embeddings], axis=0)
        all_labels = np.concatenate([train_labels, test_labels], axis=0)
        n_test = len(test_labels)
        n_train = len(train_labels)
        
        all_nn_model = NearestNeighbors(n_neighbors=2, metric=metric)
        all_nn_model.fit(all_embeddings)
        
        pred_labels = []
        for i in tqdm(range(n_test), desc="        Leave-one-out", leave=False):
            test_idx = n_train + i
            query_emb = all_embeddings[test_idx:test_idx+1]
            distances, indices = all_nn_model.kneighbors(query_emb)
            nn_idx = indices.flatten()[1] if len(indices.flatten()) > 1 else indices.flatten()[0]
            pred_labels.append(all_labels[nn_idx])
        
        pred_labels = np.array(pred_labels)
    else:
        nn_model = NearestNeighbors(n_neighbors=1, metric=metric)
        nn_model.fit(train_embeddings)
        distances, indices = nn_model.kneighbors(test_embeddings)
        pred_labels = train_labels[indices.flatten()]
    
    accuracy = accuracy_score(test_labels, pred_labels)
    macro_f1 = f1_score(test_labels, pred_labels, average="macro", zero_division=0)
    
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


def extract_embeddings(model, dataloader, device, label_to_idx, dataset_name, output_dir, desc="Extracting"):
    """Extract embeddings and labels from model."""
    model.eval()
    embeddings_list = []
    labels_list = []
    ids_list = []
    file_paths_list = []
    
    print(f"    {desc}...")
    with torch.no_grad():
        for images, metadatas, labels in tqdm(dataloader, desc=f"      {desc}", leave=False):
            images = images.to(device)
            
            # Extract features before classification head
            features = model.extract_features(images)
            embeddings_list.append(features.cpu().numpy())
            labels_list.extend(labels)
            ids_list.extend([meta['ID'] for meta in metadatas])
            file_paths_list.extend([meta['file_path'] for meta in metadatas])
    
    embeddings = np.concatenate(embeddings_list, axis=0)
    labels = encode_labels(labels_list, label_to_idx)
    
    # Save embeddings
    save_path = output_dir / f"embeddings_{dataset_name}.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump({
            'embeddings': embeddings,
            'labels': labels,
            'ids': ids_list,
            'file_paths': file_paths_list,
        }, f)
    print(f"      Saved embeddings to {save_path}")
    
    return embeddings, labels, ids_list, file_paths_list


def load_embeddings(output_dir, dataset_name):
    """Load saved embeddings."""
    save_path = output_dir / f"embeddings_{dataset_name}.pkl"
    if not save_path.exists():
        return None
    
    with open(save_path, 'rb') as f:
        data = pickle.load(f)
    return (data['embeddings'], data['labels'], data['ids'], data['file_paths'])


def get_task_columns(dataset_name):
    """Get task columns for each dataset."""
    if dataset_name == 'Allen':
        return ['Task_one', 'Task_two']
    elif dataset_name == 'HPA':
        return ['Task_one', 'Task_two', 'Task_three']
    elif dataset_name == 'CP':
        return ['Task_one', 'Task_two', 'Task_three', 'Task_four']
    else:
        return []


def filter_by_task(enriched_meta, ids, file_paths, task_col, dataset_name):
    """Filter embeddings by task column using train_test_split."""
    id_to_split = {}
    for _, row in enriched_meta.iterrows():
        match_col = 'Key' if dataset_name == 'CP' else 'ID'
        sample_id = str(row[match_col])
        split_val = str(row['train_test_split']) if 'train_test_split' in row else None
        if split_val:
            id_to_split[sample_id] = split_val
    
    file_path_to_split = {}
    for _, row in enriched_meta.iterrows():
        if 'file_path' in row and 'train_test_split' in row:
            fp = str(row['file_path'])
            split_val = str(row['train_test_split'])
            file_path_to_split[fp] = split_val
    
    train_indices = []
    test_indices = []
    
    for i, (sample_id, file_path) in enumerate(zip(ids, file_paths)):
        split_val = id_to_split.get(str(sample_id))
        if split_val is None:
            split_val = file_path_to_split.get(str(file_path))
        
        if split_val == 'Train':
            train_indices.append(i)
        elif split_val == task_col:
            test_indices.append(i)
    
    return train_indices, test_indices


def plot_umap(embeddings, labels, title, save_path, metric='cosine'):
    """Generate UMAP visualization using cosine distance."""
    if not UMAP_AVAILABLE:
        print(f"      Skipping UMAP for {title} (umap-learn not installed)")
        return
    
    print(f"      Computing UMAP (metric={metric}) for {title}...")
    
    # Normalize embeddings for cosine distance
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Fit UMAP with cosine metric
    reducer = umap.UMAP(
        n_components=2, 
        random_state=42, 
        n_neighbors=30, 
        min_dist=0.3, 
        n_epochs=500,
        metric=metric
    )
    embedding_2d = reducer.fit_transform(embeddings_norm)
    
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    if n_classes <= 20:
        cmap = plt.cm.tab20
    elif n_classes <= 40:
        cmap = plt.cm.tab40
    else:
        cmap = plt.cm.turbo
    
    label_to_color = {label: cmap(i / max(n_classes - 1, 1)) for i, label in enumerate(unique_labels)}
    colors = [label_to_color[label] for label in labels]
    
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=colors, alpha=0.7, s=8, edgecolors='none')
    
    plt.title(f"{title}\n(UMAP with {metric} distance)", fontsize=14, fontweight='bold')
    plt.xlabel('UMAP 1', fontsize=12)
    plt.ylabel('UMAP 2', fontsize=12)
    
    if n_classes <= 20:
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=label_to_color[label], label=f'Class {label}') 
                          for label in unique_labels[:20]]
        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    else:
        plt.text(0.02, 0.98, f'{n_classes} classes', transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      Saved UMAP to {save_path}")


def plot_tsne(embeddings, labels, title, save_path):
    """Generate t-SNE visualization."""
    print(f"      Computing t-SNE for {title}...")
    
    # Normalize embeddings
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Fit t-SNE with cosine metric
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000, metric='cosine')
    embedding_2d = tsne.fit_transform(embeddings_norm)
    
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    if n_classes <= 20:
        cmap = plt.cm.tab20
    elif n_classes <= 40:
        cmap = plt.cm.tab40
    else:
        cmap = plt.cm.turbo
    
    label_to_color = {label: cmap(i / max(n_classes - 1, 1)) for i, label in enumerate(unique_labels)}
    colors = [label_to_color[label] for label in labels]
    
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=colors, alpha=0.7, s=8, edgecolors='none')
    
    plt.title(f"{title}\n(t-SNE with cosine distance)", fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE 1', fontsize=12)
    plt.ylabel('t-SNE 2', fontsize=12)
    
    if n_classes <= 20:
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=label_to_color[label], label=f'Class {label}') 
                          for label in unique_labels[:20]]
        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    else:
        plt.text(0.02, 0.98, f'{n_classes} classes', transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      Saved t-SNE to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Early Fusion ViT model per CHAMMI task")
    
    parser.add_argument("--csv-file", type=str, required=True)
    parser.add_argument("--root-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--target-labels", type=str, default="Label")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    
    # Model config
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=3)
    
    # Output directory
    parser.add_argument("--output-dir", type=str, default="./evaluation_outputs_early_fusion")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip embedding extraction if already done")
    
    args = parser.parse_args()
    
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    umap_dir = output_dir / "umaps"
    umap_dir.mkdir(parents=True, exist_ok=True)
    tsne_dir = output_dir / "tsne"
    tsne_dir.mkdir(parents=True, exist_ok=True)
    
    # Create label encoder
    print("\nCreating label encoder...", flush=True)
    label_to_idx, unique_labels = create_label_encoder(args.root_dir, args.target_labels)
    num_classes = len(unique_labels)
    print(f"Total unique labels: {num_classes}", flush=True)
    
    # Initialize model
    print(f"\nInitializing model...")
    model = EarlyFusionViT(
        num_classes=num_classes,
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
    ).to(device)
    
    # Load checkpoint using the compatibility function
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint_info = load_model_checkpoint(model, args.checkpoint, device)
    model.eval()
    print(f"Loaded model from epoch {checkpoint_info.get('epoch', 'unknown')}")
    
    # Extract embeddings for each dataset
    print(f"\n{'='*70}")
    print("EXTRACTING EMBEDDINGS")
    print(f"{'='*70}")
    
    datasets = ['Allen', 'HPA', 'CP']
    
    for dataset_name in datasets:
        print(f"\n  Dataset: {dataset_name}")
        
        # Check if embeddings already exist
        if args.skip_extraction:
            loaded = load_embeddings(output_dir, dataset_name)
            if loaded is not None:
                print(f"      Loaded embeddings from disk")
                continue
        
        # Create dataloader
        dataloaders = create_dataset_specific_dataloaders(
            csv_file=args.csv_file,
            root_dir=args.root_dir,
            batch_size=args.batch_size,
            shuffle=False,
            target_labels=args.target_labels,
            split=None,
            resize_to=args.img_size,
            augment=False,
            normalize=True,
            num_workers=args.num_workers,
        )
        
        if dataset_name not in dataloaders:
            print(f"      WARNING: No dataloader for {dataset_name}")
            continue
        
        dataloader = dataloaders[dataset_name]
        
        # Extract embeddings
        extract_embeddings(
            model, dataloader, device, label_to_idx,
            dataset_name, output_dir,
            desc=f"Extracting embeddings for {dataset_name}"
        )
    
    # Evaluate per task
    print(f"\n{'='*70}")
    print("EVALUATING PER TASK")
    print(f"{'='*70}")
    
    all_results = {}
    
    for dataset_name in datasets:
        all_results[dataset_name] = {}
        print(f"\n  Dataset: {dataset_name}")
        
        # Load embeddings
        loaded = load_embeddings(output_dir, dataset_name)
        if loaded is None:
            print(f"      ERROR: Could not load embeddings for {dataset_name}")
            continue
        
        embeddings, labels, ids, file_paths = loaded
        print(f"      Loaded {len(embeddings)} embeddings")
        
        # Load enriched metadata
        enriched_meta_path = os.path.join(args.root_dir, dataset_name, 'enriched_meta.csv')
        if not os.path.exists(enriched_meta_path):
            print(f"      ERROR: enriched_meta.csv not found for {dataset_name}")
            continue
        
        enriched_meta = pd.read_csv(enriched_meta_path, low_memory=False)
        
        # Get task columns
        task_cols = get_task_columns(dataset_name)
        
        # Evaluate each task
        for task_col in task_cols:
            print(f"\n      Task: {task_col}")
            
            # Filter by task
            train_indices, test_indices = filter_by_task(enriched_meta, ids, file_paths, task_col, dataset_name)
            
            if len(train_indices) == 0 or len(test_indices) == 0:
                print(f"        WARNING: Insufficient samples (train: {len(train_indices)}, test: {len(test_indices)})")
                continue
            
            print(f"        Train samples: {len(train_indices)}, Test samples: {len(test_indices)}")
            
            # Get embeddings and labels
            train_emb = embeddings[train_indices]
            train_lbl = labels[train_indices]
            test_emb = embeddings[test_indices]
            test_lbl = labels[test_indices]
            
            # Check if novel class task
            is_novel_class_task = (dataset_name == 'HPA' and task_col == 'Task_three') or \
                                 (dataset_name == 'CP' and task_col == 'Task_four')
            
            # Evaluate with 1-NN
            metrics = evaluate_1nn(train_emb, train_lbl, test_emb, test_lbl, 
                                  metric="cosine", leave_one_out=is_novel_class_task)
            all_results[dataset_name][task_col] = metrics
            eval_type = "Leave-one-out" if is_novel_class_task else "Standard 1-NN"
            print(f"        {eval_type} - Accuracy: {metrics['accuracy']:.4f}, Macro-F1: {metrics['macro_f1']:.4f}", flush=True)
            
            # Generate visualizations
            combined_emb = np.concatenate([train_emb, test_emb], axis=0)
            combined_lbl = np.concatenate([train_lbl, test_lbl], axis=0)
            
            task_name = f"{dataset_name}_{task_col}"
            
            # UMAP (cosine distance)
            plot_umap(
                combined_emb, combined_lbl,
                f"Early Fusion ViT - {task_name}",
                umap_dir / f"umap_{task_name}.png",
                metric='cosine'
            )
            
            # t-SNE (cosine distance)
            plot_tsne(
                combined_emb, combined_lbl,
                f"Early Fusion ViT - {task_name}",
                tsne_dir / f"tsne_{task_name}.png"
            )
    
    # Print results
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    
    for dataset_name in datasets:
        print(f"\n{dataset_name}:")
        task_cols = get_task_columns(dataset_name)
        print(f"{'Task':<15} {'Accuracy':<12} {'Macro-F1':<12}")
        print("-" * 40)
        
        for task_col in task_cols:
            metrics = all_results.get(dataset_name, {}).get(task_col)
            if metrics:
                acc = metrics['accuracy']
                f1 = metrics['macro_f1']
                print(f"{task_col:<15} {acc:<12.4f} {f1:<12.4f}")
    
    # Overall averages
    print(f"\n{'='*70}")
    print("OVERALL AVERAGES")
    print(f"{'='*70}")
    
    all_acc = []
    all_f1 = []
    
    for dataset_name in datasets:
        task_cols = get_task_columns(dataset_name)
        for task_col in task_cols:
            metrics = all_results.get(dataset_name, {}).get(task_col)
            if metrics:
                all_acc.append(metrics['accuracy'])
                all_f1.append(metrics['macro_f1'])
    
    if all_acc:
        print(f"Overall:    Acc={np.mean(all_acc):.4f}, F1={np.mean(all_f1):.4f}")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"UMAPs saved to: {umap_dir}")
    print(f"t-SNE plots saved to: {tsne_dir}")


if __name__ == "__main__":
    main()
