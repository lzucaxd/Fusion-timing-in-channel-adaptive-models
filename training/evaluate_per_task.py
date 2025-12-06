"""
Evaluate both mean and attention pooling models per CHAMMI task splits (SD/OOD).
Extracts embeddings once, saves them, then filters by task for evaluation.
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
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: umap-learn not installed. UMAP visualizations will be skipped.")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.boc_vit import BoCViT
from channel_adaptive_pipeline.chammi_dataset import CHAMMIDataset, CHAMMITransform
from channel_adaptive_pipeline.chammi_grouped_dataloader import create_dataset_specific_dataloaders, _chammi_collate_fn
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
    """Evaluate using 1-NN classification.
    
    Args:
        leave_one_out: If True, use leave-one-out cross-validation for novel class tasks.
                      Combines train+test, then hides one test point at a time.
    """
    # Normalize embeddings for cosine distance
    if metric == "cosine":
        train_embeddings = train_embeddings / (np.linalg.norm(train_embeddings, axis=1, keepdims=True) + 1e-8)
        test_embeddings = test_embeddings / (np.linalg.norm(test_embeddings, axis=1, keepdims=True) + 1e-8)
    
    if leave_one_out:
        # For novel class tasks: combine train+test, then leave-one-out
        print(f"        Using leave-one-out evaluation for {len(test_labels)} test samples...", flush=True)
        all_embeddings = np.concatenate([train_embeddings, test_embeddings], axis=0)
        all_labels = np.concatenate([train_labels, test_labels], axis=0)
        n_test = len(test_labels)
        n_train = len(train_labels)
        
        # Fit NN on all data once (will exclude query point in loop)
        all_nn_model = NearestNeighbors(n_neighbors=2, metric=metric)  # Get 2 neighbors (self + actual NN)
        all_nn_model.fit(all_embeddings)
        
        pred_labels = []
        for i in tqdm(range(n_test), desc="        Leave-one-out", leave=False):
            # Hide one test point
            test_idx = n_train + i
            
            # Get 2 nearest neighbors (will include self as first, actual NN as second)
            query_emb = all_embeddings[test_idx:test_idx+1]
            distances, indices = all_nn_model.kneighbors(query_emb)
            
            # Take the second neighbor (first is self)
            nn_idx = indices.flatten()[1] if len(indices.flatten()) > 1 else indices.flatten()[0]
            pred_labels.append(all_labels[nn_idx])
        
        pred_labels = np.array(pred_labels)
    else:
        # Standard 1-NN: use train set as reference
        nn_model = NearestNeighbors(n_neighbors=1, metric=metric)
        nn_model.fit(train_embeddings)
        
        # Find nearest neighbors
        distances, indices = nn_model.kneighbors(test_embeddings)
        
        # Predict labels
        pred_labels = train_labels[indices.flatten()]
    
    # Compute metrics
    accuracy = accuracy_score(test_labels, pred_labels)
    macro_f1 = f1_score(test_labels, pred_labels, average="macro", zero_division=0)
    
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


def extract_and_save_embeddings(model, dataloader, device, label_to_idx, dataset_name, model_name, output_dir, desc="Extracting"):
    """Extract embeddings and labels from dataloader, save them."""
    model.eval()
    embeddings_list = []
    labels_list = []
    ids_list = []
    file_paths_list = []
    
    print(f"    {desc}...")
    with torch.no_grad():
        for images, metadatas, labels in tqdm(dataloader, desc=f"      {desc}", leave=False):
            images = images.to(device)
            embeddings = model.get_embedding(images)
            embeddings_list.append(embeddings.cpu().numpy())
            labels_list.extend(labels)
            ids_list.extend([meta['ID'] for meta in metadatas])
            file_paths_list.extend([meta['file_path'] for meta in metadatas])
    
    embeddings = np.concatenate(embeddings_list, axis=0)
    labels = encode_labels(labels_list, label_to_idx)
    
    # Save embeddings
    save_path = output_dir / f"embeddings_{model_name}_{dataset_name}.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump({
            'embeddings': embeddings,
            'labels': labels,
            'ids': ids_list,
            'file_paths': file_paths_list,
        }, f)
    print(f"      Saved embeddings to {save_path}")
    
    return embeddings, labels, ids_list, file_paths_list


def load_embeddings(output_dir, model_name, dataset_name):
    """Load saved embeddings."""
    save_path = output_dir / f"embeddings_{model_name}_{dataset_name}.pkl"
    if not save_path.exists():
        return None
    
    with open(save_path, 'rb') as f:
        data = pickle.load(f)
    return data['embeddings'], data['labels'], data['ids'], data['file_paths']


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
    # Create mapping from ID to train_test_split value
    id_to_split = {}
    for _, row in enriched_meta.iterrows():
        # Match by ID (or Key for CP)
        if dataset_name == 'CP':
            match_col = 'Key'
        else:
            match_col = 'ID'
        
        sample_id = str(row[match_col])
        split_val = str(row['train_test_split']) if 'train_test_split' in row else None
        if split_val:
            id_to_split[sample_id] = split_val
    
    # Also match by file_path as fallback
    file_path_to_split = {}
    for _, row in enriched_meta.iterrows():
        if 'file_path' in row and 'train_test_split' in row:
            fp = str(row['file_path'])
            split_val = str(row['train_test_split'])
            file_path_to_split[fp] = split_val
    
    # Filter indices
    train_indices = []
    test_indices = []
    
    for i, (sample_id, file_path) in enumerate(zip(ids, file_paths)):
        # Get train_test_split value for this sample
        split_val = id_to_split.get(str(sample_id))
        if split_val is None:
            split_val = file_path_to_split.get(str(file_path))
        
        if split_val == 'Train':
            train_indices.append(i)
        elif split_val == task_col:  # e.g., 'Task_one' or 'Task_two'
            test_indices.append(i)
    
    return train_indices, test_indices


def plot_umap(embeddings, labels, title, save_path, label_to_name=None):
    """Generate UMAP visualization with improved readability."""
    if not UMAP_AVAILABLE:
        print(f"      Skipping UMAP for {title} (umap-learn not installed)")
        return
    
    print(f"      Computing UMAP for {title}...")
    
    # Normalize embeddings
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Fit UMAP with adjusted parameters for better visualization
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3, n_epochs=500)
    embedding_2d = reducer.fit_transform(embeddings_norm)
    
    # Get unique labels and create a better colormap
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    # Use a colormap that works well for many classes
    if n_classes <= 20:
        cmap = plt.cm.tab20
    elif n_classes <= 40:
        cmap = plt.cm.tab40
    else:
        cmap = plt.cm.turbo
    
    # Create color mapping
    label_to_color = {label: cmap(i / max(n_classes - 1, 1)) for i, label in enumerate(unique_labels)}
    colors = [label_to_color[label] for label in labels]
    
    # Plot with larger points and better visibility
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=colors, alpha=0.7, s=10, edgecolors='none')
    
    # Add title and labels
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('UMAP 1', fontsize=12)
    plt.ylabel('UMAP 2', fontsize=12)
    
    # Add legend for a subset of classes (if not too many)
    if n_classes <= 20:
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=label_to_color[label], label=f'Class {label}') 
                          for label in unique_labels[:20]]
        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    else:
        # Just show number of classes
        plt.text(0.02, 0.98, f'{n_classes} classes', transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"      Saved UMAP to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate both models per CHAMMI task")
    
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
    
    # Output directory
    parser.add_argument("--output-dir", type=str, default="./evaluation_outputs")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip embedding extraction if already done")
    
    args = parser.parse_args()
    
    # Force flush for immediate output
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    
    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    umap_dir = output_dir / "umaps"
    umap_dir.mkdir(parents=True, exist_ok=True)
    
    # Create label encoder
    print("\nCreating label encoder...", flush=True)
    label_to_idx, unique_labels = create_label_encoder(args.root_dir, args.target_labels)
    num_classes = len(unique_labels)
    print(f"Total unique labels: {num_classes}", flush=True)
    
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
    
    datasets = ['Allen', 'HPA', 'CP']
    
    # Extract embeddings for both models
    print(f"\n{'='*70}")
    print("EXTRACTING EMBEDDINGS")
    print(f"{'='*70}")
    
    for model_name, checkpoint_path, aggregator_mode in [
        ('mean', args.mean_checkpoint, 'mean'),
        ('attn', args.attn_checkpoint, 'attn'),
    ]:
        print(f"\n  Model: {model_name.upper()} POOLING")
        
        # Load model
        if not os.path.exists(checkpoint_path):
            print(f"    ERROR: Checkpoint not found at {checkpoint_path}")
            continue
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_config = {**base_config, "aggregator_mode": aggregator_mode}
        model = BoCViT(**model_config).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"    Loaded model from epoch {checkpoint['epoch']}")
        
        # Extract embeddings for each dataset
        for dataset_name in datasets:
            print(f"\n    Dataset: {dataset_name}")
            
            # Check if embeddings already exist
            if args.skip_extraction:
                loaded = load_embeddings(output_dir, model_name, dataset_name)
                if loaded is not None:
                    print(f"      Loaded embeddings from disk")
                    continue
            
            # Create dataloader for this dataset
            dataloaders = create_dataset_specific_dataloaders(
                csv_file=args.csv_file,
                root_dir=args.root_dir,
                batch_size=args.batch_size,
                shuffle=False,
                target_labels=args.target_labels,
                split=None,  # Get all data
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
            extract_and_save_embeddings(
                model, dataloader, device, label_to_idx,
                dataset_name, model_name, output_dir,
                desc=f"Extracting {model_name} embeddings for {dataset_name}"
            )
    
    # Evaluate per task
    print(f"\n{'='*70}")
    print("EVALUATING PER TASK")
    print(f"{'='*70}")
    
    all_results = {}
    
    for model_name in ['mean', 'attn']:
        all_results[model_name] = {}
        print(f"\n  Model: {model_name.upper()} POOLING")
        
        for dataset_name in datasets:
            all_results[model_name][dataset_name] = {}
            print(f"\n    Dataset: {dataset_name}")
            
            # Load embeddings
            loaded = load_embeddings(output_dir, model_name, dataset_name)
            if loaded is None:
                print(f"      ERROR: Could not load embeddings for {model_name} {dataset_name}")
                continue
            
            embeddings, labels, ids, file_paths = loaded
            print(f"      Loaded {len(embeddings)} embeddings")
            
            # Load enriched metadata
            enriched_meta_path = os.path.join(args.root_dir, dataset_name, 'enriched_meta.csv')
            if not os.path.exists(enriched_meta_path):
                print(f"      ERROR: enriched_meta.csv not found for {dataset_name}")
                continue
            
            enriched_meta = pd.read_csv(enriched_meta_path, low_memory=False)
            print(f"      Loaded enriched metadata with {len(enriched_meta)} rows")
            
            # Get task columns for this dataset
            task_cols = get_task_columns(dataset_name)
            
            # Evaluate each task
            for task_col in task_cols:
                print(f"\n      Task: {task_col}")
                
                # Filter by task
                train_indices, test_indices = filter_by_task(enriched_meta, ids, file_paths, task_col, dataset_name)
                
                if len(train_indices) == 0:
                    print(f"        WARNING: No training samples for {task_col}")
                    continue
                if len(test_indices) == 0:
                    print(f"        WARNING: No test samples for {task_col}")
                    continue
                
                print(f"        Train samples: {len(train_indices)}, Test samples: {len(test_indices)}")
                
                # Get embeddings and labels for train and test
                train_emb = embeddings[train_indices]
                train_lbl = labels[train_indices]
                test_emb = embeddings[test_indices]
                test_lbl = labels[test_indices]
                
                # Check if this is a novel class task (H_Task3, C_Task4) - need leave-one-out
                is_novel_class_task = (dataset_name == 'HPA' and task_col == 'Task_three') or \
                                     (dataset_name == 'CP' and task_col == 'Task_four')
                
                # Evaluate with 1-NN
                metrics = evaluate_1nn(train_emb, train_lbl, test_emb, test_lbl, 
                                      metric="cosine", leave_one_out=is_novel_class_task)
                all_results[model_name][dataset_name][task_col] = metrics
                eval_type = "Leave-one-out" if is_novel_class_task else "Standard 1-NN"
                print(f"        {eval_type} - Accuracy: {metrics['accuracy']:.4f}, Macro-F1: {metrics['macro_f1']:.4f}", flush=True)
                
                # Generate UMAP
                # Combined train+test for visualization
                combined_emb = np.concatenate([train_emb, test_emb], axis=0)
                combined_lbl = np.concatenate([train_lbl, test_lbl], axis=0)
                plot_umap(
                    combined_emb, combined_lbl,
                    f"{model_name.upper()} - {dataset_name} - {task_col}",
                    umap_dir / f"umap_{model_name}_{dataset_name}_{task_col}.png"
                )
    
    # Print comparison
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY - PER TASK")
    print(f"{'='*70}")
    
    for dataset_name in datasets:
        print(f"\n{dataset_name}:")
        task_cols = get_task_columns(dataset_name)
        print(f"{'Task':<15} {'Mean Acc':<12} {'Mean F1':<12} {'Attn Acc':<12} {'Attn F1':<12}")
        print("-" * 65)
        
        for task_col in task_cols:
            mean_metrics = all_results.get('mean', {}).get(dataset_name, {}).get(task_col)
            attn_metrics = all_results.get('attn', {}).get(dataset_name, {}).get(task_col)
            
            if mean_metrics and attn_metrics:
                mean_acc = mean_metrics['accuracy']
                mean_f1 = mean_metrics['macro_f1']
                attn_acc = attn_metrics['accuracy']
                attn_f1 = attn_metrics['macro_f1']
                print(f"{task_col:<15} {mean_acc:<12.4f} {mean_f1:<12.4f} {attn_acc:<12.4f} {attn_f1:<12.4f}")
    
    # Overall averages
    print(f"\n{'='*70}")
    print("OVERALL AVERAGES")
    print(f"{'='*70}")
    
    all_mean_acc = []
    all_mean_f1 = []
    all_attn_acc = []
    all_attn_f1 = []
    
    for dataset_name in datasets:
        task_cols = get_task_columns(dataset_name)
        for task_col in task_cols:
            mean_metrics = all_results.get('mean', {}).get(dataset_name, {}).get(task_col)
            attn_metrics = all_results.get('attn', {}).get(dataset_name, {}).get(task_col)
            if mean_metrics and attn_metrics:
                all_mean_acc.append(mean_metrics['accuracy'])
                all_mean_f1.append(mean_metrics['macro_f1'])
                all_attn_acc.append(attn_metrics['accuracy'])
                all_attn_f1.append(attn_metrics['macro_f1'])
    
    if all_mean_acc:
        print(f"Mean Pooling:    Acc={np.mean(all_mean_acc):.4f}, F1={np.mean(all_mean_f1):.4f}")
        print(f"Attention Pool:  Acc={np.mean(all_attn_acc):.4f}, F1={np.mean(all_attn_f1):.4f}")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"UMAPs saved to: {umap_dir}")


if __name__ == "__main__":
    main()
