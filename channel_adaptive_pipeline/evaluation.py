import pandas as pd
import numpy as np
import os 
import torch
import torch.nn as nn

import umap 
import matplotlib.pyplot as plt
import seaborn as sb

import channel_adaptive_pipeline.utils as utils

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, classification_report, accuracy_score
from typing import Dict, List, Optional, Tuple, Any
from torch.utils.data import DataLoader

########################################################
## Helper Function to Plot Umap
########################################################

def create_umap(dataset, features_path, df_path, dest_dir, label_list):
    # Helper function
    # Create umap and save in result directory
    
    if not os.path.exists(dest_dir+ '/'):
        os.makedirs(dest_dir+ '/')
        
    print('Creating umap for features')
    
    # Load features and metadata
    features = np.load(features_path)
    df = pd.read_csv(df_path)
    
    # Split into training and testing
    tasks = list(df['train_test_split'].unique())
    tasks.remove('Train')
    
    train_idx = np.where(df['train_test_split'] == 'Train')[0]
    all_test_indices = [np.where(df[task])[0] for task in tasks]
    train_feat = features[train_idx]
    test_feat = [features[idx] for idx in all_test_indices]
    
    # Fit umap on train set
    reducer = umap.UMAP(n_neighbors=15, n_components=2)
    train_embeddings = reducer.fit_transform(train_feat)
    train_aux = pd.concat((pd.DataFrame(train_embeddings, columns=["X", "Y"]), 
                           df.loc[train_idx].reset_index()), axis=1)
    
    # Transform test set with fitted umap 
    test_aux_list = []
    for i in range(len(tasks)):
        test_embeddings = reducer.transform(test_feat[i])
        test_aux = pd.concat((pd.DataFrame(test_embeddings, columns=["X", "Y"]), 
                              df.loc[all_test_indices[i]].reset_index()), axis=1)
        test_aux_list.append(test_aux)

    # Plot the UMAP embedding
    fig, axs = plt.subplots(nrows=2, ncols=len(tasks)+1, figsize=(20*len(tasks)+1, 20))
    
    # Set up color palette
    col1 = sb.hls_palette(len(df[label_list[0]].unique())).as_hex()
    col2 = sb.hls_palette(len(df[label_list[1]].unique())).as_hex()
    
    pal1, pal2 = {}, {}
    for i in range(len(df[label_list[0]].unique())):
        val = df[label_list[0]].unique()[i]
        pal1[val] = col1[i]

    for i in range(len(df[label_list[1]].unique())):
        val = df[label_list[1]].unique()[i]
        pal2[val] = col2[i]


    # Plot train set classification label umap
    a = sb.scatterplot(ax=axs[0,0],data=train_aux, x="X", y="Y", s=5, hue='Label', palette=pal1)
    a.set(title=f'UMAP of {dataset} Train Set')

    # Plot train set subgroup umap
    c = sb.scatterplot(ax=axs[1,0],data=train_aux, x="X", y="Y", s=5, hue=label_list[1], palette=pal2)
    c.set(title=f'UMAP of {dataset} Train Set')

    legend_list = []
    for i in range(len(tasks)):
        # Plot test set classification label umap
        b = sb.scatterplot(ax=axs[0,i+1], data=test_aux_list[i], x="X", y="Y", s=5, hue='Label', 
                           palette=pal1)
        b.set(title=f'UMAP of {dataset} Test Set for {tasks[i]}')

        # Plot test set subgroup umap
        d = sb.scatterplot(ax=axs[1,i+1], data=test_aux_list[i], x="X", y="Y", s=5, hue=label_list[1], 
                           palette=pal2)
        d.set(title=f'UMAP of {dataset} Test Set for {tasks[i]}')

    # Save figure
    fig.savefig(f'{dest_dir}/umap_{dataset}.png')
    print(f'Saved umap figure at {dest_dir} as umap_{dataset}.png')
    
    return

########################################################
## Evaluation Function
########################################################

def evaluate(features_path, df_path, leave_out, leaveout_label, model_choice, use_gpu: bool, knn_metric: str):
    
    print('Running classification pipeline...')
    
    # Load features and metadata
    print('Load features...')

    features = np.load(features_path)
    df = pd.read_csv(df_path)

    # Count number of tasks
    tasks = list(df['train_test_split'].unique())
    tasks.remove('Train')
    
    # Sort task from 1->4 for consistency when displaying results
    mapper = {"Task_one": 1, "Task_two": 2, "Task_three": 3, "Task_four": 4}
    tasks = sorted(tasks, key=lambda x: mapper[x])
    
    if leave_out != None:
        leaveout_ind = tasks.index(leave_out)

    # Get index for training and each testing set
    train_indices = np.where(df['train_test_split'] == 'Train')[0]
    all_test_indices = [np.where(df[task])[0] for task in tasks]

    # Convert categorical labels to integers    
    target_value = list(df['Label'].unique())

    encoded_target = {}
    for i in range(len(target_value)):
        encoded_target[target_value[i]] = i
    df['encoded_label'] = df.Label.apply(lambda x: encoded_target[x])

    # Split data into training and testing for regular classification
    train_X = features[train_indices]
    test_Xs = [features[test_indices] for test_indices in all_test_indices]

    task_Ys = [df['encoded_label'].values for key in tasks]
    train_Ys = [task_Ys[task_ind][train_indices] for task_ind in range(len(tasks))]
    test_Ys = [task_Ys[task_ind][test_indices] for task_ind, test_indices in enumerate(all_test_indices)]

    # Data splitting for leave one out task
    if leave_out != None:
        df_takeout = df[df[leave_out]]
        groups = list(df_takeout[leaveout_label].unique())

        all_group_indices = [df_takeout[df_takeout[leaveout_label]==group].index.values for group in groups]
        all_other_indices = [df_takeout[df_takeout[leaveout_label]!=group].index.values for group in groups]

        takeout_X = [features[group_indices] for group_indices in all_group_indices]
        rest_X = [features[np.concatenate((train_indices,other_indices), axis=None)] \
                                              for other_indices in all_other_indices]

        takeout_Y = [task_Ys[leaveout_ind][group_indices] for group_indices in all_group_indices]
        rest_Y = [task_Ys[leaveout_ind][np.concatenate((train_indices,other_indices), axis=None)] \
                                                      for other_indices in all_other_indices]

    print('Train classifiers...')
    accuracies = []
    f1scores_macro = []
    reports_str = []
    reports_dict = []



    for task_ind, task in enumerate(tasks):
        if task != leave_out: # standard classification

            if model_choice == 'knn':
                model = utils.FaissKNeighbors(k=1, use_gpu=use_gpu, metric=knn_metric)
            elif model_choice == 'sgd':
                model = SGDClassifier(alpha=0.001, max_iter=100)
            else:
                print(f'{model_choice} is not implemented. Try sgd or knn.')
                break

            model.fit(train_X, train_Ys[task_ind])
            predictions = model.predict(test_Xs[task_ind])
            ground_truth = test_Ys[task_ind]

        else: # leave-one-out
            predictions = []
            ground_truth = []
            for group_ind, group in enumerate(groups):
                if model_choice == 'knn':
                    model = utils.FaissKNeighbors(k=1, use_gpu=use_gpu, metric=knn_metric)
                elif model_choice == 'sgd':
                    model = SGDClassifier(alpha=0.001, max_iter=100)
                else:
                    print(f'{model_choice} is not implemented. Try sgd or knn.')
                    break

                model.fit(rest_X[group_ind], rest_Y[group_ind])
                group_predictions = model.predict(takeout_X[group_ind])
                group_ground_truth = takeout_Y[group_ind]

                predictions.append(group_predictions)
                ground_truth.append(group_ground_truth)

            predictions = np.concatenate(predictions)
            ground_truth = np.concatenate(ground_truth)
        # Compute evaluation metrics
        int_labels = np.unique(ground_truth)
        str_labels = [target_value[idx] for idx in int_labels]
        
        accuracy = np.mean(predictions == ground_truth)
        report_str = classification_report(ground_truth, predictions, labels=int_labels, target_names=str_labels)
        report_dict = classification_report(ground_truth, predictions, labels=int_labels, \
                                            target_names=str_labels, output_dict=True)
        f1score_macro = f1_score(ground_truth, predictions, labels=np.unique(ground_truth), average='macro')

        accuracies.append(accuracy)
        f1scores_macro.append(f1score_macro)
        reports_str.append(report_str)
        reports_dict.append(report_dict)    
        
    return {
        "tasks": tasks,
        "accuracies": accuracies, 
        "f1scores_macro": f1scores_macro, 
        "reports_str":reports_str, 
        "reports_dict": reports_dict,
        "encoded_target": encoded_target
    }


########################################################
## Model Evaluation Functions
########################################################

def evaluate_model_on_dataloaders(
    model: nn.Module,
    dataloaders_dict: Dict[int, DataLoader],
    device: torch.device,
    num_classes: int,
    class_to_idx: Optional[Dict[str, int]] = None,
) -> Dict[int, Dict[str, float]]:
    """
    Evaluate model on dataloaders grouped by channel count.
    
    Args:
        model: PyTorch model to evaluate
        dataloaders_dict: Dictionary mapping channel_count -> DataLoader
        device: Device to evaluate on
        num_classes: Number of classes
        class_to_idx: Optional mapping from class names to indices
    
    Returns:
        Dictionary mapping channel_count -> metrics dict
        Example: {3: {'accuracy': 0.85, 'macro_f1': 0.82, ...}, ...}
    """
    model.eval()
    
    results = {}
    
    from tqdm import tqdm
    
    with torch.no_grad():
        for channel_count, dataloader in dataloaders_dict.items():
            all_predictions = []
            all_labels = []
            
            print(f"  Processing {channel_count} channels...")
            for batch_images, batch_metadatas, batch_labels in tqdm(dataloader, desc=f"    C={channel_count}", leave=False):
                batch_images = batch_images.to(device)
                
                # Convert labels to indices if needed
                if isinstance(batch_labels[0], str) and class_to_idx is not None:
                    batch_labels_tensor = torch.tensor([
                        class_to_idx[label] if label in class_to_idx else 0
                        for label in batch_labels
                    ], dtype=torch.long, device=device)
                else:
                    batch_labels_tensor = torch.tensor(batch_labels, dtype=torch.long, device=device)
                
                # Forward pass
                logits = model(batch_images)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch_labels_tensor.cpu().numpy())
            
            # Compute metrics
            all_predictions = np.array(all_predictions)
            all_labels = np.array(all_labels)
            
            accuracy = accuracy_score(all_labels, all_predictions)
            macro_f1 = f1_score(all_labels, all_predictions, average='macro')
            
            # Per-class metrics
            report_dict = classification_report(
                all_labels, all_predictions,
                labels=np.arange(num_classes),
                output_dict=True,
                zero_division=0,
            )
            
            results[channel_count] = {
                'accuracy': accuracy,
                'macro_f1': macro_f1,
                'per_class': report_dict,
            }
    
    return results


def evaluate_sd_ood(
    model: nn.Module,
    csv_file: str,
    root_dir: str,
    config: Any,
    device: torch.device,
    class_to_idx: Dict[str, int],
    num_classes: int,
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    Evaluate model on SD (in-distribution) and OOD (out-of-distribution) splits.
    
    Args:
        model: PyTorch model to evaluate
        csv_file: Path to combined_metadata.csv
        root_dir: Root directory of CHAMMI dataset
        config: Config object with evaluation settings
        device: Device to evaluate on
        class_to_idx: Mapping from class names to indices
        num_classes: Number of classes
    
    Returns:
        Dictionary with 'sd' and 'ood' keys, each containing results per channel_count
    """
    from channel_adaptive_pipeline.chammi_grouped_dataloader import create_grouped_chammi_dataloaders
    
    results = {}
    
    # Evaluate on SD (in-distribution) split
    print(f"Evaluating on SD split: {config.sd_split}")
    sd_dataloaders = create_grouped_chammi_dataloaders(
        csv_file=csv_file,
        root_dir=root_dir,
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
    
    sd_results = evaluate_model_on_dataloaders(
        model=model,
        dataloaders_dict=sd_dataloaders,
        device=device,
        num_classes=num_classes,
        class_to_idx=class_to_idx,
    )
    results['sd'] = sd_results
    
    # Evaluate on OOD splits
    results['ood'] = {}
    for ood_split in config.ood_splits:
        print(f"Evaluating on OOD split: {ood_split}")
        try:
            ood_dataloaders = create_grouped_chammi_dataloaders(
                csv_file=csv_file,
                root_dir=root_dir,
                batch_size=config.batch_size,
                shuffle=False,
                target_labels=config.target_labels,
                split=ood_split,
                resize_to=config.img_size,
                augment=False,
                normalize=config.normalize,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
            )
            
            # Check if dataloaders are empty
            if len(ood_dataloaders) == 0:
                print(f"  Warning: No samples found for OOD split '{ood_split}'. Skipping.")
                results['ood'][ood_split] = {}
                continue
            
            ood_results = evaluate_model_on_dataloaders(
                model=model,
                dataloaders_dict=ood_dataloaders,
                device=device,
                num_classes=num_classes,
                class_to_idx=class_to_idx,
            )
            results['ood'][ood_split] = ood_results
        except Exception as e:
            print(f"  Warning: Error evaluating OOD split '{ood_split}': {e}. Skipping.")
            results['ood'][ood_split] = {}
    
    return results


def evaluate_per_dataset(
    model: nn.Module,
    dataloaders_dict: Dict[int, DataLoader],
    device: torch.device,
    num_classes: int,
    class_to_idx: Optional[Dict[str, int]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model and compute metrics per dataset (Allen, HPA, CP).
    
    Args:
        model: PyTorch model to evaluate
        dataloaders_dict: Dictionary mapping channel_count -> DataLoader
        device: Device to evaluate on
        num_classes: Number of classes
        class_to_idx: Optional mapping from class names to indices
    
    Returns:
        Dictionary mapping dataset_source -> metrics dict
    """
    model.eval()
    
    dataset_results = {'Allen': {'predictions': [], 'labels': []},
                       'HPA': {'predictions': [], 'labels': []},
                       'CP': {'predictions': [], 'labels': []}}
    
    from tqdm import tqdm
    
    with torch.no_grad():
        for channel_count, dataloader in dataloaders_dict.items():
            print(f"  Processing {channel_count} channels for per-dataset metrics...")
            for batch_images, batch_metadatas, batch_labels in tqdm(dataloader, desc=f"    C={channel_count}", leave=False):
                batch_images = batch_images.to(device)
                
                # Convert labels to indices if needed
                if isinstance(batch_labels[0], str) and class_to_idx is not None:
                    batch_labels_tensor = torch.tensor([
                        class_to_idx[label] if label in class_to_idx else 0
                        for label in batch_labels
                    ], dtype=torch.long, device=device)
                else:
                    batch_labels_tensor = torch.tensor(batch_labels, dtype=torch.long, device=device)
                
                # Forward pass
                logits = model(batch_images)
                predictions = torch.argmax(logits, dim=1)
                
                # Group by dataset source
                for i, metadata in enumerate(batch_metadatas):
                    dataset_source = metadata.get('dataset_source', 'Unknown')
                    if dataset_source in dataset_results:
                        dataset_results[dataset_source]['predictions'].append(predictions[i].item())
                        dataset_results[dataset_source]['labels'].append(batch_labels_tensor[i].item())
    
    # Compute metrics per dataset
    final_results = {}
    for dataset_source, data in dataset_results.items():
        if len(data['predictions']) > 0:
            predictions = np.array(data['predictions'])
            labels = np.array(data['labels'])
            
            accuracy = accuracy_score(labels, predictions)
            macro_f1 = f1_score(labels, predictions, average='macro')
            
            final_results[dataset_source] = {
                'accuracy': accuracy,
                'macro_f1': macro_f1,
                'num_samples': len(predictions),
            }
    
    return final_results