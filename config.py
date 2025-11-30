"""
Configuration management for channel-adaptive ViT training.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """Configuration class for training channel-adaptive ViT models."""
    
    # Model settings
    model_type: str = 'early_fusion'  # 'early_fusion', 'late_fusion', 'hybrid_fusion'
    vit_size: str = 'tiny'  # 'tiny' or 'small'
    num_classes: Optional[int] = None  # Will be auto-detected from dataset
    img_size: int = 128
    patch_size: int = 16
    
    # Training hyperparameters
    batch_size: int = 128  # 128-256 range
    num_epochs: int = 50
    learning_rate: float = 3e-4  # Start with 3e-4, adjust to 1e-4 or 5e-4 if needed
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    
    # Scheduler settings
    warmup_epochs: int = 5
    warmup_start_lr: float = 1e-6
    min_lr: float = 1e-6
    
    # Data settings
    csv_file: str = "CHAMMI/combined_metadata.csv"
    root_dir: str = "CHAMMI"
    target_labels: str = 'Label'  # Column name in enriched_meta.csv
    split: str = 'train'  # 'train', 'test', 'val', or OOD split name
    augment: bool = True
    normalize: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    
    # Evaluation settings
    sd_split: str = 'val'  # In-distribution split for evaluation
    ood_splits: list = field(default_factory=lambda: ['ood_val', 'ood_test'])  # OOD splits from enriched_meta
    
    # Paths
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    experiment_name: str = "early_fusion_vit_tiny"
    
    # Device
    device: str = 'cuda'  # 'cuda' or 'cpu'
    gpu_id: int = 0
    
    # Checkpointing
    save_freq: int = 10  # Save checkpoint every N epochs
    save_best: bool = True  # Save best model based on validation metric
    
    # Logging
    log_freq: int = 10  # Log metrics every N batches
    use_tensorboard: bool = True
    use_wandb: bool = False  # Weights & Biases logging
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Create directories if they don't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set device
        import torch
        if self.device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            self.device = 'cpu'
    
    def get_checkpoint_path(self, epoch: Optional[int] = None, best: bool = False) -> str:
        """Get checkpoint file path."""
        if best:
            return os.path.join(self.checkpoint_dir, f"{self.experiment_name}_best.pth")
        elif epoch is not None:
            return os.path.join(self.checkpoint_dir, f"{self.experiment_name}_epoch_{epoch}.pth")
        else:
            return os.path.join(self.checkpoint_dir, f"{self.experiment_name}_latest.pth")
    
    def get_log_path(self) -> str:
        """Get log file path."""
        return os.path.join(self.log_dir, f"{self.experiment_name}.log")

