"""
Learning rate schedulers for training.
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import math


def get_optimizer(model, config):
    """
    Create AdamW optimizer.
    
    Args:
        model: PyTorch model
        config: Config object with optimizer settings
    
    Returns:
        Optimizer instance
    """
    return optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


class WarmupCosineScheduler:
    """
    Learning rate scheduler with linear warmup followed by cosine annealing.
    
    Schedule:
    - Linear warmup from warmup_start_lr to learning_rate over warmup_epochs
    - Cosine annealing from learning_rate to min_lr over remaining epochs
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        num_epochs: int,
        warmup_epochs: int,
        warmup_start_lr: float,
        min_lr: float,
    ):
        """
        Args:
            optimizer: Optimizer instance
            num_epochs: Total number of training epochs
            warmup_epochs: Number of warmup epochs
            warmup_start_lr: Starting learning rate for warmup
            min_lr: Minimum learning rate for cosine annealing
        """
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        
        # Create lambda function for learning rate schedule
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Linear warmup
                return warmup_start_lr / self.base_lr + (epoch / warmup_epochs) * (1 - warmup_start_lr / self.base_lr)
            else:
                # Cosine annealing
                progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                return self.min_lr / self.base_lr + (1 - self.min_lr / self.base_lr) * cosine_decay
        
        self.scheduler = LambdaLR(optimizer, lr_lambda)
    
    def step(self):
        """Update learning rate."""
        self.scheduler.step()
    
    def get_last_lr(self):
        """Get current learning rate."""
        return self.scheduler.get_last_lr()
    
    def state_dict(self):
        """Get scheduler state dict."""
        return self.scheduler.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load scheduler state dict."""
        self.scheduler.load_state_dict(state_dict)


def get_scheduler(optimizer, config):
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer instance
        config: Config object with scheduler settings
    
    Returns:
        Scheduler instance
    """
    return WarmupCosineScheduler(
        optimizer=optimizer,
        num_epochs=config.num_epochs,
        warmup_epochs=config.warmup_epochs,
        warmup_start_lr=config.warmup_start_lr,
        min_lr=config.min_lr,
    )

