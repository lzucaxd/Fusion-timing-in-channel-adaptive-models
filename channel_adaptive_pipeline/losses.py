"""
Loss functions for channel-adaptive model training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing.
    
    Label smoothing helps prevent overfitting and improves generalization
    by encouraging the model to be less confident in its predictions.
    """
    
    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        """
        Args:
            smoothing: Label smoothing factor (0.0 = no smoothing, 1.0 = uniform distribution)
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super().__init__()
        assert 0.0 <= smoothing <= 1.0, f"Smoothing must be in [0, 1], got {smoothing}"
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing cross-entropy loss.
        
        Args:
            logits: Model predictions of shape (B, num_classes)
            targets: Ground truth labels of shape (B,) with class indices
        
        Returns:
            Loss value
        """
        num_classes = logits.size(-1)
        
        # Convert targets to one-hot encoding
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        # Compute cross-entropy
        log_probs = F.log_softmax(logits, dim=1)
        loss = -torch.sum(true_dist * log_probs, dim=1)
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

