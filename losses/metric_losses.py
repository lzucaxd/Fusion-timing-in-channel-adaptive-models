"""
Metric learning losses for supervised training.

Implements ProxyNCA++-style loss for metric learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProxyNCA(nn.Module):
    """
    ProxyNCA++-style metric learning loss.
    
    Maintains learnable proxy vectors per class and uses cross-entropy
    over similarities to proxies.
    """
    
    def __init__(self, embed_dim: int, num_classes: int, temperature: float = 0.05):
        """
        Args:
            embed_dim: Dimension of input embeddings
            num_classes: Number of classes
            temperature: Temperature parameter for scaling similarities
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.temperature = temperature
        
        # Learnable proxies: one per class
        self.proxies = nn.Parameter(torch.randn(num_classes, embed_dim))
        
        # Initialize proxies
        nn.init.xavier_uniform_(self.proxies)
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute ProxyNCA++ loss.
        
        Args:
            embeddings: (B, embed_dim) normalized embeddings
            labels: (B,) integer class labels
        
        Returns:
            loss: scalar loss value
        """
        # Normalize embeddings and proxies
        embeddings = F.normalize(embeddings, p=2, dim=1)
        proxies = F.normalize(self.proxies, p=2, dim=1)
        
        # Compute similarities: (B, embed_dim) @ (embed_dim, num_classes) -> (B, num_classes)
        similarities = embeddings @ proxies.T
        
        # Scale by temperature
        logits = similarities / self.temperature
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        
        return loss
    
    def get_proxies(self) -> torch.Tensor:
        """Get normalized proxy vectors."""
        return F.normalize(self.proxies, p=2, dim=1)
