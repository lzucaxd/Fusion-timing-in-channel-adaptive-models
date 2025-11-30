"""
Logging utilities for training and evaluation.
"""

import os
import csv
from typing import Dict, Any, Optional
from datetime import datetime
import torch


class CSVLogger:
    """CSV logger for metrics."""
    
    def __init__(self, log_path: str):
        """
        Args:
            log_path: Path to CSV log file
        """
        self.log_path = log_path
        self.fieldnames = None
        self.file = None
        self.writer = None
        
        # Create directory if needed
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    
    def log(self, metrics: Dict[str, Any], step: int):
        """
        Log metrics to CSV.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Step/epoch number
        """
        # Add step to metrics
        row = {'step': step, **metrics}
        
        # Initialize CSV writer on first log
        if self.file is None:
            self.fieldnames = list(row.keys())
            self.file = open(self.log_path, 'w', newline='')
            self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames, extrasaction='ignore')
            self.writer.writeheader()
        else:
            # Check if new fields appeared and update fieldnames
            new_fields = set(row.keys()) - set(self.fieldnames)
            if new_fields:
                # Read existing rows
                self.file.close()
                existing_rows = []
                if os.path.exists(self.log_path):
                    with open(self.log_path, 'r', newline='') as f:
                        reader = csv.DictReader(f)
                        existing_rows = list(reader)
                        if self.fieldnames is None:
                            self.fieldnames = reader.fieldnames or []
                
                # Add new fields to fieldnames
                self.fieldnames.extend(sorted(new_fields))
                
                # Rewrite file with updated header
                self.file = open(self.log_path, 'w', newline='')
                self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames, extrasaction='ignore')
                self.writer.writeheader()
                
                # Rewrite existing rows
                for existing_row in existing_rows:
                    self.writer.writerow(existing_row)
        
        # Write row (extrasaction='ignore' handles any extra fields)
        self.writer.writerow(row)
        self.file.flush()
    
    def close(self):
        """Close the log file."""
        if self.file is not None:
            self.file.close()


class TensorBoardLogger:
    """TensorBoard logger wrapper."""
    
    def __init__(self, log_dir: str):
        """
        Args:
            log_dir: Directory for TensorBoard logs
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=log_dir)
            self.enabled = True
        except ImportError:
            print("TensorBoard not available. Install with: pip install tensorboard")
            self.writer = None
            self.enabled = False
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        if self.enabled and self.writer is not None:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """Log multiple scalars."""
        if self.enabled and self.writer is not None:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def close(self):
        """Close the writer."""
        if self.enabled and self.writer is not None:
            self.writer.close()


class Logger:
    """Unified logger for training."""
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        use_tensorboard: bool = True,
        use_csv: bool = True,
    ):
        """
        Args:
            log_dir: Directory for logs
            experiment_name: Name of experiment
            use_tensorboard: Whether to use TensorBoard
            use_csv: Whether to use CSV logging
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        
        # Initialize loggers
        self.tb_logger = None
        if use_tensorboard:
            tb_log_dir = os.path.join(log_dir, experiment_name)
            self.tb_logger = TensorBoardLogger(tb_log_dir)
        
        self.csv_logger = None
        if use_csv:
            csv_path = os.path.join(log_dir, f"{experiment_name}.csv")
            self.csv_logger = CSVLogger(csv_path)
    
    def log(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """
        Log metrics.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Step/epoch number
            prefix: Optional prefix for metric names
        """
        # Add prefix to metric names
        prefixed_metrics = {f"{prefix}/{k}" if prefix else k: v for k, v in metrics.items()}
        
        # Log to TensorBoard
        if self.tb_logger is not None:
            for tag, value in prefixed_metrics.items():
                self.tb_logger.log_scalar(tag, value, step)
        
        # Log to CSV
        if self.csv_logger is not None:
            self.csv_logger.log(prefixed_metrics, step)
    
    def close(self):
        """Close all loggers."""
        if self.tb_logger is not None:
            self.tb_logger.close()
        if self.csv_logger is not None:
            self.csv_logger.close()


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """
    Print metrics to console in a formatted way.
    
    Args:
        metrics: Dictionary of metric names to values
        prefix: Optional prefix for display
    """
    if prefix:
        print(f"\n{prefix}:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

