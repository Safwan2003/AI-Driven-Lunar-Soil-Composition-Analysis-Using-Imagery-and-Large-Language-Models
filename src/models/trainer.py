"""
Model Trainer Module
====================

Handles training of lunar soil classification models.
"""

from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim


class ModelTrainer:
    """
    Train and evaluate lunar soil classification models.
    
    Provides utilities for training, validation, and checkpointing.
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        device: str = "cuda"
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            learning_rate: Learning rate for optimizer
            device: Device to train on (cuda/cpu)
        """
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self, dataloader) -> float:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        return {
            "loss": total_loss / len(dataloader),
            "accuracy": 100.0 * correct / total
        }
