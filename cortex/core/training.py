# cortex/core/training.py

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional
import asyncio
from ..network.protocol import TenzroProtocol
from .model import CortexModel

class ModelTrainer:
    """Handles model training"""
    
    def __init__(
        self,
        model: CortexModel,
        learning_rate: float = 0.001,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
    async def train_batch(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        protocol: Optional[TenzroProtocol] = None
    ) -> Dict[str, float]:
        """Train on a single batch of data"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move data to device
        data = data.to(self.device)
        labels = labels.to(self.device)
        
        # Forward pass
        outputs = self.model(data)
        loss = self.criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            accuracy = (outputs.argmax(dim=1) == labels).float().mean().item()
        
        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy
        }
        
        # Report metrics if protocol is provided
        if protocol:
            await protocol.report_metrics(metrics)
        
        return metrics
    
    async def validate(
        self,
        data: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """Validate model performance"""
        self.model.eval()
        
        with torch.no_grad():
            data = data.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            accuracy = (outputs.argmax(dim=1) == labels).float().mean().item()
        
        return {
            "val_loss": loss.item(),
            "val_accuracy": accuracy
        }
    
    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint"""
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "model_config": self.model.get_config()
        }, path)
    
    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])