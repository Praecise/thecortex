# cortex/core/model.py

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
import json

class CortexModel(nn.Module):
    """Base model for Cortex"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return {
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "output_dim": self.output_dim
        }
    
    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """Load weights into model"""
        new_state_dict = {}
        for name, tensor in weights.items():
            # Make sure we're working with new tensor instances
            if "network." not in name:
                name = f"network.{name}"
            new_state_dict[name] = tensor.clone().detach()
        self.load_state_dict(new_state_dict, strict=False)
    
    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Get model weights"""
        # Return a copy of the state dict to prevent accidental modifications
        return {name: tensor.clone().detach() for name, tensor in self.state_dict().items()}