# cortex/utils/validation.py

import torch
from typing import Dict, Any, List

class ModelValidator:
    """Validates model weights and updates"""
    
    @staticmethod
    def validate_weights(weights: Dict[str, torch.Tensor], expected_keys: List[str]) -> bool:
        """
        Validate model weights
        
        Args:
            weights: Model state dict
            expected_keys: Expected keys in state dict
        """
        # Check all expected keys are present
        if not all(key in weights for key in expected_keys):
            return False
            
        # Check tensor validity
        for tensor in weights.values():
            if not isinstance(tensor, torch.Tensor):
                return False
            if torch.isnan(tensor).any():
                return False
            if torch.isinf(tensor).any():
                return False
                
        return True
    
    @staticmethod
    def validate_update(
        old_weights: Dict[str, torch.Tensor],
        new_weights: Dict[str, torch.Tensor],
        max_diff_threshold: float = 10.0
    ) -> bool:
        """
        Validate weight update magnitude
        
        Args:
            old_weights: Previous weights
            new_weights: New weights
            max_diff_threshold: Maximum allowed parameter change
        """
        for key in old_weights:
            if key not in new_weights:
                return False
                
            # Compute max absolute difference
            max_diff = (new_weights[key] - old_weights[key]).abs().max().item()
            
            if max_diff > max_diff_threshold:
                return False
                
        return True
    
    @staticmethod
    def compute_update_magnitude(
        old_weights: Dict[str, torch.Tensor],
        new_weights: Dict[str, torch.Tensor]
    ) -> float:
        """
        Compute magnitude of weight update
        
        Args:
            old_weights: Previous weights
            new_weights: New weights
        """
        total_diff = 0.0
        total_params = 0
        
        for key in old_weights:
            diff = (new_weights[key] - old_weights[key]).abs()
            total_diff += diff.sum().item()
            total_params += diff.numel()
            
        return total_diff / total_params if total_params > 0 else 0.0