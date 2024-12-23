# cortex/aggregation/weights.py

import torch
from typing import List, Dict, Any, Optional
import numpy as np

class WeightAggregator:
    """Aggregates weights from multiple nodes"""
    
    @staticmethod
    async def aggregate(
        weights_list: List[Dict[str, torch.Tensor]],
        weights: Optional[List[float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate weights using weighted average
        
        Args:
            weights_list: List of model state dicts
            weights: Optional weights for each model
        """
        if not weights_list:
            raise ValueError("No weights to aggregate")
        
        # Use equal weights if none provided
        if weights is None:
            weights = [1.0 / len(weights_list)] * len(weights_list)
        
        if len(weights) != len(weights_list):
            raise ValueError("Number of weights must match number of models")
        
        # Initialize aggregated weights with zeros
        aggregated = {}
        for key in weights_list[0].keys():
            aggregated[key] = torch.zeros_like(weights_list[0][key])
            
        # Compute weighted average
        for w_dict, weight in zip(weights_list, weights):
            for key in aggregated.keys():
                aggregated[key] += w_dict[key] * weight
                
        return aggregated
    
    @staticmethod
    async def aggregate_with_scores(
        weights_list: List[Dict[str, torch.Tensor]],
        performance_scores: List[float]
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate weights using performance-based weighting
        
        Args:
            weights_list: List of model state dicts
            performance_scores: Performance metric for each model
        """
        # Normalize scores to weights
        total_score = sum(performance_scores)
        if total_score == 0:
            # Fall back to equal weights if all scores are 0
            weights = [1.0 / len(weights_list)] * len(weights_list)
        else:
            weights = [score / total_score for score in performance_scores]
            
        return await WeightAggregator.aggregate(weights_list, weights)

# cortex/utils/serialization.py

import torch
import base64
import io
from typing import Dict, Any

def serialize_tensor(tensor: torch.Tensor) -> Dict[str, Any]:
    """Serialize a PyTorch tensor for network transmission"""
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return {
        "data": base64.b64encode(buffer.getvalue()).decode('utf-8'),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype)
    }

def deserialize_tensor(data: Dict[str, Any]) -> torch.Tensor:
    """Deserialize a PyTorch tensor from network transmission"""
    buffer = io.BytesIO(base64.b64decode(data["data"]))
    tensor = torch.load(buffer)
    return tensor

def serialize_model_weights(weights: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Serialize entire model state dict"""
    return {key: serialize_tensor(tensor) for key, tensor in weights.items()}

def deserialize_model_weights(data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Deserialize entire model state dict"""
    return {key: deserialize_tensor(tensor_data) for key, tensor_data in data.items()}