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