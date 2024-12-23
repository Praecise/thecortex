import functools
import asyncio
from typing import Any, Callable, Dict, Optional, Type, TypeVar
import torch
from datetime import datetime
import json
import os

# Type variable for function return type
T = TypeVar('T')

class CortexError(Exception):
    """Base exception for all Cortex errors"""
    pass

class NetworkError(CortexError):
    """Network-related errors"""
    pass

class ValidationError(CortexError):
    """Data validation errors"""
    pass

class CheckpointError(CortexError):
    """Checkpoint-related errors"""
    pass

class RecoveryManager:
    """Manages error recovery and checkpointing"""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    async def save_checkpoint(
        self,
        model_state: Dict[str, Any],
        optimizer_state: Dict[str, Any],
        training_state: Dict[str, Any],
        iteration: int
    ):
        """Save training checkpoint"""
        try:
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"checkpoint_{iteration}.pt"
            )
            metadata_path = os.path.join(
                self.checkpoint_dir,
                f"metadata_{iteration}.json"
            )
            
            # Save model and optimizer state
            torch.save({
                'model_state': model_state,
                'optimizer_state': optimizer_state,
                'training_state': training_state,
                'iteration': iteration
            }, checkpoint_path)
            
            # Save metadata
            metadata = {
                'iteration': iteration,
                'timestamp': datetime.utcnow().isoformat(),
                'training_state': training_state
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
                
        except Exception as e:
            raise CheckpointError(f"Failed to save checkpoint: {str(e)}")
    
    async def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load most recent checkpoint"""
        try:
            checkpoints = [f for f in os.listdir(self.checkpoint_dir)
                          if f.startswith("checkpoint_") and f.endswith(".pt")]
            
            if not checkpoints:
                return None
            
            # Get latest checkpoint
            latest = max(checkpoints, key=lambda x: int(x.split('_')[1].split('.')[0]))
            checkpoint_path = os.path.join(self.checkpoint_dir, latest)
            
            return torch.load(checkpoint_path)
            
        except Exception as e:
            raise CheckpointError(f"Failed to load checkpoint: {str(e)}")

def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 10.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (NetworkError,)
):
    """Decorator for retry logic with exponential backoff"""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries - 1:
                        raise
                    
                    await asyncio.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)
            
            raise last_exception
        return wrapper
    return decorator

def validate_weights(weights: Dict[str, torch.Tensor]) -> bool:
    """Validate model weights"""
    for name, tensor in weights.items():
        if not isinstance(tensor, torch.Tensor):
            raise ValidationError(f"Weight {name} is not a tensor")
        if torch.isnan(tensor).any():
            raise ValidationError(f"Weight {name} contains NaN values")
        if torch.isinf(tensor).any():
            raise ValidationError(f"Weight {name} contains infinite values")
    return True