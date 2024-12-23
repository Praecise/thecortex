import zlib
from typing import Dict, Any, List
import asyncio
import torch
from torch import Tensor
import numpy as np
import pickle
from ..utils.logging import log_manager

class WeightCompressor:
    """Handles compression of model weights for efficient transmission"""
    
    def __init__(self, compression_level: int = 6):
        self.compression_level = compression_level
    
    def compress_weights(self, weights: Dict[str, Tensor]) -> bytes:
        """Compress model weights"""
        # Convert tensors to numpy and store metadata
        weight_data = {}
        for name, tensor in weights.items():
            weight_data[name] = {
                'data': tensor.cpu().numpy(),
                'dtype': str(tensor.dtype),
                'shape': tensor.shape
            }
            
        # Serialize with pickle and compress
        serialized = pickle.dumps(weight_data)
        return zlib.compress(serialized, level=self.compression_level)
    
    def decompress_weights(self, compressed_data: bytes) -> Dict[str, Tensor]:
        """Decompress model weights"""
        # Decompress and deserialize
        decompressed = zlib.decompress(compressed_data)
        weight_data = pickle.loads(decompressed)
        
        # Reconstruct tensors
        weights = {}
        for name, data in weight_data.items():
            tensor = torch.from_numpy(data['data'].copy())  # Make array writable
            weights[name] = tensor
            
        return weights

class BatchProcessor:
    """Handles batch processing of weight updates"""
    
    def __init__(self, batch_size: int = 10, max_wait: float = 1.0):
        self.batch_size = batch_size
        self.max_wait = max_wait
        self.batch: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._batch_event = asyncio.Event()
        self._running = False
    
    async def add_to_batch(self, weights: Dict[str, Any]) -> None:
        """Add weights to current batch"""
        async with self._lock:
            self.batch.append(weights)
            if len(self.batch) >= self.batch_size:
                self._batch_event.set()
    
    async def process_batch(self) -> Dict[str, Any]:
        """Process accumulated weights"""
        if not self.batch:
            return None
            
        async with self._lock:
            weights_to_process = self.batch.copy()
            self.batch.clear()
            self._batch_event.clear()
        
        # Average weights in batch
        averaged_weights = {}
        try:
            for key in weights_to_process[0].keys():
                tensors = [weights[key] for weights in weights_to_process]
                averaged_weights[key] = torch.stack(tensors).mean(dim=0)
            
            log_manager.training_event(
                'batch_processed',
                {'batch_size': len(weights_to_process)}
            )
        except Exception as e:
            log_manager.error(
                'batch_processing_error',
                str(e)
            )
            
        return averaged_weights
    
    async def batch_worker(self) -> None:
        """Background worker for batch processing"""
        self._running = True
        while self._running:
            try:
                # Wait for batch to fill or timeout
                try:
                    await asyncio.wait_for(
                        self._batch_event.wait(),
                        timeout=self.max_wait
                    )
                except asyncio.TimeoutError:
                    pass
                
                if self.batch:
                    await self.process_batch()
                    
            except Exception as e:
                log_manager.error(
                    'batch_worker_error',
                    str(e)
                )
                await asyncio.sleep(1)  # Prevent tight loop on error
    
    async def stop(self) -> None:
        """Stop the batch worker"""
        self._running = False
        self._batch_event.set()  # Wake up worker to stop

class CacheManager:
    """Manages caching of frequent computations"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Any:
        """Get value from cache"""
        async with self._lock:
            return self.cache.get(key)
    
    async def set(self, key: str, value: Any) -> None:
        """Set value in cache"""
        async with self._lock:
            if len(self.cache) >= self.max_size:
                # Remove oldest entry
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            self.cache[key] = value
            
    async def clear(self) -> None:
        """Clear cache"""
        async with self._lock:
            self.cache.clear()