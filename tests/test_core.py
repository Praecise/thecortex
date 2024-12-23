import pytest
import pytest_asyncio
import torch
import asyncio
import json
from typing import List, Dict, Any
import os
import tempfile
from unittest.mock import AsyncMock, patch, MagicMock
from torch.utils.data import DataLoader, TensorDataset

from cortex.core.model import CortexModel
from cortex.core.training import ModelTrainer
from cortex.network.bridge import TenzroBridge
from cortex.network.protocol import TenzroProtocol
from cortex.utils.metrics import MetricsTracker
from cortex.security.encryption import EncryptionManager
from cortex.optimization.performance import WeightCompressor, BatchProcessor

class MockWebSocket:
    """Mock websocket for testing"""
    def __init__(self):
        self.messages = []
        self.closed = False
        self.connected = True
        
    async def send(self, message):
        self.messages.append(message)
        
    async def recv(self):
        if not self.messages:
            await asyncio.sleep(0.1)
            raise ConnectionError("Mock receive error")
        return self.messages.pop(0)
        
    async def close(self):
        self.closed = True
        self.connected = False

@pytest.fixture
def model() -> CortexModel:
    """Create test model"""
    return CortexModel(
        input_dim=784,
        hidden_dims=[512, 256],
        output_dim=10
    )

@pytest.fixture
def trainer(model: CortexModel) -> ModelTrainer:
    """Create test trainer"""
    return ModelTrainer(model)

@pytest.fixture
def test_data() -> tuple:
    """Create test dataset"""
    x = torch.randn(100, 784)
    y = torch.randint(0, 10, (100,))
    return x, y

@pytest.fixture
def dataloader(test_data) -> DataLoader:
    """Create test dataloader"""
    x, y = test_data
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=32, shuffle=True)

@pytest.fixture
def mock_websocket():
    """Create mock websocket"""
    return MockWebSocket()

@pytest.mark.asyncio
async def test_model_training(model: CortexModel, trainer: ModelTrainer, test_data):
    """Test basic model training"""
    x, y = test_data
    metrics = await trainer.train_batch(x, y)
    
    assert 'loss' in metrics
    assert 'accuracy' in metrics
    assert isinstance(metrics['loss'], float)
    assert isinstance(metrics['accuracy'], float)
    assert 0 <= metrics['accuracy'] <= 1
    assert metrics['loss'] >= 0

@pytest.mark.asyncio
async def test_model_validation(model: CortexModel, trainer: ModelTrainer, test_data):
    """Test model validation"""
    x, y = test_data
    metrics = await trainer.validate(x, y)
    
    assert 'val_loss' in metrics
    assert 'val_accuracy' in metrics
    assert isinstance(metrics['val_loss'], float)
    assert isinstance(metrics['val_accuracy'], float)
    assert 0 <= metrics['val_accuracy'] <= 1
    assert metrics['val_loss'] >= 0

def test_model_architecture(model: CortexModel):
    """Test model architecture"""
    assert model.input_dim == 784
    assert model.hidden_dims == [512, 256]
    assert model.output_dim == 10
    
    batch_size = 32
    x = torch.randn(batch_size, 784)
    output = model(x)
    assert output.shape == (batch_size, 10)

@pytest.mark.asyncio
async def test_weight_compression(model: CortexModel):
    """Test weight compression and decompression"""
    compressor = WeightCompressor()
    weights = model.get_weights()
    
    # Test compression
    compressed = compressor.compress_weights(weights)
    assert isinstance(compressed, bytes)
    
    # Test decompression
    decompressed = compressor.decompress_weights(compressed)
    assert set(weights.keys()) == set(decompressed.keys())
    
    # Verify weights are preserved
    for key in weights:
        assert torch.allclose(weights[key], decompressed[key], rtol=1e-5)

@pytest.mark.asyncio
async def test_batch_processing():
    """Test batch processing of weights"""
    batch_processor = BatchProcessor(batch_size=2, max_wait=0.1)
    
    # Create test weights
    weights1 = {'layer1.weight': torch.ones(10, 10)}
    weights2 = {'layer1.weight': torch.ones(10, 10) * 2}
    
    # Add weights to batch
    await batch_processor.add_to_batch(weights1)
    await batch_processor.add_to_batch(weights2)
    
    # Process batch
    result = await batch_processor.process_batch()
    assert result is not None
    assert torch.allclose(result['layer1.weight'], torch.ones(10, 10) * 1.5)
    
    # Cleanup
    await batch_processor.stop()

@pytest.mark.asyncio
async def test_encryption():
    """Test weight encryption"""
    encryption_manager = EncryptionManager()
    
    # Create test data
    test_data = {
        'layer1.weight': torch.randn(10, 10).numpy().tobytes(),
        'layer1.bias': torch.randn(10).numpy().tobytes()
    }
    
    # Test encryption/decryption
    encrypted = encryption_manager.encrypt_weights(test_data)
    decrypted = encryption_manager.decrypt_weights(encrypted)
    
    # Verify data is preserved
    assert set(test_data.keys()) == set(decrypted.keys())
    for key in test_data:
        assert test_data[key] == decrypted[key]

@pytest.mark.asyncio
async def test_distributed_training(mock_websocket):
    """Test distributed training setup"""
    num_nodes = 3
    nodes = []
    
    with patch('websockets.connect', AsyncMock(return_value=mock_websocket)):
        try:
            # Create nodes
            for i in range(num_nodes):
                protocol = TenzroProtocol(f"node_{i}")
                bridge = TenzroBridge(
                    websocket_url="ws://test",
                    node_id=f"node_{i}",
                    node_type="test_type",
                    region="test_region",
                    protocol=protocol
                )
                nodes.append(bridge)
                await bridge.connect()
                assert bridge.connected
            
            # Verify join messages were sent
            assert len(mock_websocket.messages) == num_nodes
            
            # Parse and verify messages
            for message in mock_websocket.messages:
                data = json.loads(message)
                assert data["type"] == "join"
                assert data["sender_id"].startswith("node_")
                assert "timestamp" in data
            
        finally:
            # Cleanup
            for node in nodes:
                if node.connected:
                    await node.disconnect()
            
            # Verify disconnect messages
            leave_messages = [
                msg for msg in mock_websocket.messages
                if isinstance(msg, str) and "leave" in msg
            ]
            assert len(leave_messages) == num_nodes

def test_metrics_tracking():
    """Test metrics tracking"""
    tracker = MetricsTracker()
    
    # Test metrics update
    test_metrics = {'loss': 0.5, 'accuracy': 0.95}
    tracker.update(test_metrics)
    
    # Verify metrics were recorded
    assert 'loss' in tracker.metrics
    assert 'accuracy' in tracker.metrics
    assert len(tracker.metrics['loss']) == 1
    assert len(tracker.metrics['accuracy']) == 1
    assert tracker.metrics['loss'][0] == 0.5
    assert tracker.metrics['accuracy'][0] == 0.95
    
    # Test moving average
    assert tracker.get_moving_average('loss') == 0.5
    assert tracker.get_moving_average('accuracy') == 0.95