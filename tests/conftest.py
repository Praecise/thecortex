import pytest
import pytest_asyncio
import asyncio
import tempfile
import os
from typing import Generator, Any
import torch
import yaml
from cortex.core.model import CortexModel
from cortex.core.training import ModelTrainer
from cortex.config.settings import CortexConfig
from cortex.utils.logging import LogManager
from cortex.security.encryption import EncryptionManager

@pytest_asyncio.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest_asyncio.fixture
async def model():
    """Create test model"""
    model = CortexModel(
        input_dim=784,
        hidden_dims=[512, 256],
        output_dim=10
    )
    return model

@pytest.fixture(scope="session")
def temp_dir() -> Generator[str, None, None]:
    """Create temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir

@pytest.fixture
def config(temp_dir: str) -> CortexConfig:
    """Create test configuration"""
    config_data = {
        'network': {
            'websocket_url': 'ws://test',
            'node_id': 'test_node',
            'retry_attempts': 3,
            'connection_timeout': 5
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 0.001,
            'hidden_dims': [512, 256],
            'dropout_rate': 0.2,
            'input_dim': 784,
            'output_dim': 10,
            'max_epochs': 2,
            'checkpoint_interval': 10,
            'checkpoint_dir': os.path.join(temp_dir, 'checkpoints')
        },
        'security': {
            'enable_encryption': False
        },
        'monitoring': {
            'log_level': 'INFO',
            'log_file': os.path.join(temp_dir, 'test.log')
        }
    }
    
    config_path = os.path.join(temp_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f)
    
    return CortexConfig(config_path)

@pytest.fixture
def log_manager(temp_dir: str) -> LogManager:
    """Create test log manager"""
    return LogManager(
        log_file=os.path.join(temp_dir, 'test.log'),
        level='DEBUG'
    )

@pytest.fixture
def encryption_manager() -> EncryptionManager:
    """Create test encryption manager"""
    return EncryptionManager()

@pytest.fixture
def test_data() -> tuple:
    """Create test dataset"""
    x = torch.randn(100, 784)
    y = torch.randint(0, 10, (100,))
    return x, y

@pytest_asyncio.fixture
async def trainer(model):
    """Create test trainer"""
    trainer = ModelTrainer(model)
    return trainer

@pytest.fixture(autouse=True)
def cleanup_files(temp_dir):
    """Cleanup any test files after each test"""
    yield
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            if file.endswith('.log') or file.endswith('.pt'):
                os.remove(os.path.join(root, file))

@pytest.fixture(autouse=True)
async def cleanup_cuda():
    """Cleanup CUDA memory after each test"""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()