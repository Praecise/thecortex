import pytest
import json
import os
from datetime import datetime
from cortex.utils.logging import LogManager

@pytest.mark.asyncio
async def test_logging_initialization(temp_dir):
    """Test log manager initialization"""
    log_file = os.path.join(temp_dir, "test.log")
    log_manager = LogManager(log_file=log_file)
    
    assert os.path.exists(log_file), "Log file should be created"

@pytest.mark.asyncio
async def test_training_event_logging(temp_dir):
    """Test logging of training events"""
    log_file = os.path.join(temp_dir, "test.log")
    log_manager = LogManager(log_file=log_file)
    
    test_metrics = {'accuracy': 0.95, 'loss': 0.1}
    log_manager.training_event('test', test_metrics)
    
    with open(log_file, 'r') as f:
        log_content = f.read()
        assert 'training' in log_content
        assert 'accuracy' in log_content
        assert '0.95' in log_content
        assert 'loss' in log_content
        assert '0.1' in log_content

@pytest.mark.asyncio
async def test_network_event_logging(temp_dir):
    """Test logging of network events"""
    log_file = os.path.join(temp_dir, "test.log")
    log_manager = LogManager(log_file=log_file)
    
    log_manager.network_event('connection', 'node_1', 'connected')
    
    with open(log_file, 'r') as f:
        log_content = f.read()
        assert 'network' in log_content
        assert 'node_1' in log_content
        assert 'connected' in log_content

@pytest.mark.asyncio
async def test_error_logging(temp_dir):
    """Test logging of errors"""
    log_file = os.path.join(temp_dir, "test.log")
    log_manager = LogManager(log_file=log_file)
    
    test_error = "Test error message"
    log_manager.error('test_error', test_error)
    
    with open(log_file, 'r') as f:
        log_content = f.read()
        assert 'error' in log_content
        assert 'test_error' in log_content
        assert test_error in log_content

@pytest.mark.asyncio
async def test_log_format(temp_dir):
    """Test log message format"""
    log_file = os.path.join(temp_dir, "test.log")
    log_manager = LogManager(log_file=log_file)
    
    test_metrics = {'accuracy': 1.0}
    log_manager.training_event('test', test_metrics)
    
    with open(log_file, 'r') as f:
        log_content = f.readlines()[-1]  # Get last line
        log_data = json.loads(log_content.split(' - ')[-1])
        
        assert 'timestamp' in log_data
        assert 'event' in log_data
        assert 'type' in log_data
        assert 'metrics' in log_data
        
        # Verify timestamp format
        datetime.fromisoformat(log_data['timestamp'])  # Should not raise exception