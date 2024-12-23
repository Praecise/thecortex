# tests/test_network.py

import pytest
import pytest_asyncio
import asyncio
import json
import torch
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch
from typing import Dict, Any, Optional

from cortex.network.bridge import TenzroBridge
from cortex.network.protocol import TenzroProtocol, MessageType

class MockWebSocket:
    """Mock websocket for testing"""
    def __init__(self):
        self.messages = []
        self.closed = False
        self.connected = True
        self.error_on_send = False
        self.error_on_recv = False
        
    async def send(self, message: str) -> None:
        """Mock send method"""
        if self.error_on_send:
            raise ConnectionError("Mock send error")
        self.messages.append(message)
        
    async def recv(self) -> Optional[str]:
        """Mock receive method"""
        if self.error_on_recv:
            raise ConnectionError("Mock receive error")
        if not self.messages:
            await asyncio.sleep(0.1)
        return self.messages.pop(0) if self.messages else None
        
    async def close(self) -> None:
        """Mock close method"""
        self.closed = True
        self.connected = False

@pytest.fixture
def mock_websocket():
    """Create mock websocket"""
    return MockWebSocket()

@pytest.fixture
def protocol():
    """Create test protocol"""
    return TenzroProtocol("test_node", "test", "test-region")

@pytest_asyncio.fixture
async def bridge(mock_websocket):
    """Create test bridge"""
    with patch('websockets.connect', AsyncMock(return_value=mock_websocket)):
        bridge = TenzroBridge(
            websocket_url="ws://test",
            node_id="test_node",
            node_type="test",
            region="test-region",
            protocol=TenzroProtocol("test_node", "test", "test-region")
        )
        yield bridge
        if bridge.connected:
            await bridge.disconnect()

@pytest.mark.asyncio
async def test_connection(bridge: TenzroBridge, mock_websocket: MockWebSocket):
    """Test network connection"""
    await bridge.connect()
    
    assert bridge.connected
    assert len(mock_websocket.messages) > 0
    
    # Verify join message
    join_message = json.loads(mock_websocket.messages[0])
    assert join_message["type"] == MessageType.JOIN.value
    assert join_message["sender_id"] == "test_node"
    assert join_message["sender_type"] == "test"
    assert join_message["region"] == "test-region"
    assert "timestamp" in join_message

@pytest.mark.asyncio
async def test_disconnection(bridge: TenzroBridge, mock_websocket: MockWebSocket):
    """Test network disconnection"""
    await bridge.connect()
    assert bridge.connected
    
    await bridge.disconnect()
    assert not bridge.connected
    assert mock_websocket.closed
    
    # Verify leave message was sent
    last_message = json.loads(mock_websocket.messages[-1])
    assert last_message["type"] == MessageType.LEAVE.value
    assert last_message["sender_id"] == "test_node"

@pytest.mark.asyncio
async def test_connection_error(mock_websocket: MockWebSocket):
    """Test connection error handling"""
    mock_websocket.error_on_send = True
    
    with patch('websockets.connect', AsyncMock(return_value=mock_websocket)):
        bridge = TenzroBridge(
            websocket_url="ws://test",
            node_id="test_node",
            node_type="test",
            region="test-region"
        )
        
        with pytest.raises(ConnectionError):
            await bridge.connect()
        
        assert not bridge.connected

@pytest.mark.asyncio
async def test_message_sending(bridge: TenzroBridge, mock_websocket: MockWebSocket):
    """Test message sending"""
    await bridge.connect()
    
    # Test sending a message through protocol
    test_message = await bridge.protocol.send_message(
        MessageType.TRAINING_METRICS,
        {"data": "test_data"}
    )
    await bridge.ws.send(json.dumps(test_message))
    
    sent_message = json.loads(mock_websocket.messages[-1])
    assert sent_message["type"] == MessageType.TRAINING_METRICS.value
    assert sent_message["data"] == {"data": "test_data"}

@pytest.mark.asyncio
async def test_weight_message(protocol: TenzroProtocol):
    """Test weight message handling"""
    # Create test weights
    test_weights = {
        "layer1.weight": torch.randn(10, 10),
        "layer1.bias": torch.randn(10)
    }
    
    # Test sending weights
    message = await protocol.send_weights(test_weights)
    assert message["type"] == MessageType.WEIGHTS_UPDATE.value
    assert message["sender_id"] == "test_node"
    assert "weights" in message["data"]
    
    # Test receiving weights
    received_weights = await protocol.receive_weights(message)
    for key in test_weights:
        assert torch.allclose(test_weights[key], received_weights[key])

@pytest.mark.asyncio
async def test_metrics_reporting(protocol: TenzroProtocol):
    # Test metrics
    test_metrics = {
        "loss": 0.5,
        "accuracy": 0.95,
        "learning_rate": 0.001
    }

    # Report metrics (test both legacy and new methods)
    legacy_message = await protocol.report_metrics(test_metrics)
    new_message = await protocol.report_training_metrics("test_job", test_metrics)

    # Verify legacy message
    assert legacy_message["type"] == MessageType.TRAINING_METRICS.value
    new_message = await protocol.report_training_metrics("test_job", test_metrics)
    
    # Verify legacy message
    assert legacy_message["type"] == MessageType.TRAINING_METRICS.value
    assert legacy_message["data"]["metrics"] == test_metrics
    
    # Verify new message
    assert new_message["type"] == MessageType.TRAINING_METRICS.value
    assert new_message["data"]["metrics"] == test_metrics
    assert new_message["data"]["job_id"] == "test_job"

@pytest.mark.asyncio
async def test_message_handling(protocol: TenzroProtocol):
    """Test message handling"""
    messages_received = []
    
    # Register mock handler
    async def mock_handler(message):
        messages_received.append(message)
    
    protocol.register_handler("test_type", mock_handler)
    
    # Test message
    test_message = {
        "type": "test_type",
        "data": "test_data",
        "timestamp": datetime.utcnow().isoformat()
    }
    
    await protocol.handle_message(test_message)
    
    assert len(messages_received) == 1
    assert messages_received[0] == test_message

@pytest.mark.asyncio
async def test_multiple_handlers(protocol: TenzroProtocol):
    """Test multiple message handlers"""
    handler1_messages = []
    handler2_messages = []
    
    async def handler1(message):
        handler1_messages.append(message)
        
    async def handler2(message):
        handler2_messages.append(message)
    
    # Register both handlers for same message type
    protocol.register_handler("test_type", handler1)
    protocol.register_handler("test_type", handler2)
    
    test_message = {
        "type": "test_type",
        "data": "test_data"
    }
    
    await protocol.handle_message(test_message)
    
    # Last registered handler should be used
    assert len(handler1_messages) == 0
    assert len(handler2_messages) == 1
    assert handler2_messages[0] == test_message

@pytest.mark.asyncio
async def test_invalid_message_handling(protocol: TenzroProtocol):
    """Test handling of invalid messages"""
    messages_handled = []
    
    async def test_handler(message):
        messages_handled.append(message)
    protocol.register_handler("test_type", test_handler)
    
    # Test various invalid messages
    invalid_messages = [
        {"data": "test_data"},  # Missing type
        {"type": "test_type"},  # Missing data
        {},  # Empty message
        None,  # None message
        {"type": 123, "data": "test"},  # Invalid type
    ]
    
    for message in invalid_messages:
        await protocol.handle_message(message)
    
    # Only the message with type "test_type" should be handled
    assert len(messages_handled) == 1

@pytest.mark.asyncio
async def test_resource_reporting(protocol: TenzroProtocol):
    """Test resource reporting"""
    # Get resource report
    message = await protocol.report_resources()
    
    assert message["type"] == MessageType.RESOURCE_STATUS.value
    assert "cpu_usage" in message["data"]
    assert "memory_usage" in message["data"]
    assert "gpu_usage" in message["data"]
    assert "disk_usage" in message["data"]
    assert "network_bandwidth" in message["data"]

@pytest.mark.asyncio
async def test_job_coordination(protocol: TenzroProtocol):
    """Test job coordination messages"""
    # Test job submission
    job_config = {"test": "config"}
    message = await protocol.submit_job(job_config)
    
    assert message["type"] == MessageType.JOB_SUBMIT.value
    assert message["data"]["config"] == job_config
    
    # Test job status update
    message = await protocol.update_job_status("test_job", "running", {"progress": 0.5})
    
    assert message["type"] == MessageType.JOB_STATUS.value
    assert message["data"]["job_id"] == "test_job"
    assert message["data"]["status"] == "running"
    assert message["data"]["metrics"]["progress"] == 0.5

@pytest.mark.asyncio
async def test_timestamp_generation(protocol: TenzroProtocol):
    """Test protocol timestamp generation"""
    timestamp = protocol._get_timestamp()
    
    # Verify timestamp format
    datetime.fromisoformat(timestamp)  # Should not raise
    
    # Verify timestamp is current
    timestamp_dt = datetime.fromisoformat(timestamp)
    now = datetime.utcnow()
    assert abs((now - timestamp_dt).total_seconds()) < 1