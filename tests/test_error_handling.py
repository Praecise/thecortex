# tests/test_error_handling.py

import pytest
import pytest_asyncio
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import torch
from datetime import datetime, timedelta
from fastapi.testclient import TestClient

from cortex.api.endpoints import app
from cortex.network.protocol import TenzroProtocol, MessageType
from cortex.network.discovery import NodeDiscoveryService
from cortex.coordination.job_coordinator import JobCoordinator

import pytest
from unittest.mock import AsyncMock
from cortex.network.protocol import TenzroProtocol
from cortex.coordination.job_coordinator import JobCoordinator
from cortex.network.discovery import NodeDiscoveryService

@pytest.fixture
def mock_protocol():
    return AsyncMock(spec=TenzroProtocol)

@pytest.fixture  
def mock_coordinator():
    return AsyncMock(spec=JobCoordinator)

@pytest.fixture
def mock_discovery():
    return AsyncMock(spec=NodeDiscoveryService)

class TestErrorHandling:
    @pytest.fixture
    def test_client(self):
        return TestClient(app)

    async def test_invalid_model_config(self, test_client):
        """Test submission with invalid model config"""
        invalid_config = {
            "model_config": {
                "input_dim": "invalid",  # Should be int
                "hidden_dims": [512, 256],
                "output_dim": 10
            },
            "training_config": {
                "batch_size": 32,
                "learning_rate": 0.001
            },
            "global_node_urls": ["ws://test:8080"]
        }
        
        response = test_client.post("/api/v1/jobs/submit", json=invalid_config)
        assert response.status_code == 422  # Validation error

    async def test_missing_global_nodes(self, test_client):
        """Test submission without global nodes"""
        config = {
            "model_config": {
                "input_dim": 784,
                "hidden_dims": [512, 256],
                "output_dim": 10
            },
            "training_config": {
                "batch_size": 32,
                "learning_rate": 0.001
            },
            "global_node_urls": []  # Empty list
        }
        
        response = test_client.post("/api/v1/jobs/submit", json=config)
        assert response.status_code == 500
        assert "Failed to submit job" in response.json()["detail"]

    async def test_network_timeout(self, test_client, mock_protocol):
        """Test handling of network timeouts"""
        mock_protocol.submit_job = AsyncMock(side_effect=asyncio.TimeoutError)
        
        with patch('cortex.api.endpoints.TenzroProtocolV2', return_value=mock_protocol):
            response = test_client.post(
                "/api/v1/jobs/submit",
                json={
                    "model_config": {"input_dim": 784, "hidden_dims": [512], "output_dim": 10},
                    "training_config": {"batch_size": 32, "learning_rate": 0.001},
                    "global_node_urls": ["ws://test:8080"]
                }
            )
            
        assert response.status_code == 500
        assert "timeout" in response.json()["detail"].lower()

    async def test_job_not_found(self, test_client):
        """Test requesting status for non-existent job"""
        response = test_client.get("/api/v1/jobs/nonexistent/status")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    async def test_results_not_ready(self, test_client, mock_coordinator):
        """Test requesting results before they're ready"""
        job_id = "test_job"
        mock_coordinator.jobs = {
            job_id: Mock(results=None, status="training")
        }
        
        with patch('cortex.api.endpoints.services', {"job_coordinator": mock_coordinator}):
            response = test_client.get(f"/api/v1/jobs/{job_id}/results")
            assert response.status_code == 400
            assert "not yet available" in response.json()["detail"].lower()

    async def test_node_failure_recovery(self, mock_discovery):
        """Test recovery from node failure"""
        # Simulate node failure
        failed_node = "node1"
        mock_discovery.nodes = {
            failed_node: Mock(
                node_id=failed_node,
                available=False,
                last_updated=datetime.utcnow() - timedelta(minutes=5)
            )
        }
        
        # Run cleanup
        await mock_discovery._remove_inactive_nodes()
        assert failed_node not in mock_discovery.nodes

    async def test_invalid_weight_update(self, mock_protocol):
        """Test handling of invalid weight updates"""
        invalid_weights = {
            "layer1.weight": torch.randn(10, 10),
            "layer2.weight": "invalid"  # Invalid tensor
        }
        
        with pytest.raises(ValueError):
            await mock_protocol.send_message(
                MessageType.WEIGHTS_UPDATE,
                {"weights": invalid_weights}
            )

    async def test_resource_overflow(self, mock_discovery):
        """Test handling of resource capacity overflow"""
        # Simulate resource overflow
        node = Mock(
            node_id="test_node",
            capacity=Mock(
                cpu_cores=4,
                memory_total=8589934592
            ),
            current_load={
                "cpu_usage": 120.0,  # Over 100%
                "memory_usage": 110.0
            }
        )
        
        mock_discovery.nodes["test_node"] = node
        available = await mock_discovery.get_available_nodes(
            min_cpu=10,
            min_memory=10
        )
        assert "test_node" not in available

    async def test_concurrent_job_submission(self, test_client):
        """Test handling of concurrent job submissions"""
        async def submit_job():
            return test_client.post(
                "/api/v1/jobs/submit",
                json={
                    "model_config": {"input_dim": 784, "hidden_dims": [512], "output_dim": 10},
                    "training_config": {"batch_size": 32, "learning_rate": 0.001},
                    "global_node_urls": ["ws://test:8080"]
                }
            )
        
        # Submit multiple jobs concurrently
        tasks = [submit_job() for _ in range(5)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all submissions were handled
        for response in responses:
            assert response.status_code in [200, 500]  # Either success or handled error

    async def test_invalid_region(self, test_client, mock_discovery):
        """Test handling of invalid region in network status"""
        mock_discovery.active_regions = {"invalid_region"}
        mock_discovery.nodes = {}
        
        with patch('cortex.api.endpoints.services', {"discovery": mock_discovery}):
            response = test_client.get("/api/v1/network/status")
            assert response.status_code == 200
            assert response.json()["total_nodes"] == 0

    async def test_protocol_version_mismatch(self, mock_protocol):
        """Test handling of protocol version mismatch"""
        message = {
            "version": "invalid",
            "type": "join",
            "sender_id": "test_node",
            "data": {}
        }
        
        with pytest.raises(ValueError):
            await mock_protocol.handle_message(message)