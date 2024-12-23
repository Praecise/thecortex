# tests/test_integration.py

import pytest
import pytest_asyncio
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
import torch

from cortex.api.endpoints import app
from cortex.network.protocol import TenzroProtocol, MessageType
from cortex.network.discovery import NodeDiscoveryService
from cortex.coordination.job_coordinator import JobCoordinator
from cortex.deployment.manager import ModelDeploymentManager

@pytest.fixture
def test_client():
    return TestClient(app)

@pytest.fixture
def mock_protocol():
    protocol = AsyncMock(spec=TenzroProtocol)
    protocol.node_id = "test_node"
    protocol.node_type = "local"
    protocol.region = "test_region"
    return protocol

@pytest.fixture
def mock_discovery():
    discovery = AsyncMock(spec=NodeDiscoveryService)
    discovery.nodes = {}
    discovery.active_regions = set()
    return discovery

@pytest.fixture
def test_model_config():
    return {
        "input_dim": 784,
        "hidden_dims": [512, 256],
        "output_dim": 10,
        "model_type": "feedforward"
    }

@pytest.fixture
def test_training_config():
    return {
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 2,
        "optimizer": "adam",
        "loss_function": "cross_entropy"
    }

class TestJobSubmission:
    @pytest_asyncio.fixture
    async def setup_job(self, test_client, mock_protocol, test_model_config, test_training_config):
        global_nodes = ["ws://test-global-1:8080", "ws://test-global-2:8080"]
        
        with patch('cortex.api.endpoints.TenzroProtocol', return_value=mock_protocol):
            response = test_client.post(
                "/api/v1/jobs/submit",
                json={
                    "neural_model_config": {"test": "config"},
                    "training_config": {"test": "config"},
                    "global_node_urls": ["ws://test-global-1:8080"]
                }
            )
        return response, mock_protocol

    async def test_job_submission(self, setup_job):
        response, mock_protocol = setup_job
        
        assert response.status_code == 200
        assert "job_id" in response.json()
        mock_protocol.submit_job.assert_called_once()

    async def test_job_status_monitoring(self, test_client, setup_job):
        initial_response, mock_protocol = setup_job
        job_id = initial_response.json()["job_id"]
        
        # Test status endpoint
        response = test_client.get(f"/api/v1/jobs/{job_id}/status")
        assert response.status_code == 200
        status_data = response.json()
        assert "status" in status_data
        assert "metrics" in status_data

class TestNetworkMonitoring:
    @pytest_asyncio.fixture
    async def setup_network(self, mock_discovery):
        # Setup mock network data
        mock_discovery.nodes = {
            "node1": Mock(
                node_id="node1",
                node_type="local",
                region="us-east",
                available=True,
                capacity=Mock(
                    cpu_cores=4,
                    memory_total=8589934592,  # 8GB
                    gpu_memory=4294967296,    # 4GB
                    disk_space=107374182400   # 100GB
                )
            ),
            "node2": Mock(
                node_id="node2",
                node_type="local",
                region="us-west",
                available=True,
                capacity=Mock(
                    cpu_cores=8,
                    memory_total=17179869184,  # 16GB
                    gpu_memory=None,
                    disk_space=214748364800    # 200GB
                )
            )
        }
        mock_discovery.active_regions = {"us-east", "us-west"}
        return mock_discovery

    async def test_network_status(self, test_client, setup_network):
        with patch('cortex.api.endpoints.services', {"discovery": setup_network}):
            response = test_client.get("/api/v1/network/status")
            assert response.status_code == 200
            data = response.json()
            
            assert "regions" in data
            assert "total_nodes" in data
            assert data["total_nodes"] == 2
            assert len(data["regions"]) == 2

    async def test_resource_availability(self, test_client, setup_network):
        with patch('cortex.api.endpoints.services', {"discovery": setup_network}):
            response = test_client.get("/api/v1/network/resources")
            assert response.status_code == 200
            data = response.json()
            
            assert data["available_nodes"] == 2
            assert data["total_cpu_cores"] == 12
            assert data["gpu_nodes"] == 1
            assert len(data["regions"]) == 2

class TestModelDeployment:
    @pytest_asyncio.fixture
    async def setup_deployment(self, mock_protocol, test_model_config):
        deployment_manager = ModelDeploymentManager("local", "test_node")
        
        # Create test job
        job_id = "test_job_123"
        target_nodes = ["node1", "node2"]
        
        return deployment_manager, job_id, target_nodes, test_model_config

    async def test_model_deployment(self, setup_deployment):
        deployment_manager, job_id, target_nodes, model_config = setup_deployment
        
        # Test deployment
        success = await deployment_manager.deploy_model(
            job_id,
            model_config,
            target_nodes
        )
        assert success
        
        # Verify model is tracked
        assert job_id in deployment_manager.deployed_models
        
        # Test cleanup
        await deployment_manager.cleanup_deployment(job_id)
        assert job_id not in deployment_manager.deployed_models

class TestProtocolMessages:
    async def test_message_handling(self, mock_protocol):
        # Test join message
        join_message = {
            "type": MessageType.JOIN.value,
            "sender_id": "test_node",
            "sender_type": "local",
            "region": "test_region",
            "data": {
                "resources": {
                    "cpu_cores": 4,
                    "memory_total": 8589934592
                }
            }
        }
        
        await mock_protocol.handle_message(join_message)
        mock_protocol._handle_join.assert_called_once_with(join_message)
        
        # Test training metrics
        metrics_message = {
            "type": MessageType.TRAINING_METRICS.value,
            "sender_id": "test_node",
            "data": {
                "job_id": "test_job",
                "metrics": {
                    "loss": 0.5,
                    "accuracy": 0.95
                }
            }
        }
        
        await mock_protocol.handle_message(metrics_message)
        assert mock_protocol.report_training_metrics.called

def test_health_check(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "version" in data