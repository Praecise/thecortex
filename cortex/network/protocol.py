# cortex/network/protocol.py

from enum import Enum
from typing import Dict, Any, List, Optional, Callable, Awaitable
from dataclasses import dataclass
import json
from datetime import datetime

class MessageType(Enum):
    # Node Management
    JOIN = "join"
    LEAVE = "leave"
    HEARTBEAT = "heartbeat"
    
    # Job Coordination
    JOB_SUBMIT = "job_submit"
    JOB_STATUS = "job_status"
    JOB_RESULT = "job_result"
    
    # Model Management
    MODEL_DEPLOY = "model_deploy"
    MODEL_UPDATE = "model_update"
    WEIGHTS = "weights"  # Legacy support
    
    # Resource Management
    RESOURCE_STATUS = "resource_status"
    RESOURCE_REQUEST = "resource_request"
    
    # Training Coordination
    TRAINING_METRICS = "training_metrics"
    METRICS = "metrics"  # Legacy support
    WEIGHTS_UPDATE = "weights_update"
    AGGREGATION_REQUEST = "aggregation_request"
    AGGREGATION_RESPONSE = "aggregation_response"

@dataclass
class NodeInfo:
    node_id: str
    node_type: str
    region: str
    resources: Dict[str, Any]
    status: str
    last_seen: datetime

class TenzroProtocol:
    """Protocol for Tenzro network communication"""
    
    def __init__(self, node_id: str, node_type: str = "local", region: str = "default"):
        self.node_id = node_id
        self.node_type = node_type
        self.region = region
        self.nodes: Dict[str, NodeInfo] = {}
        self.message_handlers: Dict[str, Callable[[Dict[str, Any]], Awaitable[None]]] = {}
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default message handlers"""
        self.register_handler(MessageType.JOIN.value, self._handle_join)
        self.register_handler(MessageType.LEAVE.value, self._handle_leave)
        self.register_handler(MessageType.HEARTBEAT.value, self._handle_heartbeat)
        self.register_handler(MessageType.RESOURCE_STATUS.value, self._handle_resource_status)
        
        # Legacy handler mappings
        self.register_handler(MessageType.WEIGHTS.value, self._handle_weights)
        self.register_handler(MessageType.METRICS.value, self._handle_metrics)
    
    def register_handler(
        self,
        message_type: str,
        handler: Callable[[Dict[str, Any]], Awaitable[None]]
    ) -> None:
        """Register message handler"""
        self.message_handlers[message_type] = handler
    
    async def send_message(self, message_type: MessageType | str, data: Dict[str, Any],
                          target_id: Optional[str] = None) -> Dict[str, Any]:
        """Create and send message"""
        # Handle both MessageType enum and legacy string types
        msg_type = message_type.value if isinstance(message_type, MessageType) else message_type
        
        message = {
            "type": msg_type,
            "sender_id": self.node_id,
            "sender_type": self.node_type,
            "region": self.region,
            "target_id": target_id,
            "data": data,
            "timestamp": self._get_timestamp()
        }
        return message
    
    async def handle_message(self, message: Optional[Dict[str, Any]]) -> None:
        """Handle incoming message"""
        if not message or not isinstance(message, dict):
            return
            
        message_type = message.get("type")
        if not message_type:
            return
            
        handler = self.message_handlers.get(message_type)
        if handler:
            await handler(message)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        return datetime.utcnow().isoformat()
    
    # Node Management Handlers
    async def _handle_join(self, message: Dict[str, Any]):
        """Handle node join message"""
        node_id = message["sender_id"]
        self.nodes[node_id] = NodeInfo(
            node_id=node_id,
            node_type=message["sender_type"],
            region=message["region"],
            resources=message["data"].get("resources", {}),
            status="active",
            last_seen=datetime.utcnow()
        )
    
    async def _handle_leave(self, message: Dict[str, Any]):
        """Handle node leave message"""
        node_id = message["sender_id"]
        if node_id in self.nodes:
            del self.nodes[node_id]
    
    async def _handle_heartbeat(self, message: Dict[str, Any]):
        """Handle heartbeat message"""
        node_id = message["sender_id"]
        if node_id in self.nodes:
            self.nodes[node_id].last_seen = datetime.utcnow()
            self.nodes[node_id].resources = message["data"].get("resources", {})
    
    async def _handle_resource_status(self, message: Dict[str, Any]):
        """Handle resource status update"""
        node_id = message["sender_id"]
        if node_id in self.nodes:
            self.nodes[node_id].resources = message["data"]
    
    # Legacy Handlers (for backward compatibility)
    async def _handle_weights(self, message: Dict[str, Any]):
        """Handle legacy weights message"""
        await self._handle_weight_update(message)
    
    async def _handle_metrics(self, message: Dict[str, Any]):
        """Handle legacy metrics message"""
        await self._handle_training_metrics(message)
    
    # Legacy Methods (for backward compatibility)
    async def report_metrics(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Report training metrics (legacy method)"""
        return await self.report_training_metrics(None, metrics)
    
    async def send_weights(self, weights: Dict[str, Any]) -> Dict[str, Any]:
        """Send model weights (legacy method)"""
        message = await self.send_message(
            MessageType.WEIGHTS_UPDATE,
            {"weights": weights}
        )
        return message
    
    async def receive_weights(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Receive weights (legacy method)"""
        return message.get("data", {}).get("weights", {})
    
    # Job Coordination Methods
    async def submit_job(self, job_config: Dict[str, Any]) -> Dict[str, Any]:
        """Submit new training job"""
        return await self.send_message(
            MessageType.JOB_SUBMIT,
            {"config": job_config}
        )
    
    async def update_job_status(self, job_id: str, status: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Update job status"""
        return await self.send_message(
            MessageType.JOB_STATUS,
            {
                "job_id": job_id,
                "status": status,
                "metrics": metrics
            }
        )
    
    # Model Management Methods
    async def deploy_model(self, job_id: str, model_data: Dict[str, Any],
                          target_nodes: List[str]) -> Dict[str, Any]:
        """Deploy model to nodes"""
        return await self.send_message(
            MessageType.MODEL_DEPLOY,
            {
                "job_id": job_id,
                "model_data": model_data,
                "target_nodes": target_nodes
            }
        )
    
    # Resource Management Methods
    async def report_resources(self) -> Dict[str, Any]:
        """Report node resource status"""
        resources = await self._get_resource_metrics()
        return await self.send_message(
            MessageType.RESOURCE_STATUS,
            resources
        )
    
    async def _get_resource_metrics(self) -> Dict[str, Any]:
        """Get current resource metrics"""
        return {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "gpu_usage": 0.0,
            "disk_usage": 0.0,
            "network_bandwidth": 0.0
        }
    
    # Training Coordination Methods
    async def report_training_metrics(self, job_id: Optional[str], metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Report training metrics"""
        return await self.send_message(
            MessageType.TRAINING_METRICS,
            {
                "job_id": job_id,
                "metrics": metrics
            }
        )
    
    async def request_aggregation(self, job_id: str, weights: Dict[str, Any]) -> Dict[str, Any]:
        """Request weights aggregation"""
        return await self.send_message(
            MessageType.AGGREGATION_REQUEST,
            {
                "job_id": job_id,
                "weights": weights
            }
        )