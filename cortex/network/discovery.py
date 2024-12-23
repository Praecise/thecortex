# cortex/network/discovery.py

import asyncio
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
import psutil
import json
from dataclasses import dataclass

@dataclass
class NodeCapacity:
    cpu_cores: int
    memory_total: int
    gpu_memory: Optional[int]
    disk_space: int
    network_bandwidth: int

@dataclass
class NodeStatus:
    node_id: str
    node_type: str
    region: str
    capacity: NodeCapacity
    current_load: Dict[str, float]
    available: bool
    last_updated: datetime

class NodeDiscoveryService:
    """Manages node discovery and resource monitoring"""
    
    def __init__(self, node_id: str, node_type: str, region: str):
        self.node_id = node_id
        self.node_type = node_type
        self.region = region
        self.nodes: Dict[str, NodeStatus] = {}
        self.active_regions: Set[str] = set()
        self._heartbeat_interval = 30  # seconds
        self._cleanup_interval = 120  # seconds
        self._tasks = set()
        
    async def start(self):
        """Start discovery service"""
        self._tasks.add(asyncio.create_task(self._heartbeat_loop()))
        self._tasks.add(asyncio.create_task(self._cleanup_loop()))
        
    async def stop(self):
        """Stop discovery service"""
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        
    async def _heartbeat_loop(self):
        """Send periodic heartbeats"""
        while True:
            try:
                status = await self._get_node_status()
                await self._broadcast_status(status)
                await asyncio.sleep(self._heartbeat_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Heartbeat error: {str(e)}")
                await asyncio.sleep(5)
                
    async def _cleanup_loop(self):
        """Clean up inactive nodes"""
        while True:
            try:
                await self._remove_inactive_nodes()
                await asyncio.sleep(self._cleanup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Cleanup error: {str(e)}")
                await asyncio.sleep(5)
                
    async def _get_node_status(self) -> Dict[str, Any]:
        """Get current node status"""
        try:
            # Get CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Get network metrics
            net_io = psutil.net_io_counters()
            
            status = {
                "node_id": self.node_id,
                "node_type": self.node_type,
                "region": self.region,
                "capacity": {
                    "cpu_cores": psutil.cpu_count(),
                    "memory_total": memory.total,
                    "disk_space": disk.total,
                    "network_bandwidth": net_io.bytes_sent + net_io.bytes_recv
                },
                "current_load": {
                    "cpu_usage": cpu_percent,
                    "memory_usage": memory.percent,
                    "disk_usage": disk.percent,
                    "network_usage": (net_io.bytes_sent + net_io.bytes_recv) / 1024 / 1024  # MB
                },
                "available": True,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return status
            
        except Exception as e:
            print(f"Error getting node status: {str(e)}")
            return None
            
    async def update_node_status(self, status_data: Dict[str, Any]):
        """Update status of a node"""
        node_id = status_data["node_id"]
        region = status_data["region"]
        
        self.active_regions.add(region)
        
        self.nodes[node_id] = NodeStatus(
            node_id=node_id,
            node_type=status_data["node_type"],
            region=region,
            capacity=NodeCapacity(
                cpu_cores=status_data["capacity"]["cpu_cores"],
                memory_total=status_data["capacity"]["memory_total"],
                gpu_memory=status_data["capacity"].get("gpu_memory"),
                disk_space=status_data["capacity"]["disk_space"],
                network_bandwidth=status_data["capacity"]["network_bandwidth"]
            ),
            current_load=status_data["current_load"],
            available=status_data["available"],
            last_updated=datetime.utcnow()
        )
        
    async def get_available_nodes(self, 
                                node_type: Optional[str] = None,
                                region: Optional[str] = None,
                                min_cpu: Optional[float] = None,
                                min_memory: Optional[float] = None) -> List[str]:
        """Get available nodes matching criteria"""
        available_nodes = []
        
        for node_id, status in self.nodes.items():
            if not status.available:
                continue
                
            if node_type and status.node_type != node_type:
                continue
                
            if region and status.region != region:
                continue
                
            if min_cpu and status.current_load["cpu_usage"] > (100 - min_cpu):
                continue
                
            if min_memory and status.current_load["memory_usage"] > (100 - min_memory):
                continue
                
            available_nodes.append(node_id)
            
        return available_nodes
        
    async def get_region_status(self) -> Dict[str, List[str]]:
        """Get status of all regions"""
        region_nodes = {}
        for region in self.active_regions:
            nodes = await self.get_available_nodes(region=region)
            region_nodes[region] = nodes
        return region_nodes
        
    async def _remove_inactive_nodes(self):
        """Remove nodes that haven't sent heartbeat"""
        current_time = datetime.utcnow()
        inactive_threshold = timedelta(seconds=self._cleanup_interval * 2)
        
        inactive_nodes = [
            node_id for node_id, status in self.nodes.items()
            if (current_time - status.last_updated) > inactive_threshold
        ]
        
        for node_id in inactive_nodes:
            del self.nodes[node_id]
            
        # Update active regions
        self.active_regions = {status.region for status in self.nodes.values()}