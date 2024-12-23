# cortex/api/endpoints.py

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import asyncio
import json
from datetime import datetime

from ..coordination.job_coordinator import JobCoordinator
from ..deployment.manager import ModelDeploymentManager
from ..network.discovery import NodeDiscoveryService
from ..network.protocol import TenzroProtocol, MessageType

class JobSubmission(BaseModel):
    neural_model_config: Dict[str, Any]  # Renamed from model_config
    training_config: Dict[str, Any]
    global_node_urls: List[str]

class JobStatus(BaseModel):
    job_id: str
    status: str
    metrics: Optional[Dict[str, Any]]
    error: Optional[str]

app = FastAPI(title="Tenzro Network API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store service instances
services: Dict[str, Any] = {}

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    services["job_coordinator"] = JobCoordinator(
        node_type="local",  # API runs on local node
        node_id="api_node"
    )
    services["deployment_manager"] = ModelDeploymentManager(
        node_type="local",
        node_id="api_node"
    )
    services["discovery"] = NodeDiscoveryService(
        node_id="api_node",
        node_type="local",
        region="default"
    )
    await services["discovery"].start()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup services on shutdown"""
    if "discovery" in services:
        await services["discovery"].stop()

# Job Management Endpoints
@app.post("/api/v1/jobs/submit", response_model=Dict[str, str])
async def submit_job(job: JobSubmission, background_tasks: BackgroundTasks):
    """Submit new training job"""
    try:
        # Initialize connection to global nodes
        protocols = []
        for url in job.global_node_urls:
            protocol = TenzroProtocol(
                node_id="api_node",
                node_type="local",
                region="default"
            )
            protocols.append(protocol)
            
        # Submit job to all global nodes
        job_ids = []
        for protocol in protocols:
            response = await protocol.submit_job({
                "model_config": job.neural_model_config,
                "training_config": job.training_config
            })
            job_ids.append(response.get("data", {}).get("job_id"))
            
        if not job_ids:
            raise HTTPException(status_code=500, detail="Failed to submit job to any global node")
            
        # Return first job ID (they should all be the same)
        return {"job_id": job_ids[0]}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/jobs/{job_id}/status", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get status of training job"""
    try:
        coordinator = services["job_coordinator"]
        job = coordinator.jobs.get(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
            
        return JobStatus(
            job_id=job_id,
            status=job.status.value,
            metrics=job.metrics,
            error=None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/jobs/{job_id}/results")
async def get_job_results(job_id: str):
    """Get results of completed job"""
    try:
        coordinator = services["job_coordinator"]
        job = coordinator.jobs.get(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
            
        if not job.results:
            raise HTTPException(status_code=400, detail="Results not yet available")
            
        return job.results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Model Management Endpoints
@app.post("/api/v1/models/{job_id}/finetune")
async def finetune_model(job_id: str, config: Dict[str, Any]):
    """Request model fine-tuning"""
    try:
        coordinator = services["job_coordinator"]
        job = coordinator.jobs.get(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
            
        # Submit fine-tuning job
        finetune_job_id = await coordinator.submit_job(
            model_config=config,
            initiator_id="api_node"
        )
        
        return {"finetune_job_id": finetune_job_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Network Status Endpoints
@app.get("/api/v1/network/status")
async def get_network_status():
    """Get status of connected nodes"""
    try:
        discovery = services["discovery"]
        region_status = await discovery.get_region_status()
        
        return {
            "regions": region_status,
            "total_nodes": sum(len(nodes) for nodes in region_status.values()),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/network/resources")
async def get_network_resources():
    """Get resource availability across network"""
    try:
        discovery = services["discovery"]
        nodes = discovery.nodes
        
        resources = {
            "available_nodes": len([n for n in nodes.values() if n.available]),
            "total_cpu_cores": sum(n.capacity.cpu_cores for n in nodes.values()),
            "total_memory": sum(n.capacity.memory_total for n in nodes.values()),
            "gpu_nodes": len([n for n in nodes.values() if n.capacity.gpu_memory]),
            "regions": list(discovery.active_regions)
        }
        
        return resources
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health Check Endpoint
@app.get("/health")
async def health_check():
    """API health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }