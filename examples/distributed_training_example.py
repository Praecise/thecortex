# examples/distributed_training_example.py

import asyncio
import torch
import torch.nn as nn
import requests
import json
from datetime import datetime
import aiohttp
import time

async def train_model_example():
    """Example of submitting and monitoring a training job"""
    
    # Define model configuration
    model_config = {
        "input_dim": 784,  # MNIST example
        "hidden_dims": [512, 256],
        "output_dim": 10,
        "model_type": "feedforward"
    }
    
    # Define training configuration
    training_config = {
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 10,
        "optimizer": "adam",
        "loss_function": "cross_entropy"
    }
    
    # Global node URLs (in a real setup, these would be actual URLs)
    global_nodes = [
        "ws://global-node-1.tenzro.org:8080",
        "ws://global-node-2.tenzro.org:8080"
    ]
    
    # API endpoint (local node)
    api_url = "http://localhost:8000"
    
    try:
        # Submit training job
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{api_url}/api/v1/jobs/submit",
                json={
                    "model_config": model_config,
                    "training_config": training_config,
                    "global_node_urls": global_nodes
                }
            ) as response:
                result = await response.json()
                job_id = result["job_id"]
                print(f"Submitted job with ID: {job_id}")
                
            # Monitor training progress
            while True:
                async with session.get(
                    f"{api_url}/api/v1/jobs/{job_id}/status"
                ) as response:
                    status = await response.json()
                    
                    print(f"\nJob Status: {status['status']}")
                    if status.get('metrics'):
                        print("Current Metrics:")
                        for key, value in status['metrics'].items():
                            print(f"  {key}: {value}")
                            
                    if status['status'] in ['completed', 'failed']:
                        break
                        
                await asyncio.sleep(5)  # Poll every 5 seconds
                
            # If training completed successfully, get results
            if status['status'] == 'completed':
                async with session.get(
                    f"{api_url}/api/v1/jobs/{job_id}/results"
                ) as response:
                    results = await response.json()
                    print("\nTraining Results:")
                    print(json.dumps(results, indent=2))
                    
                # Optional: Request fine-tuning
                finetune_config = {
                    "learning_rate": 0.0001,
                    "epochs": 5,
                    "freeze_layers": ["layer1", "layer2"]
                }
                
                async with session.post(
                    f"{api_url}/api/v1/models/{job_id}/finetune",
                    json=finetune_config
                ) as response:
                    finetune_result = await response.json()
                    print(f"\nSubmitted fine-tuning job: {finetune_result['finetune_job_id']}")
                    
    except Exception as e:
        print(f"Error during training: {str(e)}")

# Network monitoring example
async def monitor_network_example():
    """Example of monitoring network status"""
    
    api_url = "http://localhost:8000"
    
    try:
        async with aiohttp.ClientSession() as session:
            # Get network status
            async with session.get(
                f"{api_url}/api/v1/network/status"
            ) as response:
                status = await response.json()
                print("\nNetwork Status:")
                print(f"Total Nodes: {status['total_nodes']}")
                print("\nNodes by Region:")
                for region, nodes in status['regions'].items():
                    print(f"  {region}: {len(nodes)} nodes")
                    
            # Get resource availability
            async with session.get(
                f"{api_url}/api/v1/network/resources"
            ) as response:
                resources = await response.json()
                print("\nResource Availability:")
                print(f"Available Nodes: {resources['available_nodes']}")
                print(f"Total CPU Cores: {resources['total_cpu_cores']}")
                print(f"Total Memory: {resources['total_memory'] / (1024**3):.2f} GB")
                print(f"GPU Nodes: {resources['gpu_nodes']}")
                print(f"Active Regions: {', '.join(resources['regions'])}")
                
    except Exception as e:
        print(f"Error monitoring network: {str(e)}")

if __name__ == "__main__":
    # Run examples
    asyncio.run(train_model_example())
    print("\n" + "="*50 + "\n")
    asyncio.run(monitor_network_example())