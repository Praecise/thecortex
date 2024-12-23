# List of global nodes to use
global_nodes = [
    "ws://global-1.tenzro.org:8080",
    "ws://global-2.tenzro.org:8080"
]

# Submit training job
async def submit_training():
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8000/api/v1/jobs/submit",
            json={
                "model_config": model_config,
                "training_config": training_config,
                "global_node_urls": global_nodes
            }
        ) as response:
            result = await response.json()
            return result["job_id"]
```

2. Monitor Training Progress:
```python
async def monitor_training(job_id):
    async with aiohttp.ClientSession() as session:
        while True:
            async with session.get(
                f"http://localhost:8000/api/v1/jobs/{job_id}/status"
            ) as response:
                status = await response.json()
                print(f"Status: {status['status']}")
                if status.get('metrics'):
                    print("Metrics:", status['metrics'])
                
                if status['status'] in ['completed', 'failed']:
                    break
                    
            await asyncio.sleep(5)  # Poll every 5 seconds

# Get final results
async def get_results(job_id):
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"http://localhost:8000/api/v1/jobs/{job_id}/results"
        ) as response:
            return await response.json()
```

3. Complete Training Flow:
```python
async def run_training():
    try:
        # Submit job
        job_id = await submit_training()
        print(f"Submitted job: {job_id}")
        
        # Monitor progress
        await monitor_training(job_id)
        
        # Get results
        results = await get_results(job_id)
        print("Training completed!")
        print("Final Results:", json.dumps(results, indent=2))
        
    except Exception as e:
        print(f"Error during training: {str(e)}")

# Run the training
asyncio.run(run_training())
```

## Network Monitoring

You can monitor the network status and resource availability:

```python
async def monitor_network():
    async with aiohttp.ClientSession() as session:
        # Get network status
        async with session.get(
            "http://localhost:8000/api/v1/network/status"
        ) as response:
            status = await response.json()
            print("\nNetwork Status:")
            print(f"Total Nodes: {status['total_nodes']}")
            print("Nodes by Region:")
            for region, nodes in status['regions'].items():
                print(f"  {region}: {len(nodes)} nodes")
        
        # Get resource availability
        async with session.get(
            "http://localhost:8000/api/v1/network/resources"
        ) as response:
            resources = await response.json()
            print("\nResource Availability:")
            print(f"Available Nodes: {resources['available_nodes']}")
            print(f"Total CPU Cores: {resources['total_cpu_cores']}")
            print(f"GPU Nodes: {resources['gpu_nodes']}")
```

## Advanced Usage

1. Fine-tuning a Trained Model:
```python
async def finetune_model(job_id):
    finetune_config = {
        "learning_rate": 0.0001,
        "epochs": 5,
        "freeze_layers": ["layer1", "layer2"]
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"http://localhost:8000/api/v1/models/{job_id}/finetune",
            json=finetune_config
        ) as response:
            result = await response.json()
            return result["finetune_job_id"]
```

2. Custom Training Configuration:
```python
# Example with custom training configuration
advanced_config = {
    "model_config": {
        "input_dim": 784,
        "hidden_dims": [1024, 512, 256],
        "output_dim": 10,
        "model_type": "feedforward",
        "activation": "relu",
        "dropout_rate": 0.3
    },
    "training_config": {
        "batch_size": 64,
        "learning_rate": 0.001,
        "epochs": 20,
        "optimizer": {
            "name": "adam",
            "betas": [0.9, 0.999],
            "weight_decay": 1e-5
        },
        "scheduler": {
            "name": "cosine",
            "T_max": 10,
            "eta_min": 1e-6
        },
        "early_stopping": {
            "patience": 5,
            "min_delta": 1e-4
        }
    }
}
```

## Best Practices

1. Network Configuration:
   - Use multiple global nodes for redundancy
   - Monitor network status regularly
   - Check resource availability before submitting large jobs

2. Training Configuration:
   - Start with smaller models/datasets to test network
   - Use early stopping to prevent wasted resources
   - Monitor training metrics to detect issues early

3. Error Handling:
   - Always implement proper error handling
   - Check job status regularly
   - Save job IDs for reference

4. Resource Management:
   - Monitor resource usage across nodes
   - Avoid overloading single regions
   - Consider peak usage times

## Troubleshooting

Common issues and solutions:

1. Connection Issues:
```python
# Check network health
async def check_network_health():
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "http://localhost:8000/health"
            ) as response:
                return await response.json()
    except Exception as e:
        print(f"Network health check failed: {str(e)}")
```

2. Training Issues:
- Verify model configuration
- Check resource availability
- Monitor training metrics
- Review node logs

3. Performance Issues:
- Check network latency
- Monitor resource utilization
- Review batch size and learning rate
- Consider node distribution

## Next Steps

1. Explore advanced features:
   - Custom model architectures
   - Advanced training strategies
   - Network optimization
   - Performance tuning

2. Integration with existing systems:
   - Data pipeline integration
   - Monitoring system integration
   - Result processing pipelines

3. Scale your deployment:
   - Add more nodes
   - Optimize resource usage
   - Implement custom metrics
   - Add redundancy Getting Started with Tenzro Network

This tutorial will guide you through setting up and using the Tenzro Network for distributed model training.

## Prerequisites
- Python 3.8+
- PyTorch
- FastAPI
- aiohttp

## Network Setup

1. Start a Global Node:
```bash
# Set environment variables
export NODE_TYPE=global
export NODE_ID=global-1
export REGION=us-east

# Start the node
python -m cortex.network.node --config config/global.yaml
```

2. Start Regional Nodes:
```bash
# For each region
export NODE_TYPE=regional
export NODE_ID=regional-1
export REGION=us-east

python -m cortex.network.node --config config/regional.yaml
```

3. Start Local Node with API:
```bash
# Start API server
export NODE_TYPE=local
export NODE_ID=local-1
export REGION=us-east

uvicorn cortex.api.endpoints:app --host 0.0.0.0 --port 8000
```

## Training a Model

1. Prepare Your Model:
```python
# Define model configuration
model_config = {
    "input_dim": 784,
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

#