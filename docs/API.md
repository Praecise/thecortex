# Cortex API Documentation

## Overview
The Cortex API provides endpoints for managing distributed training jobs, model deployment, and network resources.

## Base URL
```
Production: https://api.thecortex.xyz/v1
Staging: https://api.staging.thecortex.xyz/v1
```

## Authentication
All requests must include an API key in the header:
```bash
Authorization: Bearer your-api-key
```

## Endpoints

### Job Management

#### Submit Training Job
```http
POST /jobs/submit
```

Request Body:
```json
{
  "neural_model_config": {
    "input_dim": 784,
    "hidden_dims": [512, 256],
    "output_dim": 10,
    "dropout_rate": 0.2
  },
  "training_config": {
    "batch_size": 32,
    "learning_rate": 0.001,
    "max_epochs": 100,
    "early_stopping_patience": 5
  },
  "global_node_urls": [
    "ws://node1.thecortex.xyz",
    "ws://node2.thecortex.xyz"
  ]
}
```

Response:
```json
{
  "job_id": "job_123456",
  "status": "submitted",
  "timestamp": "2024-01-23T12:34:56Z"
}
```

#### Get Job Status
```http
GET /jobs/{job_id}/status
```

Response:
```json
{
  "job_id": "job_123456",
  "status": "training",
  "metrics": {
    "loss": 0.234,
    "accuracy": 0.945,
    "epoch": 23
  },
  "error": null,
  "last_updated": "2024-01-23T12:34:56Z"
}
```

#### Get Job Results
```http
GET /jobs/{job_id}/results
```

Response:
```json
{
  "job_id": "job_123456",
  "training_duration": 3600,
  "final_metrics": {
    "loss": 0.123,
    "accuracy": 0.975,
    "val_loss": 0.145,
    "val_accuracy": 0.962
  },
  "model_artifacts": {
    "weights_url": "s3://cortex-models/job_123456/weights.pt",
    "config_url": "s3://cortex-models/job_123456/config.json"
  }
}
```

### Model Management

#### Deploy Model
```http
POST /models/{job_id}/deploy
```

Request Body:
```json
{
  "deployment_config": {
    "min_nodes": 1,
    "max_nodes": 5,
    "gpu_required": true,
    "memory_request": "16Gi"
  },
  "serving_config": {
    "batch_size": 32,
    "timeout_ms": 100,
    "max_concurrent_requests": 100
  }
}
```

Response:
```json
{
  "deployment_id": "deploy_123456",
  "status": "deploying",
  "endpoints": {
    "inference": "https://inference.thecortex.xyz/models/job_123456",
    "metrics": "https://metrics.thecortex.xyz/models/job_123456"
  }
}
```

#### Fine-tune Model
```http
POST /models/{job_id}/finetune
```

Request Body:
```json
{
  "fine_tuning_config": {
    "learning_rate": 0.0001,
    "epochs": 10,
    "freeze_layers": ["layer1", "layer2"]
  },
  "dataset_config": {
    "dataset_url": "s3://cortex-data/fine-tuning/dataset1",
    "validation_split": 0.2
  }
}
```

Response:
```json
{
  "finetune_job_id": "finetune_123456",
  "status": "submitted",
  "original_job_id": "job_123456"
}
```

### Network Management

#### Get