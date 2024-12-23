# examples/distributed_training.py

import asyncio
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from cortex.core.model import CortexModel
from cortex.core.training import ModelTrainer
from cortex.network.bridge import TenzroBridge
from cortex.network.protocol import TenzroProtocol
from cortex.utils.metrics import MetricsTracker

async def main():
    # Initialize node
    node_id = "example_node_1"
    websocket_url = "ws://localhost:8765"  # Your Tenzro node endpoint
    
    # Create protocol and bridge
    protocol = TenzroProtocol(node_id)
    bridge = TenzroBridge(websocket_url, node_id, protocol)
    
    # Initialize model
    model = CortexModel(
        input_dim=784,  # Example for MNIST
        hidden_dims=[512, 256],
        output_dim=10
    )
    
    # Initialize trainer and metrics
    trainer = ModelTrainer(model)
    metrics_tracker = MetricsTracker()
    
    # Create example dataset
    # In practice, this would be your real training data
    x = torch.randn(1000, 784)
    y = torch.randint(0, 10, (1000,))
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Setup message handlers
    async def handle_weight_update(message):
        weights = await protocol.receive_weights(message)
        model.load_weights(weights)
        print(f"Received weight update from {message['node_id']}")
    
    protocol.register_handler("weights", handle_weight_update)
    
    # Connect to network
    await bridge.connect()
    
    # Start training
    print("Starting distributed training...")
    try:
        for epoch in range(10):
            epoch_metrics = []
            for batch_idx, (data, target) in enumerate(dataloader):
                # Train batch
                metrics = await trainer.train_batch(data, target, protocol)
                metrics_tracker.update(metrics)
                epoch_metrics.append(metrics)
                
                # Every N batches, share weights
                if batch_idx % 10 == 0:
                    weights = model.get_weights()
                    await protocol.send_weights(weights)
                
                # Print progress
                if batch_idx % 5 == 0:
                    loss_avg = metrics_tracker.get_moving_average('loss')
                    acc_avg = metrics_tracker.get_moving_average('accuracy')
                    print(f"Epoch {epoch}, Batch {batch_idx}: "
                          f"Loss={loss_avg:.4f}, Accuracy={acc_avg:.4f}")
            
            # End of epoch
            avg_metrics = {
                key: np.mean([m[key] for m in epoch_metrics])
                for key in epoch_metrics[0].keys()
            }
            print(f"Epoch {epoch} completed: {avg_metrics}")
            
            # Check for early stopping
            if metrics_tracker.should_stop_early('loss'):
                print("Early stopping triggered")
                break
    
    except KeyboardInterrupt:
        print("Training interrupted")
    
    finally:
        # Cleanup
        await bridge.disconnect()

if __name__ == "__main__":
    asyncio.run(main())