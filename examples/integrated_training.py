import asyncio
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml
import os
from cortex.integration.manager import CortexManager

async def main():
    # Create example configuration
    config = {
        'network': {
            'websocket_url': 'ws://localhost:8765',
            'node_id': 'example_node_1',
            'retry_attempts': 3,
            'connection_timeout': 30,
            'batch_size': 10,
            'max_wait': 1.0
        },
        'training': {
            'input_dim': 784,
            'hidden_dims': [512, 256],
            'output_dim': 10,
            'batch_size': 32,
            'learning_rate': 0.001,
            'dropout_rate': 0.2,
            'max_epochs': 10,
            'checkpoint_interval': 100,
            'checkpoint_dir': './checkpoints'
        },
        'security': {
            'enable_encryption': True,
            'encryption_key': None  # Will be auto-generated
        },
        'monitoring': {
            'log_level': 'INFO',
            'log_file': 'training.log'
        }
    }
    
    # Save configuration
    os.makedirs('config', exist_ok=True)
    config_path = 'config/cortex_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Create example dataset
    x = torch.randn(1000, 784)
    y = torch.randint(0, 10, (1000,))
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    try:
        # Initialize Cortex manager
        manager = CortexManager(config_path)
        await manager.initialize()
        
        # Load checkpoint if exists
        state = await manager.load_checkpoint()
        if state:
            print(f"Resumed from iteration {state['iteration']}")
        
        # Start training
        print("Starting distributed training...")
        await manager.start_training(dataloader)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error during training: {str(e)}")
    finally:
        # Cleanup will be handled automatically
        pass

if __name__ == "__main__":
    asyncio.run(main())