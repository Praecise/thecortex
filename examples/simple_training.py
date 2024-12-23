# examples/simple_training.py

import asyncio
import torch
from torch.utils.data import DataLoader, TensorDataset
from cortex.core.model import CortexModel
from cortex.core.training import ModelTrainer

async def main():
    # Initialize model
    model = CortexModel(
        input_dim=784,
        hidden_dims=[512, 256],
        output_dim=10
    )
    
    # Initialize trainer
    trainer = ModelTrainer(model)
    
    # Create example dataset
    x = torch.randn(1000, 784)
    y = torch.randint(0, 10, (1000,))
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Training loop
    print("Starting training...")
    for epoch in range(5):
        epoch_loss = 0
        epoch_acc = 0
        batches = 0
        
        for data, target in dataloader:
            metrics = await trainer.train_batch(data, target)
            epoch_loss += metrics['loss']
            epoch_acc += metrics['accuracy']
            batches += 1
            
            if batches % 10 == 0:
                print(f"Epoch {epoch}, Batch {batches}: "
                      f"Loss={metrics['loss']:.4f}, "
                      f"Accuracy={metrics['accuracy']:.4f}")
        
        # Epoch summary
        avg_loss = epoch_loss / batches
        avg_acc = epoch_acc / batches
        print(f"Epoch {epoch} completed: "
              f"Avg Loss={avg_loss:.4f}, "
              f"Avg Accuracy={avg_acc:.4f}")
        
        # Save checkpoint
        trainer.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")

if __name__ == "__main__":
    asyncio.run(main())