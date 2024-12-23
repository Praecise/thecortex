import asyncio
import torch
from cortex.core.model import CortexModel
from cortex.core.training import ModelTrainer

async def test_core():
    # Test model initialization
    model = CortexModel(
        input_dim=784,
        hidden_dims=[512, 256],
        output_dim=10
    )
    
    # Create sample data
    x = torch.randn(32, 784)
    y = torch.randint(0, 10, (32,))
    
    # Test forward pass
    output = model(x)
    print(f"Model output shape: {output.shape}")
    
    # Test trainer
    trainer = ModelTrainer(model)
    metrics = await trainer.train_batch(x, y)
    print(f"Training metrics: {metrics}")

if __name__ == "__main__":
    asyncio.run(test_core())