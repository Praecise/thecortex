import pytest
import torch
import torch.nn as nn
from cortex.core.model import CortexModel
from cortex.core.training import ModelTrainer
from cortex.network.protocol import TenzroProtocol

class TestModelTrainer:
    @pytest.fixture
    def model(self):
        return CortexModel(784, [512, 256], 10)
    
    @pytest.fixture
    def trainer(self, model):
        return ModelTrainer(model)
    
    @pytest.fixture
    def protocol(self):
        return TenzroProtocol("test_node_1")
    
    @pytest.fixture
    def sample_batch(self):
        batch_size = 32
        return {
            "data": torch.randn(batch_size, 784),
            "labels": torch.randint(0, 10, (batch_size,))
        }
    
    @pytest.mark.asyncio
    async def test_training_step(self, trainer, sample_batch):
        """Test single training step"""
        metrics = await trainer.train_batch(
            sample_batch["data"],
            sample_batch["labels"]
        )
        
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert metrics["loss"] > 0
    
    @pytest.mark.asyncio
    async def test_validation(self, trainer, sample_batch):
        """Test validation step"""
        metrics = await trainer.validate(
            sample_batch["data"],
            sample_batch["labels"]
        )
        
        assert "val_loss" in metrics
        assert "val_accuracy" in metrics
        assert 0 <= metrics["val_accuracy"] <= 1
    
    def test_checkpoint_saving(self, trainer, tmp_path):
        """Test checkpoint saving and loading"""
        checkpoint_path = tmp_path / "checkpoint.pt"
        
        # Save initial weights
        initial_weights = trainer.model.get_weights()
        trainer.save_checkpoint(str(checkpoint_path))
        
        # Modify weights
        for param in trainer.model.parameters():
            param.data = torch.randn_like(param.data)
        
        # Load checkpoint
        trainer.load_checkpoint(str(checkpoint_path))
        loaded_weights = trainer.model.get_weights()
        
        # Verify weights restored
        for key in initial_weights:
            assert torch.allclose(
                initial_weights[key],
                loaded_weights[key]
            )
    
    @pytest.mark.asyncio
    async def test_learning_rate_effect(self, model):
        """Test effect of different learning rates"""
        torch.manual_seed(42)
        batch = {
            "data": torch.randn(32, 784),
            "labels": torch.randint(0, 10, (32,))
        }
        
        losses = []
        lrs = [0.1, 0.01, 0.001]
        
        for lr in lrs:
            model_copy = CortexModel(784, [512, 256], 10)
            model_copy.load_state_dict(model.state_dict())
            trainer = ModelTrainer(model_copy, learning_rate=lr)
            
            # Train for multiple steps to get stable loss
            for _ in range(5):
                metrics = await trainer.train_batch(
                    batch["data"],
                    batch["labels"]
                )
            losses.append(metrics["loss"])
        
        # Verify that loss is reasonable
        assert all(0 < loss < 10 for loss in losses), "Losses should be in reasonable range"
