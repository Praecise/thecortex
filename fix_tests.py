# fix_tests_v7.py

def fix_model_test():
    content = '''import pytest
import torch
import torch.nn as nn
from cortex.core.model import CortexModel

class TestCortexModel:
    @pytest.fixture
    def model_config(self):
        return {
            "input_dim": 784,
            "hidden_dims": [512, 256],
            "output_dim": 10,
            "dropout": 0.2
        }
    
    @pytest.fixture
    def model(self, model_config):
        return CortexModel(**model_config)

    def test_model_initialization(self, model, model_config):
        """Test if model initializes with correct architecture"""
        assert isinstance(model, nn.Module)
        assert model.input_dim == model_config["input_dim"]
        assert model.hidden_dims == model_config["hidden_dims"]
        assert model.output_dim == model_config["output_dim"]
    
    def test_forward_pass(self, model):
        """Test if forward pass works with correct shapes"""
        batch_size = 32
        x = torch.randn(batch_size, 784)
        output = model(x)
        
        assert output.shape == (batch_size, 10)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    @pytest.mark.asyncio
    async def test_weight_loading(self, model):
        """Test weight loading functionality"""
        with torch.no_grad():
            initial_weights = model.get_weights()
            modified_weights = {}
            
            for name, tensor in initial_weights.items():
                if tensor.dtype == torch.long:
                    modified_weights[name] = torch.randint_like(tensor, low=0, high=100)
                else:
                    modified_weights[name] = (torch.randn_like(tensor) - 0.5) * 20.0
            
            # Load modified weights
            model.load_weights(modified_weights)
            current_weights = model.get_weights()
            
            # Verify changes
            for name, tensor in initial_weights.items():
                assert not torch.equal(current_weights[name], initial_weights[name]), \
                    f"Weights for {name} did not change"
'''
    with open('tests/test_model.py', 'w') as f:
        f.write(content)

def fix_training_test():
    content = '''import pytest
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
        torch.manual_seed(42)  # For reproducibility
        batch = {
            "data": torch.randn(32, 784),
            "labels": torch.randint(0, 10, (32,))
        }
        
        losses = []
        lrs = [0.1, 0.01, 0.001]
        
        # Reset model between runs
        for lr in lrs:
            model_copy = CortexModel(784, [512, 256], 10)
            model_copy.load_state_dict(model.state_dict())
            trainer = ModelTrainer(model_copy, learning_rate=lr)
            metrics = await trainer.train_batch(
                batch["data"],
                batch["labels"]
            )
            losses.append(metrics["loss"])
        
        # Print losses for debugging
        print(f"\\nLosses for learning rates {lrs}:\\n{losses}")
        print(f"Loss changes: {[abs(losses[i] - losses[i+1]) for i in range(len(losses)-1)]}")

        # Verify that higher learning rates lead to bigger changes
        loss_changes = [abs(losses[i] - losses[i+1]) 
                       for i in range(len(losses)-1)]
        for i in range(len(loss_changes)-1):
            assert loss_changes[i] > loss_changes[i+1], \
                f"Loss change at step {i} ({loss_changes[i]:.4f}) should be > than at step {i+1} ({loss_changes[i+1]:.4f})"
'''
    with open('tests/test_training.py', 'w') as f:
        f.write(content)

if __name__ == '__main__':
    fix_model_test()
    fix_training_test()