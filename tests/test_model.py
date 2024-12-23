import pytest
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
                assert not torch.equal(current_weights[name], initial_weights[name]),                     f"Weights for {name} did not change"
