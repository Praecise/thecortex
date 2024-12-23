from typing import List, Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field, validator
import os
import yaml
from pathlib import Path

class NetworkConfig(BaseSettings):
    """Network-related configuration"""
    websocket_url: str
    node_id: str
    retry_attempts: int = 3
    connection_timeout: int = 30
    batch_size: int = 10
    max_wait: float = 1.0
    
    @validator('websocket_url')
    def validate_url(cls, v: str) -> str:
        if not v.startswith(('ws://', 'wss://')):
            raise ValueError('Invalid websocket URL')
        return v

class TrainingConfig(BaseSettings):
    """Training-related configuration"""
    input_dim: int = 784
    output_dim: int = 10  # Added output_dim
    batch_size: int = 32
    learning_rate: float = 0.001
    hidden_dims: List[int] = [512, 256]
    dropout_rate: float = 0.2
    max_epochs: int = 100
    early_stopping_patience: int = 5
    validation_interval: int = 10
    checkpoint_interval: int = 100
    checkpoint_dir: str = "./checkpoints"
    
    class Config:
        extra = 'allow'  # Allow extra fields
    
    @validator('learning_rate')
    def validate_learning_rate(cls, v: float) -> float:
        if v <= 0 or v >= 1:
            raise ValueError('Learning rate must be between 0 and 1')
        return v

class SecurityConfig(BaseSettings):
    """Security-related configuration"""
    encryption_key: Optional[str] = None
    enable_encryption: bool = True
    key_rotation_interval: int = 24  # hours
    
    class Config:
        extra = 'allow'

class MonitoringConfig(BaseSettings):
    """Monitoring-related configuration"""
    log_level: str = "INFO"
    metrics_port: int = 9090
    enable_prometheus: bool = True
    log_file: Optional[str] = None
    
    class Config:
        extra = 'allow'

class CortexConfig:
    """Main configuration class"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.network: NetworkConfig
        self.training: TrainingConfig
        self.security: SecurityConfig
        self.monitoring: MonitoringConfig
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file or environment"""
        config_data: Dict[str, Any] = {}
        
        # Load from file if provided
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
        
        # Initialize config objects with data or environment variables
        self.network = NetworkConfig(**config_data.get('network', {}))
        self.training = TrainingConfig(**config_data.get('training', {}))
        self.security = SecurityConfig(**config_data.get('security', {}))
        self.monitoring = MonitoringConfig(**config_data.get('monitoring', {}))