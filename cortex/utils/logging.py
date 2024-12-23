import logging
import json
from datetime import datetime
from typing import Any, Dict, Optional
from prometheus_client import Counter, Gauge, Histogram

class MetricsCollector:
    """Collects and exposes metrics for monitoring"""
    
    def __init__(self):
        # Training metrics
        self.training_iterations = Counter(
            'cortex_training_iterations_total',
            'Total number of training iterations'
        )
        self.training_loss = Gauge(
            'cortex_training_loss',
            'Current training loss'
        )
        self.model_accuracy = Gauge(
            'cortex_model_accuracy',
            'Current model accuracy'
        )
        
        # Network metrics
        self.weight_sync_time = Histogram(
            'cortex_weight_sync_seconds',
            'Time spent synchronizing weights',
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )
        self.network_errors = Counter(
            'cortex_network_errors_total',
            'Total number of network errors'
        )
        
        # System metrics
        self.memory_usage = Gauge(
            'cortex_memory_usage_bytes',
            'Current memory usage in bytes'
        )
        self.gpu_utilization = Gauge(
            'cortex_gpu_utilization_percent',
            'Current GPU utilization percentage'
        )

class LogManager:
    """Manages structured logging for the application"""
    
    def __init__(self, log_file: Optional[str] = None, level: int = logging.INFO):
        self.logger = logging.getLogger('cortex')
        self.logger.setLevel(level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(self._get_formatter())
        self.logger.addHandler(console_handler)
        
        # File handler if log file specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(self._get_formatter())
            self.logger.addHandler(file_handler)
    
    @staticmethod
    def _get_formatter() -> logging.Formatter:
        """Get log formatter"""
        return logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _format_log_data(self, data: Dict[str, Any]) -> str:
        """Format log data as JSON string"""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            **data
        }
        return json.dumps(log_data)
    
    def training_event(self, event_type: str, metrics: Dict[str, Any]):
        """Log training related events"""
        log_data = {
            'event': 'training',
            'type': event_type,
            'metrics': metrics
        }
        self.logger.info(self._format_log_data(log_data))
    
    def network_event(self, event_type: str, node_id: str, status: str):
        """Log network related events"""
        log_data = {
            'event': 'network',
            'type': event_type,
            'node_id': node_id,
            'status': status
        }
        self.logger.info(self._format_log_data(log_data))
    
    def error(self, error_type: str, error_message: str, extra: Optional[Dict[str, Any]] = None):
        """Log error events"""
        log_data = {
            'event': 'error',
            'type': error_type,
            'message': error_message,
            **(extra or {})
        }
        self.logger.error(self._format_log_data(log_data))

# Global instances
metrics = MetricsCollector()
log_manager = LogManager()