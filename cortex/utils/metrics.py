# cortex/utils/metrics.py

from typing import Dict, List, Any
import numpy as np
from collections import defaultdict

class MetricsTracker:
    """Tracks and analyzes training metrics"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.windows = defaultdict(list)
        self.window_size = 10
        
    def update(self, metrics: Dict[str, float]) -> None:
        """Update metrics with new values"""
        for key, value in metrics.items():
            self.metrics[key].append(value)
            self.windows[key].append(value)
            if len(self.windows[key]) > self.window_size:
                self.windows[key].pop(0)
                
    def get_moving_average(self, metric: str) -> float:
        """Get moving average for metric"""
        values = self.windows[metric]
        return sum(values) / len(values) if values else 0.0
    
    def get_improvement_rate(self, metric: str) -> float:
        """Calculate improvement rate for metric"""
        values = self.metrics[metric]
        if len(values) < 2:
            return 0.0
            
        # Use linear regression to estimate improvement rate
        x = np.arange(len(values))
        y = np.array(values)
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    def should_stop_early(
        self,
        metric: str,
        patience: int = 5,
        min_improvement: float = 1e-4
    ) -> bool:
        """Check if training should stop early"""
        values = self.metrics[metric]
        if len(values) < patience:
            return False
            
        # Check if improvement is below threshold
        recent_values = values[-patience:]
        improvements = [abs(recent_values[i] - recent_values[i-1]) 
                       for i in range(1, len(recent_values))]
                       
        return max(improvements) < min_improvement