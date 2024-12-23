# cortex/integration/manager.py

import asyncio
from typing import Dict, Any, Optional, Set, List
import torch
from datetime import datetime
import os

from ..core.model import CortexModel
from ..core.training import ModelTrainer
from ..network.bridge import TenzroBridge
from ..network.protocol import TenzroProtocol, MessageType
from ..security.encryption import EncryptionManager
from ..optimization.performance import WeightCompressor, BatchProcessor, CacheManager
from ..utils.error_handling import RecoveryManager, retry_with_backoff
from ..utils.logging import log_manager
from ..utils.metrics import MetricsTracker
from ..config.settings import CortexConfig
from ..coordination.job_coordinator import JobCoordinator, JobStatus
from ..deployment.manager import ModelDeploymentManager
from ..network.discovery import NodeDiscoveryService

class CortexManager:
    """Central integration manager for Cortex components"""
    
    def __init__(self, config_path: str):
        # Load configuration
        self.config = CortexConfig(config_path)
        
        # Initialize components
        self.model: Optional[CortexModel] = None
        self.trainer: Optional[ModelTrainer] = None
        self.bridge: Optional[TenzroBridge] = None
        self.protocol: Optional[TenzroProtocolV2] = None
        
        # Initialize coordination components
        self.job_coordinator = JobCoordinator(
            node_type=self.config.network.node_type,
            node_id=self.config.network.node_id
        )
        self.deployment_manager = ModelDeploymentManager(
            node_type=self.config.network.node_type,
            node_id=self.config.network.node_id
        )
        self.discovery_service = NodeDiscoveryService(
            node_id=self.config.network.node_id,
            node_type=self.config.network.node_type,
            region=self.config.network.region
        )
        
        # Initialize utility components
        self.encryption = EncryptionManager(self.config.security.encryption_key)
        self.compressor = WeightCompressor()
        self.batch_processor = BatchProcessor(
            batch_size=self.config.training.batch_size
        )
        self.cache = CacheManager()
        self.metrics_tracker = MetricsTracker()
        
        # Ensure checkpoint directory exists
        os.makedirs(self.config.training.checkpoint_dir, exist_ok=True)
        self.recovery = RecoveryManager(self.config.training.checkpoint_dir)
        
        # Track running tasks
        self.tasks: Set[asyncio.Task] = set()
        
    async def initialize(self) -> None:
        """Initialize all components"""
        try:
            # Create model
            self.model = CortexModel(
                input_dim=self.config.training.input_dim,
                hidden_dims=self.config.training.hidden_dims,
                output_dim=self.config.training.output_dim,
                dropout=self.config.training.dropout_rate
            )
            
            # Create trainer
            self.trainer = ModelTrainer(
                model=self.model,
                learning_rate=self.config.training.learning_rate
            )
            
            # Initialize network components if not already set
            if not self.bridge:
                self.protocol = TenzroProtocol(
                    node_id=self.config.network.node_id,
                    node_type=self.config.network.node_type,
                    region=self.config.network.region
                )
                self.bridge = TenzroBridge(
                    websocket_url=self.config.network.websocket_url,
                    node_id=self.config.network.node_id,
                    node_type=self.config.network.node_type,
                    region=self.config.network.region,
                    protocol=self.protocol
                )
            
            # Start discovery service
            await self.discovery_service.start()
            
            # Register message handlers
            self._register_handlers()
            
            # Start background tasks
            self._start_background_tasks()
            
            log_manager.training_event(
                'initialization',
                {'status': 'completed'}
            )
            
        except Exception as e:
            log_manager.error(
                'initialization_error',
                str(e)
            )
            raise
    
    def _register_handlers(self) -> None:
        """Register protocol message handlers"""
        if self.protocol:
            self.protocol.register_handler(
                MessageType.WEIGHTS_UPDATE.value,
                self._handle_weight_update
            )
            self.protocol.register_handler(
                MessageType.TRAINING_METRICS.value,
                self._handle_metrics_update
            )
            self.protocol.register_handler(
                MessageType.JOB_SUBMIT.value,
                self._handle_job_submission
            )
            self.protocol.register_handler(
                MessageType.MODEL_DEPLOY.value,
                self._handle_model_deployment
            )
            self.protocol.register_handler(
                MessageType.AGGREGATION_REQUEST.value,
                self._handle_aggregation_request
            )
    
    def _start_background_tasks(self) -> None:
        """Start background processing tasks"""
        self.tasks.add(
            asyncio.create_task(self.batch_processor.batch_worker())
        )
        self.tasks.add(
            asyncio.create_task(self._resource_reporting_loop())
        )
    
    async def _resource_reporting_loop(self):
        """Periodically report resource status"""
        while True:
            try:
                await self.protocol.report_resources()
                await asyncio.sleep(30)  # Report every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                log_manager.error('resource_reporting_error', str(e))
                await asyncio.sleep(5)
    
    async def _handle_weight_update(self, message: Dict[str, Any]) -> None:
        """Handle incoming weight updates"""
        try:
            weights = message.get('data', {}).get('weights', {})
            
            # Decrypt weights if encryption enabled
            if self.config.security.enable_encryption:
                weights = self.encryption.decrypt_weights(weights)
            
            # Decompress weights
            weights = self.compressor.decompress_weights(weights)
            
            # Add to batch processor
            await self.batch_processor.add_to_batch(weights)
            
            log_manager.network_event(
                'weight_update',
                message['sender_id'],
                'processed'
            )
            
        except Exception as e:
            log_manager.error(
                'weight_update_error',
                str(e)
            )
            raise
    
    async def _handle_metrics_update(self, message: Dict[str, Any]) -> None:
        """Handle incoming metrics updates"""
        try:
            metrics = message.get('data', {}).get('metrics', {})
            job_id = message.get('data', {}).get('job_id')
            sender_id = message.get('sender_id', 'unknown')
            
            if job_id:
                # Update job metrics
                await self.job_coordinator.update_job_metrics(job_id, metrics)
            
            # Log received metrics
            log_manager.training_event(
                'metrics_received',
                {'node_id': sender_id, 'job_id': job_id, 'metrics': metrics}
            )
            
            # Update metrics tracking
            self.metrics_tracker.update(metrics)
            
            # Check for early stopping
            if self._should_stop_early(metrics):
                await self._handle_early_stopping(job_id, sender_id, metrics)
                
        except Exception as e:
            log_manager.error(
                'metrics_update_error',
                str(e)
            )
            raise
    
    async def _handle_job_submission(self, message: Dict[str, Any]) -> None:
        """Handle new job submission"""
        try:
            job_config = message.get('data', {}).get('config', {})
            sender_id = message.get('sender_id')
            
            # Submit job to coordinator
            job_id = await self.job_coordinator.submit_job(
                job_config,
                sender_id
            )
            
            # If global node, handle deployment
            if self.config.network.node_type == "global":
                await self._handle_global_deployment(job_id, job_config)
            
            # Send response
            await self.protocol.send_message(
                MessageType.JOB_STATUS,
                {
                    "job_id": job_id,
                    "status": JobStatus.SUBMITTED.value
                },
                target_id=sender_id
            )
            
        except Exception as e:
            log_manager.error(
                'job_submission_error',
                str(e)
            )
            raise
    
    async def _handle_model_deployment(self, message: Dict[str, Any]) -> None:
        """Handle model deployment request"""
        try:
            job_id = message.get('data', {}).get('job_id')
            model_data = message.get('data', {}).get('model_data')
            target_nodes = message.get('data', {}).get('target_nodes', [])
            
            # Deploy model
            success = await self.deployment_manager.deploy_model(
                job_id,
                model_data,
                target_nodes
            )
            
            if success:
                await self.job_coordinator.update_job_status(
                    job_id,
                    JobStatus.TRAINING
                )
            
        except Exception as e:
            log_manager.error(
                'model_deployment_error',
                str(e)
            )
            raise
    
    async def _handle_aggregation_request(self, message: Dict[str, Any]) -> None:
        """Handle weight aggregation request"""
        try:
            job_id = message.get('data', {}).get('job_id')
            weights = message.get('data', {}).get('weights')
            
            # Process weights through batch processor
            aggregated = await self.batch_processor.process_batch()
            
            if aggregated:
                # Send aggregated weights
                await self.protocol.send_message(
                    MessageType.AGGREGATION_RESPONSE,
                    {
                        "job_id": job_id,
                        "weights": aggregated
                    }
                )
            
        except Exception as e:
            log_manager.error(
                'aggregation_error',
                str(e)
            )
            raise
    
    def _should_stop_early(self, metrics: Dict[str, Any]) -> bool:
        """Check if training should stop early"""
        if not hasattr(self.config.training, 'early_stopping_patience'):
            return False
            
        return (
            'loss' in metrics and
            self.metrics_tracker.should_stop_early(
                'loss',
                patience=self.config.training.early_stopping_patience
            )
        )
    
    async def _handle_early_stopping(
        self,
        job_id: str,
        node_id: str,
        metrics: Dict[str, Any]
    ) -> None:
        """Handle early stopping"""
        log_manager.training_event(
            'early_stopping',
            {
                'job_id': job_id,
                'node_id': node_id,
                'final_metrics': metrics
            }
        )
        
        if job_id:
            await self.job_coordinator.update_job_status(
                job_id,
                JobStatus.COMPLETED
            )
            
        await self.cleanup()
    
    async def _handle_global_deployment(
        self,
        job_id: str,
        job_config: Dict[str, Any]
    ) -> None:
        """Handle deployment for global node"""
        try:
            # Get available regions
            region_status = await self.discovery_service.get_region_status()
            
            if not region_status:
                raise Exception("No regions available for deployment")
            
            # Deploy to each region
            for region, nodes in region_status.items():
                if nodes:
                    await self.deployment_manager.deploy_model(
                        job_id,
                        job_config,
                        nodes
                    )
            
        except Exception as e:
            log_manager.error(
                'global_deployment_error',
                str(e)
            )
            raise
    
    @retry_with_backoff(max_retries=3)
    async def start_training(self, data_loader: torch.utils.data.DataLoader) -> None:
        """Start model training"""
        try:
            # Connect to network if not connected
            if not (self.bridge and self.bridge.connected):
                await self.bridge.connect()
            
            # Training loop
            for epoch in range(self.config.training.max_epochs):
                for batch_idx, (data, target) in enumerate(data_loader):
                    # Train batch
                    metrics = await self.trainer.train_batch(
                        data,
                        target,
                        self.protocol
                    )
                    
                    # Update metrics tracker
                    self.metrics_tracker.update(metrics)
                    
                    # Get and process weights
                    weights = self.model.get_weights()
                    
                    # Compress weights
                    compressed = self.compressor.compress_weights(weights)
                    
                    # Encrypt if enabled
                    if self.config.security.enable_encryption:
                        compressed = self.encryption.encrypt_weights(compressed)
                    
                    # Send weights update
                    await self.protocol.send_message(
                        MessageType.WEIGHTS_UPDATE,
                        {"weights": compressed}
                    )
                    
                    # Save checkpoint if needed
                    if batch_idx % self.config.training.checkpoint_interval == 0:
                        await self.save_checkpoint(epoch * len(data_loader) + batch_idx)
                    
                    log_manager.training_event(
                        'batch_completed',
                        {
                            'epoch': epoch,
                            'batch': batch_idx,
                            'metrics': metrics
                        }
                    )
                    
                    # Check for early stopping
                    if self._should_stop_early(metrics):
                        log_manager.training_event(
                            'early_stopping',
                            {'epoch': epoch, 'batch': batch_idx, 'metrics': metrics}
                        )
                        return
                    
        except Exception as e:
            log_manager.error(
                'training_error',
                str(e)
            )
            raise
        finally:
            await self.cleanup()
    
    async def save_checkpoint(self, iteration: int) -> None:
        """Save training checkpoint"""
        await self.recovery.save_checkpoint(
            model_state=self.model.state_dict(),
            optimizer_state=self.trainer.optimizer.state_dict(),
            training_state={
                'iteration': iteration,
                'metrics': self.metrics_tracker.metrics
            },
            iteration=iteration
        )
    
    async def load_checkpoint(self, iteration: Optional[int] = None) -> None:
        """Load training checkpoint
        
        Args:
            iteration: Optional specific iteration to load. If None, loads latest.
        """
        try:
            checkpoint = await self.recovery.load_latest_checkpoint()
            if checkpoint:
                # Load model state
                if self.model:
                    self.model.load_state_dict(checkpoint['model_state'])
                else:
                    log_manager.error('checkpoint_load_error', 'Model not initialized')
                    return

                # Load optimizer state
                if self.trainer:
                    self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state'])
                else:
                    log_manager.error('checkpoint_load_error', 'Trainer not initialized')
                    return

                # Load metrics
                if 'metrics' in checkpoint['training_state']:
                    for metric_name, metric_values in checkpoint['training_state']['metrics'].items():
                        self.metrics_tracker.metrics[metric_name] = metric_values

                log_manager.training_event(
                    'checkpoint_loaded',
                    {
                        'iteration': checkpoint['training_state'].get('iteration'),
                        'timestamp': datetime.utcnow().isoformat()
                    }
                )
            else:
                log_manager.training_event(
                    'checkpoint_load_skipped',
                    {'reason': 'No checkpoint found'}
                )

        except Exception as e:
            log_manager.error(
                'checkpoint_load_error',
                str(e)
            )
            raise