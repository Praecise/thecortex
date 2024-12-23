# Cortex Operational Runbook

## System Overview
Cortex is a distributed neural network training framework operating across multiple nodes and regions.

### Key Components
- API Nodes
- Training Nodes
- Global Nodes
- Monitoring Stack
- Message Queue System

## Daily Operations

### Health Checks
1. Monitor System Health
```bash
# Check node status
kubectl get nodes -o wide

# Check pod status
kubectl get pods -n cortex

# Check service status
kubectl get services -n cortex
```

2. Monitor Training Jobs
```bash
# Get job status
kubectl logs -f deployment/cortex-trainer -n cortex

# Check training metrics
curl http://api.thecortex.xyz/api/v1/metrics
```

3. Check Resource Usage
```bash
# Get node resource usage
kubectl top nodes

# Get pod resource usage
kubectl top pods -n cortex
```

### Routine Maintenance

#### Daily Tasks
- Review error logs
- Clean up completed jobs
- Verify backup completion
- Check alert status

#### Weekly Tasks
- Rotate encryption keys
- Review performance metrics
- Update model checkpoints
- Clean old data

#### Monthly Tasks
- Certificate rotation
- Security patches
- Capacity planning review
- Performance optimization

## Common Procedures

### Scaling Operations
```bash
# Scale training nodes
kubectl scale deployment cortex-trainer -n cortex --replicas=5

# Scale API nodes
kubectl scale deployment cortex-api -n cortex --replicas=3
```

### Certificate Management
```bash
# Check certificate expiration
kubectl get secret cortex-tls -n cortex -o jsonpath='{.metadata.annotations.cert-manager\.io/expiration}'

# Rotate certificates
kubectl delete secret cortex-tls -n cortex
kubectl apply -f deploy/k8s/certificate.yaml
```

### Log Management
```bash
# Export logs
kubectl logs deployment/cortex-api -n cortex > api_logs.txt

# Enable debug logging
kubectl set env deployment/cortex-api -n cortex LOG_LEVEL=DEBUG
```

### Backup Procedures
```bash
# Backup model weights
aws s3 sync s3://cortex-checkpoints-prod/models/ backup/models/

# Backup configuration
kubectl get configmap -n cortex -o yaml > config_backup.yaml
```

## Alert Response Procedures

### High Memory Usage Alert
1. Check memory consumption:
```bash
kubectl top pods -n cortex --sort-by=memory
```
2. Identify memory leaks using profiler
3. Consider scaling up or restarting affected pods
4. Review recent changes that might impact memory usage

### Training Job Failure
1. Check job logs:
```bash
kubectl logs job/training-job-name -n cortex
```
2. Verify data pipeline integrity
3. Check model configuration
4. Review resource allocation
5. Restart job if necessary

### Network Connectivity Issues
1. Check network policies:
```bash
kubectl get networkpolicies -n cortex
```
2. Verify DNS resolution
3. Check service endpoints
4. Review recent network changes

## Performance Optimization

### GPU Optimization
1. Monitor GPU utilization:
```bash
nvidia-smi -l 1
```
2. Check batch size configuration
3. Verify data loading pipeline
4. Optimize model architecture if needed

### Memory Optimization
1. Review memory allocation:
```bash
kubectl describe nodes | grep -A 5 "Allocated resources"
```
2. Adjust resource limits
3. Optimize cache usage
4. Consider data sharding

## Troubleshooting Guide

### Common Issues

#### Pod Crashlooping
1. Check pod status:
```bash
kubectl describe pod <pod-name> -n cortex
```
2. Review container logs
3. Verify resource limits
4. Check for configuration errors

#### Training Stalls
1. Review training metrics
2. Check data pipeline
3. Verify network connectivity
4. Inspect GPU utilization

#### API Latency
1. Check service status
2. Review network policies
3. Monitor database connections
4. Verify cache hit rates

## Emergency Procedures

### Service Outage
1. Check system status
2. Review error logs
3. Verify infrastructure health
4. Initiate failover if needed
5. Notify stakeholders

### Data Loss Prevention
1. Stop affected services
2. Verify backup integrity
3. Begin data recovery
4. Document incident
5. Review security measures

## Contact Information

### On-Call Rotation
- Primary: ops-primary@yourdomain.com
- Secondary: ops-secondary@yourdomain.com
- Emergency: +1-XXX-XXX-XXXX

### Escalation Path
1. On-call engineer
2. DevOps team lead
3. System architect
4. CTO