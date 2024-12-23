#!/bin/bash
# deploy/scripts/deploy.sh

set -e

# Configuration
DOCKER_REGISTRY="your-registry"
KUBE_CONTEXT="your-cluster"
NAMESPACE="cortex"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Log function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

# Check dependencies
check_dependencies() {
    log "Checking dependencies..."
    
    if ! command -v kubectl &> /dev/null; then
        error "kubectl not found. Please install kubectl."
    fi
    
    if ! command -v docker &> /dev/null; then
        error "docker not found. Please install docker."
    fi
}

# Build and push Docker images
build_images() {
    log "Building Docker images..."
    
    docker build -t ${DOCKER_REGISTRY}/cortex:latest -f Dockerfile --target production .
    docker build -t ${DOCKER_REGISTRY}/cortex:dev -f Dockerfile --target development .
    
    log "Pushing images to registry..."
    docker push ${DOCKER_REGISTRY}/cortex:latest
    docker push ${DOCKER_REGISTRY}/cortex:dev
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log "Deploying to Kubernetes..."
    
    # Switch context
    kubectl config use-context ${KUBE_CONTEXT}
    
    # Create namespace if not exists
    kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply configurations
    kubectl apply -f deploy/k8s/configmap.yaml
    kubectl apply -f deploy/k8s/secret.yaml
    
    # Deploy components
    log "Deploying API nodes..."
    kubectl apply -f deploy/k8s/cortex-api.yaml
    
    log "Deploying training nodes..."
    kubectl apply -f deploy/k8s/cortex-trainer.yaml
    
    log "Deploying services and ingress..."
    kubectl apply -f deploy/k8s/service.yaml
    kubectl apply -f deploy/k8s/ingress.yaml
}

# Monitor deployment status
monitor_deployment() {
    log "Monitoring deployment status..."
    
    kubectl rollout status deployment/cortex-api -n ${NAMESPACE}
    kubectl rollout status deployment/cortex-trainer -n ${NAMESPACE}
    
    log "Checking pod status..."
    kubectl get pods -n ${NAMESPACE}
}

# Setup monitoring
setup_monitoring() {
    log "Setting up monitoring stack..."
    
    # Apply Prometheus configuration
    kubectl apply -f deploy/prometheus/
    
    # Apply Grafana configuration
    kubectl apply -f deploy/grafana/
    
    # Wait for monitoring stack
    kubectl rollout status deployment/prometheus -n ${NAMESPACE}
    kubectl rollout status deployment/grafana -n ${NAMESPACE}
}

# Cleanup function
cleanup() {
    warn "Cleaning up resources..."
    kubectl delete pods -n ${NAMESPACE} --field-selector status.phase=Failed
    kubectl delete pods -n ${NAMESPACE} --field-selector status.phase=Succeeded
}

# Main deployment process
main() {
    log "Starting deployment process..."
    
    check_dependencies
    
    # Build and deploy based on arguments
    case "$1" in
        "build")
            build_images
            ;;
        "deploy")
            deploy_kubernetes
            monitor_deployment
            ;;
        "monitor")
            setup_monitoring
            ;;
        "all")
            build_images
            deploy_kubernetes
            setup_monitoring
            monitor_deployment
            ;;
        *)
            error "Invalid command. Use: build, deploy, monitor, or all"
            ;;
    esac
    
    cleanup
    log "Deployment completed successfully!"
}

# Execute with error handling
if [ "$#" -ne 1 ]; then
    error "Usage: $0 [build|deploy|monitor|all]"
fi

main "$1"