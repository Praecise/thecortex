# deploy/terraform/main.tf

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# EKS Cluster
module "eks" {
  source          = "terraform-aws-modules/eks/aws"
  cluster_name    = var.cluster_name
  cluster_version = "1.27"
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  # Node groups configuration
  eks_managed_node_groups = {
    gpu = {
      name = "gpu-node-group"
      
      instance_types = ["p3.2xlarge"]
      capacity_type  = "ON_DEMAND"
      
      min_size     = 1
      max_size     = 5
      desired_size = 3
      
      labels = {
        role = "training"
      }
      
      taints = [
        {
          key    = "nvidia.com/gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
    }
    
    cpu = {
      name = "cpu-node-group"
      
      instance_types = ["c5.2xlarge"]
      capacity_type  = "SPOT"
      
      min_size     = 2
      max_size     = 10
      desired_size = 3
      
      labels = {
        role = "api"
      }
    }
  }
}

# VPC Configuration
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  
  name = "cortex-vpc"
  cidr = "10.0.0.0/16"
  
  azs             = ["${var.aws_region}a", "${var.aws_region}b", "${var.aws_region}c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  
  enable_nat_gateway = true
  single_nat_gateway = true
  
  tags = {
    Environment = var.environment
    Project     = "cortex"
  }
}

# ECR Repository
resource "aws_ecr_repository" "cortex" {
  name = "cortex"
  
  image_scanning_configuration {
    scan_on_push = true
  }
}

# S3 Bucket for checkpoints
resource "aws_s3_bucket" "checkpoints" {
  bucket = "cortex-checkpoints-${var.environment}"
  
  versioning {
    enabled = true
  }
  
  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "cortex" {
  name              = "/aws/eks/${var.cluster_name}/cortex"
  retention_in_days = 30
}
