# outputs.tf
output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
}

output "ecr_repository_url" {
  description = "ECR repository URL"
  value       = aws_ecr_repository.cortex.repository_url
}

output "checkpoint_bucket" {
  description = "S3 bucket for checkpoints"
  value       = aws_s3_bucket.checkpoints.id
}