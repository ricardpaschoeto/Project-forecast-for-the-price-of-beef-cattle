
# Exemplo mínimo (ajuste para sua conta)
resource "aws_s3_bucket" "data" {
  bucket = var.s3_bucket
}

# (Sugestão) Adicionar: IAM roles/policies, SageMaker Domain/Notebook, RDS (Aurora), ECR, MWAA, CloudWatch etc.
