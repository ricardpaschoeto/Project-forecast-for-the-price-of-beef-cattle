
# Arquitetura Integrada (Negócio + MLOps)

Este documento descreve como o roteiro de negócio (fontes, variáveis, EDA, modelos, métricas, cronograma) 
se integra ao pipeline e à operação MLOps (IaC, orquestração, Feature Store, MLflow, deploy, monitoramento, BI).

- **Fontes & ETL** → Airflow (MWAA), Great Expectations, S3/RDS.
- **Feature Store** → SageMaker Feature Store (grupos mercado/clima/sazonal).
- **Experimentos/Registry** → MLflow.
- **Validação Temporal** → TSCV + walk-forward + backtesting.
- **Deploy** → SageMaker endpoints (blue/green/canary), API Gateway + Lambda/Fargate.
- **Observabilidade** → Evidently + CloudWatch; alertas; retraining automático.
- **Consumo** → App Flask + Power BI.
