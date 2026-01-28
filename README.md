
# Projeto Boi Gordo — Esqueleto MLOps/ML

Repositório base para **previsão de preços da arroba do boi gordo** com práticas de **MLOps**. 
Integra **roteiro de negócio/modelagem** e **processo MLOps** em um único projeto.

> **Principais pilares**: IaC (Terraform) · Pipelines de Dados (Airflow/MWAA) · Feature Store (SageMaker) · 
> Experimentos/Registry (MLflow) · Treino/Avaliação com validação temporal · Deploy (SageMaker) · 
> Observabilidade (Evidently + CloudWatch) · App Flask · Power BI · CI/CD (GitHub Actions).

---

## 📁 Estrutura do Repositório

```
projeto-boi-gordo/
│
├─ infrastructure/                     # Infraestrutura do projeto
│  └─ terraform/                       # Templates Terraform
│
├─ data-pipeline/                      # Pipeline de dados (Airflow/ETL)
│  ├─ dags/                            # DAGs do Airflow
│  │   ├─ ingestion_market_dag.py      # Dag: mercado
│  │   ├─ ingestion_macro_dag.py       # Dag: macroeconomia
│  │   ├─ ingestion_climate_dag.py     # Dag: clima
│  │   ├─ build_master_table_dag.py    # Dag principal (Master Table)
│  │
│  ├─ operators/                       # Operadores customizados Airflow
│  │   ├─ market_operator.py
│  │   ├─ macro_operator.py
│  │   ├─ climate_operator.py
│  │   ├─ calendar_operator.py
│  │   └─ master_operator.py
│  │
│  ├─ sensors/                         # Sensores Airflow
│  │   ├─ file_sensor.py
│  │   └─ partition_sensor.py
│  │
│  ├─ utils/                           # Utilitários usados no pipeline
│  │   ├─ __init__.py
│  │   ├─ date_utils.py                # Datas e intervalos
│  │   ├─ cache_utils.py               # Cache determinístico
│  │   ├─ http_utils.py                # GET com retry/backoff
│  │   └─ s3_utils.py                  # Funções S3
│  │
│  ├─ modules/                         # Núcleo de ETL (organizado por domínio)
│  │   ├─ market/                      # Dados de mercado e tarifas
│  │   ├─ macro/                       # Dados macroeconômicos
│  │   ├─ climate/                     # Dados climáticos
│  │   ├─ seasonality/                 # Dados sazonais e calendário
│  │   └─ master/                      # Mestre: une todos os módulos
│  │
│  ├─ great_expectations/              # Data Quality
│  │   ├─ expectations/
│  │   ├─ checkpoints/
│  │   └─ great_expectations.yml
│  │
│  └─ README.md                        # Doc específica do pipeline de dados
│
├─ ml-pipeline/                        # Pipeline de Machine Learning
│  ├─ features/                        # Definições YAML das features
│  ├─ feature_store.py                 # Interface com Feature Store
│  ├─ train.py                         # Treinamento de modelos
│  ├─ evaluate.py                      # Avaliação e métricas
│  ├─ pipeline.yaml                    # Pipeline declarativo (MLflow)
│  └─ MODEL_CARD.md                    # Ficha técnica do modelo
│
├─ monitoring/                         # Monitoramento em produção
│  └─ model_monitor.py                 # Drift, métricas e alertas
│
├─ app/                                # Aplicação (dashboard + API REST)
│  ├─ app.py
│  ├─ templates/
│  └─ static/
│
├─ powerbi/                            # Integração Power BI
│  └─ powerbi_integration.py
│
├─ k8s/                                # Deploy em Kubernetes
│  └─ deployment.yaml
│
├─ tests/                              # Testes (unitário, integração, e2e)
│  ├─ unit/
│  ├─ integration/
│  └─ e2e/
│
├─ docs/                               # Documentação
│  └─ ARCHITECTURE.md
│
├─ config/                             # Configurações gerais
│  └─ settings.example.yaml
│
├─ .github/workflows/                  # CI/CD (GitHub Actions)
│  └─ ml-pipeline.yml
│
├─ Dockerfile                          # Imagem Docker do projeto
├─ requirements.txt                    # Dependências Python
├─ .gitignore
├─ .dvcignore
├─ dvc.yaml                            # Pipeline de dados (DVC opcional)
└─ README.md                           # Este arquivo
```

---

## 🚀 Quickstart (dev local)

> Pré-requisitos: Python 3.10+, Git, AWS CLI configurado, Terraform, Docker (opcional para app), Make (opcional).

```bash
# criar venv e instalar dependências
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# (opcional) configurar variáveis de ambiente
cp config/settings.example.yaml config/settings.yaml
cp .env.example .env

# inicializar MLflow local (artefatos em ./mlruns)
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

# executar um treino de exemplo
python ml-pipeline/train.py --experiment "boi-gordo-dev" --window 30

# avaliar
python ml-pipeline/evaluate.py --run-id <MLFLOW_RUN_ID>

# executar app Flask (desenvolvimento)
export FLASK_APP=app/app.py && flask run -p 5000
```

---

## 🔗 Mapeamento ROTEIRO → MLOps (resumo)
- **Fontes & ETL** (B3, CEPEA/USP, BCB, IBGE, INMET, USDA/CME, APIs) → `data-pipeline/dags/*` + Great Expectations.
- **Features mercado/macro/clima/sazonais** → `ml-pipeline/feature_store.py` + `features/*.yaml`.
- **Modelos** (ARIMA/Prophet/LSTM/GRU/Transformers, RF/XGB/LGBM, SVR, ensembles) → `ml-pipeline/train.py`.
- **Validação temporal & métricas** (RMSE/MAE/MAPE, Directional Accuracy, backtesting) → `ml-pipeline/evaluate.py`.
- **Deploy** (SageMaker + API Gateway/Lambda/Fargate) → IaC em `infrastructure/terraform/` e app Flask.
- **Monitoramento** (drift, performance, alertas) → `monitoring/model_monitor.py`.
- **Power BI** (dashboards, refresh) → `powerbi/powerbi_integration.py`.

---

## 🧪 CI/CD
O workflow `ml-pipeline.yml` executa: validação de dados → treino → avaliação → deploy (staging→prod via Terraform), 
com testes unitários/integrados/E2E em *gates*.

---

## 🔒 Governança
Modelos são registrados no **MLflow Model Registry** com *model cards* (dados, pré-processamento, riscos, intended use), 
lineage/versionamento, aprovação por estágios (Staging → Production) e auditoria.

---

## 📅 Roadmap sugerido
1. Provisionar infra mínima (S3, IAM, SageMaker, RDS dev) via Terraform.
2. Implementar ingestão de **1–2 fontes** (ex.: CEPEA/BCB) e expectations básicas.
3. Publicar **MVP de features** na Feature Store (janelas e sazonais) e treinar 2–3 modelos.
4. Expor endpoint de *staging* no SageMaker e conectar a app Flask.
5. Ativar monitoramento (drift + performance) e refresh para Power BI.
6. Habilitar *blue/green* e *retraining* automático.

