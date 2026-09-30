# 🥩 Forecasting Brazilian Beef Cattle Prices with Machine Learning

> End-to-end Machine Learning project for forecasting Brazilian beef cattle prices using market, climate, and time-series features.

[![Python](https://img.shields.io/badge/Pythonsvg]()
[![Machine Learning](https://img.shields.io/badge/M.svg]()
[![Time Series](https://img.shields.io/badgeorecasting-orange.svg]()
[![Optuna](https://img.shields.io/badge/Optuna-Hyperparametertion-red.svg]()
[![Status](https://img.shields.io/badge/Statusopment-yellow.svg]()

---

## 📌 Overview

Forecasting agricultural commodity prices is a complex challenge influenced by supply, demand, climate conditions, production costs, market sentiment, and macroeconomic factors.

This project aims to build a robust and scalable machine learning pipeline capable of predicting Brazilian beef cattle prices while integrating:

- Historical market prices
- Climate indicators
- Feature engineering techniques
- Hyperparameter optimization
- Experiment tracking
- MLOps best practices

The goal is not only to achieve accurate forecasts but also to create a production-ready forecasting framework that can support data-driven decision making in agribusiness.

---

## 🎯 Business Problem

Beef cattle producers, traders, and agribusiness companies need reliable forecasts to:

- Improve commercialization strategies
- Optimize purchasing decisions
- Reduce exposure to market volatility
- Support production planning
- Increase operational efficiency

By anticipating price movements, stakeholders can make more informed decisions and reduce uncertainty.

---

## 🏗️ Project Architecture

```text
D:.
|
|--README.md        
+---app
|   |   boi_app.py
|   |   fastapi_app.py
|   |   requirements.txt
|   |   test_scenarios.py
|   |   __init__.py
|           
+---config
|       settings.example.yaml
|       
+---data_pipeline
|   |   __init__.py
|   |   
|   +---dags
|   |   |   automated_retraining.py
|   |   |   clean_pipeline.py
|   |   |   data_validation.py
|   |   |   exploration_pipeline.py
|   |   |   extraction_pipeline.py
|   |   |   feature_engineering.py
|   |   |   load_pipeline.py
|   |   |   normalize_pipeline.py
|   |   |   scenario_point_forecasting.py
|   |   |   transform_pipeline.py
|   |   |   __init__.py
|   |   |   
|   +---great_expectations
|   |   +---checkpoints
|   |   |       daily_price_checkpoint.yml
|   |   |       
|   |   \---expectations
|   |           price_schema.json
|   |           
|   +---modules
|   |   |   __init__.py
|   |   |   
|   |   +---climate
|   |   |       climate_builder.py
|   |   |       inmet_api.py
|   |   |       noaa_api.py
|   |   |       terraclimate_api.py
|   |   |       __init__.py
|   |   |       
|   |   +---macro
|   |   |       bcb_api.py
|   |   |       macro_builder.py
|   |   |       __init__.py
|   |   |       
|   |   +---market
|   |   |       alpha_api.py
|   |   |       comex_api.py
|   |   |       market_builder.py
|   |   |       nasdaq_api.py
|   |   |       wits_api.py
|   |   |       __init__.py
|   |   |       
|   |   +---master
|   |   |       master_builder.py
|   |   |       __init__.py
|   |   |       
|   |   \---seasonality
|   |           calendar_builder.py
|   |           __init__.py
|   |           
|   +---operators
|   |       
|   +---sensors
|   |       alphav_FX_DAILY_USD_BRL_54d63c3650a1b1379814068ef20b94ed.parquet
|   |       dados_agro_atualizado.csv
|   |       dados_limpos.csv
|   |       dados_modelo_fs.csv
|   |       dados_modelo_lasso.csv
|   |       dados_modelo_lasso_norm.csv
|   |       dados_modelo_rfe.csv
|   |       
|   +---utils
|   |       cache_utils.py
|   |       createTables.py
|   |       date_utils.py
|   |       http_utils.py
|   |       Table.sql
|   |       __init__.py
|   |       
|   \---__pycache__
|           __init__.cpython-310.pyc
|           __init__.cpython-313.pyc
|           
+---infrastructure
|   \---terraform
|           main.tf
|           outputs.tf
|           providers.tf
|           variables.tf
|           
+---k8s
|       deployment.yaml
|       
+---ml-pipeline
|   |   evaluate.py
|   |   feature_specs.json
|   |   feature_store.py
|   |   pipeline.yaml
|   |   train.py
|   |   
|   +---config
|   |       feature_specs.json
|   |       
|   +---features
|   |       feature_specs.yaml
|   |       
|   +---models
|   |       model_lstm.h5
|   |       model_lstm.keras
|   |       
|   \---pipeline
|       +---data
|       +---evaluation
|       +---models
|       \---tracking
+---monitoring
|   |   model_monitor.py
|   |   
|   \---dashboards
+---notebooks
|       dados_agro.csv
|       dados_agro_atualizado.csv
|       evaluate.ipynb
|       normalidade_series.ipynb
|       projeto_01_eda.ipynb
|       projeto_01_extraction.ipynb
|       projeto_01_models.ipynb
|       projeto_01_transform.ipynb
|       transformed_data.csv
|       
+---powerbi
|       powerbi_integration.py
|       
\---tests
    |   __init__.py
    |   
    +---e2e
    |       test_model_serving.py
    |       test_prediction_pipeline.py
    |       
    +---integration
    |   |   test_api_integration.py
    |   |   test_pipeline_integration.py
    |   |   __init__.py
    |   |   
    |           
    +---unit
           test_data_processing.py
           test_feature_engineering.py
           test_model_training.py

            

