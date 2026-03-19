# -*- coding: utf-8 -*-
"""
Pipeline completo para previsão de séries temporais (one-step forecasting)
usando TensorFlow + Optuna + MLflow.

Autor: Copilot
"""
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Callable
import warnings

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import LearningRateScheduler

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

import optuna
import mlflow
import mlflow.tensorflow
import matplotlib.pyplot as plt
from  data_pipeline.dags.normalize_pipeline import normalize_time_series, inverse_transform_column, PipelineConfig

def create_windowed_df(X_df:pd.DataFrame,
                       y_df: pd.Series, 
                       window_size: int): 

    """
    Constrói janelas multivariadas utilizando DataFrames.

    Args:
        X_df: DataFrame (T, n_features).
        y_df: Série (T,) do target.
        window_size: número de passos usados como entrada.

    Returns:
        X_list: lista onde cada item é um DataFrame (window_size x n_features)
        y_list: Series com o y futuro correspondente.
    """

    X_list, y_list = [], []
    T = len(X_df)

    for i in range(T - window_size):
        window_X = X_df.iloc[i : i + window_size].copy()
        window_y = y_df.iloc[i + window_size]

        X_list.append(window_X)
        y_list.append(window_y)

    return X_list, pd.Series(y_list, name=y_df.name)

def convert_windows_to_numpy(X_list):

    """
    Converte uma lista de DataFrames (cada janela) para numpy (N, window, n_features)
    """

    X_np = np.stack([df.values for df in X_list])

    return X_np

def build_lstm_model(trial: optuna.Trial, window_size: int, n_features: int) -> tf.keras.Model:
    """
    Constrói um modelo LSTM com hiperparâmetros sugeridos pelo Optuna.

    Args:
        trial: Objeto Optuna Trial para sugerir hiperparâmetros.
        window_size: Tamanho da janela de entrada.
        n_features: Numero de features.

    Returns:
        Modelo LSTM compilado.
    """

    units = trial.suggest_int("n_units", 32, 128)
    layers = trial.suggest_int("n_layers", 1, 5, 10)
    dropout_p = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

    model = Sequential()

    # Primeira camada
    return_sequences = (layers > 1)
    model.add(LSTM(units, input_shape=(window_size, n_features),  
                   return_sequences=return_sequences))
    if dropout_p > 0:
        model.add(Dropout(dropout_p))

    # Camadas intermediárias
    for _ in range(1, layers - 1):
        model.add(LSTM(units, return_sequences=True))
        if dropout_p > 0:
            model.add(Dropout(dropout_p))

    # Última camada
    if layers > 1:
        model.add(LSTM(units, return_sequences=False))
        if dropout_p > 0:
            model.add(Dropout(dropout_p))

    
    model.add(Dense(1))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='mae'
    )

    return model

def train_optuna(
        X_list: list,
        y_series: pd.Series,
        window_size: int,
        n_splits: int = 3,
        n_trials: int = 20,
) -> optuna.Study:
    """
    Realiza otimização de hiperparâmetros usando Optuna.

    Args:
        data: Série temporal como numpy array.
        window_size: Tamanho da janela de entrada.
        n_split: Número de splits para TimeSeriesSplit.
        n_trials: Número de trials para Optuna.

    Returns:
        Estudo Optuna com os resultados da otimização.
    """

    # Converte para numpy apenas aqui
    X_np = convert_windows_to_numpy(X_list)
    y_np = y_series.values

    tscv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial: optuna.Trial) -> float:
        mae_scores = []

        for train_idx, val_idx in tscv.split(X_np):
            X_train, X_val = X_np[train_idx], X_np[val_idx]
            y_train, y_val = y_np[train_idx], y_np[val_idx]

            model = build_lstm_model(trial, window_size, X_np.shape[-1])

            # Scheduler simples: reduz lr a cada época
            def scheduler(epoch, lr):
                return lr * 0.95

            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val), 
                epochs=30, 
                batch_size=32, 
                verbose=0,
                callbacks=[LearningRateScheduler(scheduler)]
            )

            preds = model.predict(X_val).ravel()
            mae_score = mean_squared_error(y_val, preds)
            mae_scores.append(mae_score)

        return float(np.mean(mae_scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    return study

# Função para criar o experimento
def cria_experimento(experiment_name):

    # Verifica se o experimento já existe pelo nome
    if experiment := mlflow.get_experiment_by_name(experiment_name):
        # Se o experimento existir, retorna seu ID
        return experiment.experiment_id
    else:
        # Se o experimento não existir, cria um novo e retorna seu ID
        return mlflow.create_experiment(experiment_name)
    
def train_pipeline(df: pd.DataFrame,
                   path: str,
                   target_col: str,
                   feature_cols: Optional[list[str]] = None,  
                   window_size: int = 30,
                   test_size: int = 60,
                   optuna_trails: int = 20,
                   tscv_splits: int = 3,
                   experiment_name= "boi_gordo") -> None:
    """
    Pipeline completa de treinamento e avaliação do modelo LSTM.

    Args:
        df: DataFrame contendo a série temporal.
        window_size: Tamanho da janela de entrada.
        test_size: Tamanho do conjunto de teste.

    Returns:
        modelo final, previsões do teste.
    """
    assert target_col in df.columns

    # Seleciona features automaticamente
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    cols = feature_cols + [target_col]
    cfg = PipelineConfig()
    cfg.interpolate = True

    # 1) Split temporal
    train_df = df.iloc[:-test_size].copy()
    test_df  = df.iloc[-(test_size + window_size):].copy()

    # 2) Fit + transform no TREINO (salva artefatos por coluna)
    train_norm, _ = normalize_time_series(
        isTrain=True,
        df=train_df,
        cols=cols,
        config=cfg,
        #artifact_dir="models"  # diretório preferido

    )

    # 3) Transform no TESTE usando history (últimos 'd' valores originais do treino por coluna)
    test_norm, _ = normalize_time_series(
        isTrain=False,
        df=test_df,
        cols=cols,
        config=cfg,
        #artifact_dir="models"
    )

    # Split
    X_train_df = train_norm[[f"{c}_norm" for c in feature_cols]]
    y_train_df = train_norm[f"{target_col}_norm"]
    X_test_df = test_norm[[f"{c}_norm" for c in feature_cols]]
    y_test_df = test_norm[f"{target_col}_norm"]

    # Criar janelas
    X_train_list, y_train_series = create_windowed_df(X_train_df, y_train_df, window_size)
    X_test_list, y_test_series = create_windowed_df(X_test_df, y_test_df, window_size)

    # Optuna
    study = train_optuna(X_train_list, y_train_series, window_size, n_splits=tscv_splits, n_trials=optuna_trails)

    # Treino final
    id_experimento = cria_experimento(experiment_name)

    with mlflow.start_run(experiment_id = id_experimento, run_name = experiment_name, nested = True):

        mlflow.log_params(study.best_params)

        X_train_np = convert_windows_to_numpy(X_train_list)
        y_train_np = y_train_series.values
        X_test_np = convert_windows_to_numpy(X_test_list)
        y_test_np = y_test_series.values

        model = build_lstm_model(study.best_trial, window_size, X_train_np.shape[-1])

        def scheduler(epoch, lr):
            return lr * 0.95

 
        model.fit(X_train_np, y_train_np,
                  epochs=50,
                  verbose=1,
                  batch_size=study.best_params.get("batch_size", 32),
                  callbacks=[LearningRateScheduler(scheduler)])
       

        preds = model.predict(X_test_np).ravel()

        y_pred_norm_df = pd.DataFrame({"boi_negociado_norm": preds}, index=test_norm.index[-len(preds):])

        y_pred_inv = inverse_transform_column(
            df_norm = y_pred_norm_df,
            cols= [target_col],
            start_values=train_df[[target_col]]
        )

        # Métricas

        mae_score = mean_absolute_error(y_test_np, preds)
        mse_score = np.sqrt(mean_squared_error(y_test_np, preds))
        mape_score = np.mean(np.abs((y_test_np - preds) / (np.abs(y_test_np) + 1e-8))) * 100

        mlflow.log_metric("MAE", mae_score)
        mlflow.log_metric("MSE", mse_score)
        mlflow.log_metric("MAPE", mape_score)
        try:
            model.save(path)
        except Exception as e:
            warnings.warn(f"Falha ao salvar o modelo: {e}")
        # mlflow.keras.log_model(model, artifact_path="model_lstm")
        # print(mlflow.get_artifact_uri("model_lstm"))

        # Gráfico final
        plt.figure(figsize=(10, 5))
        plt.plot(y_test_np, label="Real (norm)")
        plt.plot(preds, label="Prévia (norm)")
        plt.legend()
        plt.title("Previsão vs Real (normalizado)")
        plt.tight_layout()
        plt.savefig("plot_norm_df.png")
        mlflow.log_artifact("plot_norm_df.png")

    return model, y_pred_inv, (mae_score, mse_score, mape_score)


caminho = Path(os.path.abspath(__file__))

root_dir = caminho.parent.parent
df_path = os.path.join(root_dir, 'data_pipeline' ,'sensors', 'dados_modelo_lasso.csv')
model_path = os.path.join(caminho.parent, 'models','model_lstm.h5')
model, y_pred_inv, _ = train_pipeline(df=pd.read_csv(df_path, index_col=0, parse_dates=True), path=model_path,target_col="boi_negociado", optuna_trails=20, tscv_splits=3, 
                                      test_size=60)

print(y_pred_inv)