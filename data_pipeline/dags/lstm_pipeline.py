# -*- coding: utf-8 -*-
"""
ARIMA/SARIMA Pipeline Completo
- Carrega série temporal de CSV (date_col, value_col)
- Testa estacionariedade (ADF, KPSS) e sugere d (e D se sazonal)
- Plota ACF/PACF
- Faz grid search por AIC para ARIMA ou SARIMA
- Ajusta melhor modelo, valida em holdout e prevê horizonte futuro
Autor: você :)
"""


"""
Pipeline completo para previsão de séries temporais (one-step forecasting)
usando TensorFlow + Optuna + MLflow.

Autor: Copilot
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Callable

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import LearningRateScheduler

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

import optuna
import mlflow
import mlflow.tensorflow
import matplotlib.pyplot as plt

def create_windwed_dataset(data: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
 
    """
    Cria janelas de dados para previsão one-step.

    Args:
        data: Série temporal como numpy array.
        window_size: Tamanho da janela de entrada.

    Returns:
        X: Entradas no formato [amostras, janela, 1]
        y: Próximo valor da série.
    """

    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X).reshape(-1,window_size, 1), np.array(y)

def build_lstm_model(trial: optuna.Trial, window_size: int) -> tf.keras.Model:
    """
    Constrói um modelo LSTM com hiperparâmetros sugeridos pelo Optuna.

    Args:
        trial: Objeto Optuna Trial para sugerir hiperparâmetros.
        window_size: Tamanho da janela de entrada.

    Returns:
        Modelo LSTM compilado.
    """

    n_units = trial.suggest_int("n_units", 32, 128)
    n_layers = trial.suggest_int("n_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

    model = Sequential()

    for i in range(n_layers):
        return_sequences = (i < n_layers - 1)
        model.add(LSTM(n_units, return_sequences=return_sequences))
        if dropout > 0:
            model.add(tf.keras.layers.Dropout(dropout))
    
    model.add(Dense(1))


    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='mse'
    )

    return model

def train_optuna(
        data: np.ndarray,
        window_size: int,
        n_split: int = 3,
        n_trials: int = 10,
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

    X, y = create_windwed_dataset(data, window_size)

    def objective(trial: optuna.Trial) -> float:
        tscv = TimeSeriesSplit(n_splits=n_split)
        mae_scores = []

        for train_idx, val_idx in tscv.split(data):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = build_lstm_model(trial, window_size)

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

            preds = model.predict(X_val, verbose=0).flatten()
            mae_score = mean_squared_error(y_val, preds)
            mae_scores.append(mae_score)

        return np.mean(mae_scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    return study
    
def train_pipeline(data: np.ndarray, 
                   window_size: int,
                   test_size: int = 60) -> None:
    """
    Pipeline completa de treinamento e avaliação do modelo LSTM.

    Args:
        data: Série temporal como numpy array.
        window_size: Tamanho da janela de entrada.
        test_size: Tamanho do conjunto de teste.

    Returns:
        modelo final, previsões do teste.
    """

    mlflow.set_experiment("Previsão_Boi_Gordo")

    with mlflow.start_run():

        study = train_optuna(data, window_size)
        best_params = study.best_params

        mlflow.log_params(best_params)

        # Treina modelo final com os melhores hiperparâmetros
        model = build_lstm_model(study.best_trial, window_size)

        train_data = data[:-test_size]
        test_data = data[-(test_size + window_size):]

        X_train, y_train = create_windwed_dataset(train_data, window_size)
        X_test, y_test = create_windwed_dataset(test_data, window_size)

        def scheduler(epoch, lr):
            return lr * 0.95

        model.fit(
            X_train, y_train,
            epochs=50, 
            batch_size=32, 
            verbose=1,
            callbacks=[LearningRateScheduler(scheduler)]
        )

        preds = model.predict(X[-test_size:], verbose=0).flatten()

        # Métricas
        mae_score = mean_absolute_error(y_test, preds)
        mse_score = np.sqrt(mean_squared_error(y_test, preds))
        mape_score = np.mean(np.abs((y_test - preds) / y_test)) * 100

        mlflow.log_metric("MAE", mae_score)
        mlflow.log_metric("MSE", mse_score)
        mlflow.log_metric("MAPE", mape_score)
        mlflow.tensorflow.log_model(model, "model_lstm")

        # Gráfico final
        plt.figure(figsize=(10, 5))
        plt.plot(y_test, label="Real")
        plt.plot(preds, label="Previsão")
        plt.legend()
        plt.title("Previsão vs Real - Boi Gordo")
        plt.savefig("forecast_plot.png")
        mlflow.log_artifact("forecast_plot.png")

    return model, preds, (mae_score, mse_score, mape_score)
