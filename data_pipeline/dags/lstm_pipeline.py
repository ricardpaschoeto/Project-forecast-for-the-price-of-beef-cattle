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
