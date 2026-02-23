
"""scenario_point_forecasting.py (INFERENCE-ONLY)

Ajuda a **montar o DataFrame de entrada** e **chamar o modelo treinado**.

> Este módulo **NÃO TREINA** e **NÃO** faz seleção de features. Todo o pré-processamento e
> seleção devem estar **dentro do pipeline salvo** no momento do treinamento (MLflow/Joblib).

Uso típico (API / Streamlit):
-----------------------------
model = mlflow.sklearn.load_model(MODEL_URI)  # ou joblib.load(...)
from scenario_point_forecasting import PointScenarioConfig, predict_from_point_scenario
conf = PointScenarioConfig(date_col='Date', target_col='preco_arroba', exog_cols=['selic','ipca','cambio'])

pred = predict_from_point_scenario(
    model,
    date_value='2026-03-10',
    exog_values={'selic': 10.5, 'ipca': 0.35, 'cambio': 5.10},
    conf=conf,
)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


@dataclass
class PointScenarioConfig:
    """Esquema do input para previsão por cenário **pontual**.

    - date_col: nome da coluna de data no pipeline treinado
    - target_col: nome do alvo (apenas para referência/validação)
    - exog_cols: lista de exógenas esperadas pelo pipeline treinado
    - categorical_cols: categóricas opcionais que o pipeline pode esperar
    """
    date_col: str
    target_col: str
    exog_cols: List[str]
    categorical_cols: Optional[List[str]] = None


def make_inference_row(
    *,
    date_value: str | pd.Timestamp,
    exog_values: Dict[str, float],
    conf: PointScenarioConfig,
    categorical_values: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Cria um DataFrame (1 linha) no formato esperado pelo pipeline treinado.

    **Não aplica pré-processamento** (o pipeline treinado deve conter isso).
    """
    row: Dict[str, Any] = {conf.date_col: pd.to_datetime(date_value)}

    # exógenas obrigatórias
    for c in conf.exog_cols:
        if c not in exog_values:
            raise KeyError(f"Exógena '{c}' não informada no cenário.")
        row[c] = exog_values[c]

    # categóricas opcionais
    for c in (conf.categorical_cols or []):
        if not categorical_values or c not in categorical_values:
            raise KeyError(f"Categórica '{c}' não informada no cenário.")
        row[c] = categorical_values[c]

    return pd.DataFrame([row])


def predict_from_point_scenario(
    model: Pipeline,
    *,
    date_value: str | pd.Timestamp,
    exog_values: Dict[str, float],
    conf: PointScenarioConfig,
    categorical_values: Optional[Dict[str, Any]] = None,
) -> float:
    """Prevê o alvo para a data-alvo com valores **pontuais** de exógenas.

    O `model` deve ser **um pipeline treinado** (sklearn) contendo todas as etapas de
    pré-processamento (scaler/encoder/calendário/seleção, se houver) + estimador final.
    """
    X = make_inference_row(
        date_value=date_value,
        exog_values=exog_values,
        conf=conf,
        categorical_values=categorical_values,
    )
    y_hat = model.predict(X)
    return float(np.asarray(y_hat).reshape(-1)[0])
