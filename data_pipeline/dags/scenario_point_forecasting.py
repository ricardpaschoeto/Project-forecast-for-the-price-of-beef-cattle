
"""scenario_point_forecasting.py

Previsão do alvo (ex.: preço da arroba do boi gordo) a partir de um **cenário pontual**
(um único valor por exógena) para uma **data-alvo específica**.

Entrada do usuário
------------------
- date_value: data-alvo (ex.: 2026-03-10)
- exog_values: dict com 1 valor por exógena (SELIC, IPCA, câmbio, ...)

Saída
-----
- 1 número: previsão do preço na data-alvo.

Não há sequência temporal na entrada, então LSTM/lag/rolling não se aplicam.

Inclui seleção moderna de features (opcional): ElasticNetCV + SelectFromModel + TimeSeriesSplit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit


def add_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Adiciona features de calendário a partir de uma coluna de data."""
    out = df.copy()
    dt = pd.to_datetime(out[date_col])
    out["dow"] = dt.dt.dayofweek
    out["week"] = dt.dt.isocalendar().week.astype(int)
    out["month"] = dt.dt.month
    out["quarter"] = dt.dt.quarter
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)
    return out


class TimeFeaturesTransformer(BaseEstimator, TransformerMixin):
    """Transformer sklearn para adicionar features de calendário."""

    def __init__(self, date_col: str):
        self.date_col = date_col

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return add_time_features(X, self.date_col)


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """Remove colunas explicitamente."""

    def __init__(self, cols: Optional[Sequence[str]] = None):
        self.cols = list(cols or [])

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.cols:
            return X
        cols_to_drop = [c for c in self.cols if c in X.columns]
        return X.drop(columns=cols_to_drop)


@dataclass
class PointScenarioConfig:
    """Configuração do pipeline para cenário pontual."""

    date_col: str
    target_col: str
    exog_cols: List[str]

    categorical_cols: Optional[List[str]] = None
    drop_cols: Optional[List[str]] = None

    imputer_num_strategy: str = "median"
    scale_numeric: bool = True
    onehot_sparse: bool = True

    feature_selection: bool = True
    n_splits: int = 5
    selection_threshold: str | float = "median"

    model_params: Optional[Dict[str, Any]] = None


def make_pipeline(conf: PointScenarioConfig) -> Pipeline:
    """Pipeline: time-features -> preprocess -> (select) -> model."""

    model_params = conf.model_params or {}
    cat_cols = conf.categorical_cols or []

    # Feature engineering
    fe_steps = [("time", TimeFeaturesTransformer(conf.date_col))]

    drop_list = list(dict.fromkeys([conf.target_col] + (conf.drop_cols or [])))
    fe_steps.append(("drop", DropColumnsTransformer(drop_list)))

    fe = Pipeline(steps=fe_steps)

    # Preprocess
    # num_steps = [("imputer", SimpleImputer(strategy=conf.imputer_num_strategy))]
    # if conf.scale_numeric:
    #     num_steps.append(("scaler", StandardScaler(with_mean=not conf.onehot_sparse)))
    # num_pipe = Pipeline(num_steps)

    # cat_pipe = Pipeline(
    #     steps=[
    #         ("imputer", SimpleImputer(strategy="most_frequent")),
    #         ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=conf.onehot_sparse)),
    #     ]
    # )

    # pre = ColumnTransformer(
    #     transformers=[
    #         ("num", num_pipe, selector(dtype_include=np.number)),
    #         ("cat", cat_pipe, cat_cols),
    #     ],
    #     remainder="drop",
    #     verbose_feature_names_out=False,
    # )

    steps: List[Tuple[str, Any]] = [("fe", fe)] # ("pre", pre)

    # Feature selection (opcional)
    if conf.feature_selection:
        cv = TimeSeriesSplit(n_splits=conf.n_splits)
        enet = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9, 1.0],
            cv=cv,
            random_state=42,
            max_iter=5000,
        )
        steps.append(("select", SelectFromModel(enet, threshold=conf.selection_threshold)))

    model = HistGradientBoostingRegressor(**model_params)
    steps.append(("model", model))

    return Pipeline(steps)


def fit_single_model(df: pd.DataFrame, *, conf: PointScenarioConfig) -> Pipeline:
    """Treina um único modelo: (exógenas + calendário) -> target na mesma data."""

    data = df.sort_values(conf.date_col).copy()
    y = data[conf.target_col].astype(float)

    keep_cols = [conf.date_col] + list(conf.exog_cols) + (conf.categorical_cols or [])
    X = data[keep_cols].copy()

    pipe = make_pipeline(conf)
    pipe.fit(X, y)
    return pipe


def predict_from_point_scenario(
    model: Pipeline,
    *,
    date_value: str | pd.Timestamp,
    exog_values: Dict[str, float],
    conf: PointScenarioConfig,
    categorical_values: Optional[Dict[str, Any]] = None,
) -> float:
    """Prevê o target para uma data-alvo usando valores pontuais das exógenas."""

    row: Dict[str, Any] = {conf.date_col: pd.to_datetime(date_value)}

    for c in conf.exog_cols:
        if c not in exog_values:
            raise KeyError(f"Exógena '{c}' não informada no cenário.")
        row[c] = exog_values[c]

    for c in (conf.categorical_cols or []):
        if not categorical_values or c not in categorical_values:
            raise KeyError(f"Categórica '{c}' não informada no cenário.")
        row[c] = categorical_values[c]

    X = pd.DataFrame([row])
    y_hat = model.predict(X)
    return float(np.asarray(y_hat).reshape(-1)[0])


def save_pipeline(model: Pipeline, conf: PointScenarioConfig, out_dir: str | Path) -> Path:
    """Salva o pipeline e a configuração."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "pipeline.joblib")
    (out_dir / "config.json").write_text(json.dumps(asdict(conf), indent=2, ensure_ascii=False), encoding="utf-8")
    return out_dir


def load_pipeline(in_dir: str | Path) -> Tuple[Pipeline, PointScenarioConfig]:
    """Carrega pipeline e configuração."""

    in_dir = Path(in_dir)
    model = joblib.load(in_dir / "pipeline.joblib")
    conf = PointScenarioConfig(**json.loads((in_dir / "config.json").read_text(encoding="utf-8")))
    return model, conf
