# feature_engineering.py
from __future__ import annotations
from dataclasses import dataclass
from itertools import groupby
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.feature_selection import RFE
from sklearn.feature_selection import mutual_info_regression

from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor

import pandas as pd


# =========================
# 1) TRANSFORMERS BÁSICOS
# =========================

class TimeFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Adiciona features de calendário a partir de uma coluna de data.
    - 'dow' (0=segunda..6=domingo)
    - 'week' ISO
    - 'month', 'quarter'
    - codificação cíclica p/ 'month' (sin/cos)
    """
    def __init__(self, date_col: str):
        self.date_col = date_col

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        dt = pd.to_datetime(X[self.date_col], dayfirst=True, errors="coerce")
        X["dow"] = dt.dt.dayofweek
        X["week"] = dt.dt.isocalendar().week.astype(int)
        X["month"] = dt.dt.month
        X["quarter"] = dt.dt.quarter
        # cíclicas p/ mês (evita ordinalidade artificial)
        X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12)
        X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12)
        return X

class LagFeatures(BaseEstimator, TransformerMixin):
    """
    Cria lags da variável alvo no próprio DataFrame (cuidado com vazamento).
    - target_col: nome da coluna de target no DF (deve estar presente em X).
    - lags: tupla/lista de defasagens (ex.: (1, 7, 28))
    Pré‑requisito: X deve estar ordenado por data ANTES desta etapa.
    """
    def __init__(self, target_col: str, lags=(1, 7, 28), date_col: Optional[str] = None):
        self.target_col = target_col
        self.lags = lags
        self.date_col = date_col

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        if self.date_col:
            X = X.sort_values(self.date_col)
        for lag in self.lags:
            X[f"{self.target_col}_lag{lag}"] = X[self.target_col].shift(lag)
        return X


class RollingStats(BaseEstimator, TransformerMixin):
    """
    Cria estatísticas móveis da variável alvo (mean/std/min/max) com janelas definidas.
    - windows: lista de janelas (ex.: [7, 14, 28])
    Por padrão, usa closed='left' (janela até o dia anterior) para evitar vazamento.
    """
    def __init__(self, target_col: str, windows: List[int], date_col: Optional[str] = None):
        self.target_col = target_col
        self.windows = windows
        self.date_col = date_col

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        if self.date_col:
            X = X.sort_values(self.date_col)
        for w in self.windows:
            roll = (
                X[self.target_col]
                .rolling(window=w, min_periods=1)
                .agg(["mean", "std", "min", "max"])
            )
            X[f"{self.target_col}_mean_{w}"] = roll["mean"].shift(1)  # closed='left'
            X[f"{self.target_col}_std_{w}"]  = roll["std"].shift(1)
            X[f"{self.target_col}_min_{w}"]  = roll["min"].shift(1)
            X[f"{self.target_col}_max_{w}"]  = roll["max"].shift(1)
        return X

class FeatureSelection():
    def __init__(self, df: pd.DataFrame, target_col: str):
        self.df = df
        self.y = df[target_col]
        self.X = df.drop(columns=[target_col])
        self.feature_names = self.X.columns

    def _create_lagged_features(self, lags=12):
        X_lagged = pd.concat(
            [
                self.X.shift(lag).add_suffix(f"_lag{lag}") for lag in range(1, lags + 1)
            ],
            axis=1
        )

        X_lagged = X_lagged.dropna()
        y_aligned = self.y.iloc[lags:]

        return X_lagged, y_aligned
    
    def temporal_mi_selection(self, lags=12, top_k=30):
        X_lagged, y_lagged = self._create_lagged_features(lags)
        mi = mutual_info_regression(X_lagged, y_lagged)
        mi_series = pd.Series(mi, index=X_lagged.columns)

        # Agrupar MI por feature original
        grouped_mi = mi_series.groupby(
            lambda x: x.split("_lag")[0]
            ).sum()

        return grouped_mi.sort_values(ascending=False).head(top_k).index.tolist()  # Retorna os nomes das top k features
    
    def temporal_xgb_selection(self, lags=12, top_k=20):
        X_lagged, y_lagged = self._create_lagged_features(lags)

        xgb = XGBRegressor(
            n_estimators=500, 
            max_depth=6, 
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )

        xgb.fit(X_lagged, y_lagged)

        importances = pd.Series(
            xgb.feature_importances_,
              index=X_lagged.columns
            )

        # Agrupar importância por feature original
        grouped_importance = (
            importances
            .groupby(lambda col: col.split("_lag")[0])
            .sum()
        )

        return (
                grouped_importance
                .sort_values(ascending=False)
                .head(top_k)
                .index
                .tolist()
          )  # Retorna os nomes das top k features
    
    def temporal_elasticnet_selection(self, lags=12):
        X_lagged, y_lagged = self._create_lagged_features(lags)

        X_scaled = StandardScaler().fit_transform(X_lagged)

        model = ElasticNet(
            alpha=0.05,
            l1_ratio=0.5,
            random_state=42
        )

        model.fit(X_scaled, y_lagged)

        coefs = pd.Series(model.coef_, index=X_lagged.columns)

        # Agrupar coeficientes por feature original
        grouped = coefs.abs().groupby(
            lambda x: x.split("_lag")[0]
            ).sum()

        return grouped[grouped > 0].index.tolist()
    
    def temporal_consensus_selection(self, lags=12, min_votes=2, top_n=None):
        
        methods = {
            "mi": self.temporal_mi_selection(lags=lags, top_k=30),
            "xgb": self.temporal_xgb_selection(lags=lags, top_k=20),
            "enet": self.temporal_elasticnet_selection(lags=lags),
        }

        votes = {}

        for feature_list in methods.values():
            for f in feature_list:
                votes[f] = votes.get(f, 0) + 1

        votes_ordered = dict(
            sorted(
                votes.items(),
                key=lambda x: x[1],
                reverse=True
            )
        )        

        filtered = [
            f for f, count in votes_ordered.items() 
            if count >= min_votes
        ]

        if top_n is not None:
            filtered = filtered[:top_n]

        return filtered, votes_ordered
    
    def print_features(self, features: List[str], votes: Dict[str, int]):
        print(f"Features selecionadas ({len(features)}):")
        for f in features:
            print(f"{f} - {votes.get(f, 0)} votos")

# ==========================================
# 2) FUNÇÃO PRINCIPAL: make_pipeline(conf)
# ==========================================

def make_pipeline(conf: Dict[str, Any]) -> Pipeline:
    """
    Constrói e retorna um sklearn.Pipeline para feature engineering + modelagem.

    Parâmetros (conf):
    ---------------
    date_col: str
        Nome da coluna de data no dataset de entrada.
    target_col: str
        Nome da coluna alvo (ex.: 'demand_kg').
    categorical_cols: List[str]
        Colunas categóricas para One-Hot (ex.: ['regiao','canal']).
    numeric_cols: List[str]
        Colunas numéricas "brutas" (que já existem antes das features de tempo/lag).
    lags: List[int] | tuple[int]
        Lags da variável alvo (ex.: (1,7,28)).
    rolling_windows: List[int]
        Janelas p/ estatísticas móveis do alvo (ex.: [7,14,28]).
    exogenous: List[Dict]
        Lista de dicionários com dataframes a mesclar e chaves, ex.:
        [
          {'df': sazonal_df, 'on': ['Date']},
          {'df': tarifas_df, 'on': ['Date']}
        ]
    feature_selection= True,
    model_params: Dict[str, Any]
        Parâmetros do HistGradientBoostingRegressor (opcional).
    imputer_num_strategy: str
        Estratégia do imputador numérico ('median', 'mean'). Default 'median'.
    scale_numeric: bool
        Se True, aplica StandardScaler em numéricos após imputação.

    Retorno:
    -------
    sklearn.pipeline.Pipeline
        Pipeline pronto para .fit(X, y) e .predict(X).

    Observações:
    -----------
    - Para séries temporais, faça validação com TimeSeriesSplit e NUNCA embaralhe.
    - Os lags/rollings usam shift(1) para evitar vazamento.
    - O merge exógeno assume que 'Date' (ou a(s) chave(s) em conf['exogenous'][i]['on'])
      existe em X e nos DataFrames exógenos.
    """
    date_col: str      = conf["date_col"]
    target_col: float    = conf["target_col"]
    #cat_cols: List[str] = conf.get("categorical_cols", [])
    #base_num_cols: List[str] = conf.get("numeric_cols", [])
    lags = conf.get("lags", (1, 7, 28))
    rolling_windows: List[int] = conf.get("rolling_windows", [7, 14, 28])
    exogenous = List[Dict] = conf.get("df", ),
    model_params = conf.get("model_params", {})
    #imputer_num_strategy = conf.get("imputer_num_strategy", "median")
    #scale_numeric = conf.get("scale_numeric", True)

    # 2.1) Bloco de engenharia de features "determinísticos"
    fe_steps = [
        ("time", TimeFeaturesTransformer(date_col=date_col)),
        #("lags", LagFeatures(target_col=target_col, lags=lags, date_col=date_col)),
        #("rolling", RollingStats(target_col=target_col, windows=rolling_windows, date_col=date_col)),
        ("fe_selection", FeatureSelection()),
    ]
    fe_pipe = Pipeline(steps=fe_steps)

    # 2.2) Bloco de pré-processamento por tipo de coluna
    # Seleciona numéricos dinamicamente (dtypes) + acrescenta 'base_num_cols' explicitamente
    # Após a etapa de FE, teremos novos numéricos (lags/rolling/month_sin/cos), então
    # usamos um ColumnTransformer com seletores combinados.

    # numeric_transformers = [("imputer", SimpleImputer(strategy=imputer_num_strategy))]
    # if scale_numeric:
    #     numeric_transformers.append(("scaler", StandardScaler()))
    # num_pipeline = Pipeline(steps=numeric_transformers)

    # cat_pipeline = Pipeline(steps=[
    #     ("imputer", SimpleImputer(strategy="most_frequent")),
    #     ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    # ])

    # pre = ColumnTransformer(
    #     transformers=[
    #         # numéricos dinâmicos por dtype + garantimos base_num_cols
    #         ("num", num_pipeline, selector(dtype_include=np.number)),
    #         ("cat", cat_pipeline, cat_cols),
    #     ],
    #     remainder="drop",
    #     verbose_feature_names_out=False,
    # )

    # 2.3) Modelo (padrão: HistGradientBoostingRegressor)
    #model = HistGradientBoostingRegressor(**model_params)

    # 2.4) Pipeline final
    pipe = Pipeline(steps=[
        ("fe", fe_pipe),     # engenharia de features
        #("pre", pre),        # pré-processamento (imput/scaler/one-hot)
        #("model", model),    # regressão
    ])

    return pipe


# ===========================
# 3) EXEMPLO DE USO / CONF
# ===========================

@dataclass
class ExampleConfig:
    date_col: str = "Date"
    target_col: str = "demand_kg"
    #categorical_cols: List[str] = None
    #numeric_cols: List[str] = None
    #lags: tuple = (1, 7, 28)
    #rolling_windows: List[int] = None
    #exogenous: List[Dict[str, Any]] = None
    #model_params: Dict[str, Any] = None
    feature_selection=True,
    #imputer_num_strategy: str = "median"
    #scale_numeric: bool = True


def example_build():
    # Suponha que você já tem DF principal (X) com Date, demand_kg, etc.
    # E dois DataFrames exógenos carregados: sazonal_df, tarifas_df
    # (por exemplo, os CSV que produzimos no projeto)
    sazonal_df = pd.read_csv("sazonal_consumo_carne_bovina_BR_2020_2025_diario.csv", parse_dates=["Date"])
    tarifas_df = pd.read_csv("trump_tariffs_brazil_daily_2020_2025.csv", parse_dates=["Date"])

    conf = dict(
        date_col="Date",
        target_col="demand_kg",
        #categorical_cols=["regiao", "canal"],     # se existirem
        #numeric_cols=["preco_kg", "temp_media"],  # se existirem no seu X
        lags=(1, 7, 14, 28),
        rolling_windows=[7, 14, 28],
        exogenous=[
            {"df": sazonal_df, "on": ["Date"]},
            {"df": tarifas_df, "on": ["Date"]},
        ],
        model_params={"max_depth": 6, "learning_rate": 0.08, "max_iter": 500},
        #imputer_num_strategy="median",
        #scale_numeric=True,
    )

    pipe = make_pipeline(conf)
    return pipe
    