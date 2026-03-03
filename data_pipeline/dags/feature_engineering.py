# feature_engineering.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.feature_selection import RFE
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

    def select_features(self):
        model_lr = LinearRegression()
        rfe = RFE(estimator=model_lr, n_features_to_select=10)
        rfe.fit(self.X, self.y)

        selected_rfe = [self.feature_names[i] for i in range(self.X.shape[1]) if rfe.support_[i]]

        return selected_rfe

    # 3. Aplicar Lasso para seleção de atributos
    def lasso_selection(self):
        model_lasso = Lasso(alpha=0.4)
        model_lasso.fit(self.X, self.y)

        selected_lasso = [self.feature_names[i] for i, coef in enumerate(model_lasso.coef_) if coef != 0]

        return selected_lasso

    # Exibir resultados
    def display_selected_features(self):
        selected_rfe = self.select_features()
        selected_lasso = self.lasso_selection()

        print("=== Seleção de Atributos com RFE ===")
        print(f"Features selecionadas pelo RFE: {selected_rfe}\n")

        print("=== Seleção de Atributos com Lasso ===")
        print(f"Features selecionadas pelo Lasso: {selected_lasso}\n")

        return selected_rfe, selected_lasso


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
    