# fastapi_app.py
"""
FastAPI para previsão por cenário **pontual** (data-alvo + um valor por exógena).

- Carrega o modelo treinado via MLflow (`MODEL_URI`).
- **Normaliza na inferência** usando o mesmo código de normalização do projeto
  (`data_pipelome/dags/normalize_pipeline.py`).
- Suporta dois modos de normalização (defina `NORMALIZE_MODE`):
    1) `scaler_only` (padrão): aplica **somente** os artefatos persistidos (PowerTransformer + MinMaxScaler)
       ao ponto informado (sem diferenciar/ADF). Útil quando só há **um ponto** e **sem histórico**.
    2) `series`: requer um `history` opcional no payload. Concatena histórico + ponto e chama
       `normalize_time_series(isTrain=False, ...)` para gerar as colunas `*_norm` e usa a **última linha**.

OBS IMPORTANTES
---------------
- O módulo `normalize_pipeline.py` (fornecido por você) define `transformer_dir`/`scale_dir` relativos ao
  caminho do próprio arquivo (../operators). Em produção, garanta que **os mesmos joblibs** gerados no treino
  estejam acessíveis nesse diretório (ex.: `data_pipelome/operators/`).
- Se você treinou o modelo **somente** com features `*_norm`, a inferência **precisa** produzir as mesmas colunas.
  Este app cuida disso nos dois modos acima.
"""

from __future__ import annotations

import importlib
import os
from typing import Dict, Optional, List

import joblib
import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---- Config via ENV ---------------------------------------------------
MODEL_URI = os.getenv('MODEL_URI')  # e.g., 'runs:/<run_id>/model'
DATE_COL = os.getenv('DATE_COL', 'Date')
TARGET_COL = os.getenv('TARGET_COL', 'preco_arroba')
EXOG_COLS = [c.strip() for c in os.getenv('EXOG_COLS', 'selic,ipca,cambio').split(',') if c.strip()]
CATEG_COLS = [c.strip() for c in os.getenv('CATEG_COLS', '').split(',') if c.strip()]
NORM_COLS = [c.strip() for c in os.getenv('NORM_COLS', ','.join(EXOG_COLS)).split(',') if c.strip()]

# data_pipelome/dags is where normalize_pipeline.py lives
NORMALIZE_MODULE = os.getenv('NORMALIZE_MODULE', 'data_pipelome.dags.normalize_pipeline')
NORMALIZE_MODE = os.getenv('NORMALIZE_MODE', 'scaler_only')  # 'scaler_only' | 'series'

# ---- API App ----------------------------------------------------------
app = FastAPI(title="Boi Gordo - Scenario Forecast API (pontual)")
model = None
normalize_mod = None
pt = None
scaler = None


# ---- Pydantic Schemas -------------------------------------------------
class HistoryPoint(BaseModel):
    Date: str
    # As exógenas dinâmicas: aceitamos valores genéricos e validamos depois
    # Ex.: {"Date": "2026-02-10", "selic": 10.5, "ipca": 0.35, ...}
    # Campos adicionais serão mesclados no DataFrame.

class ScenarioRequest(BaseModel):
    date_value: str = Field(..., example="2026-03-10")
    exog_values: Dict[str, float]
    categorical_values: Optional[Dict[str, str]] = None
    history: Optional[List[Dict[str, float]]] = None  # usado se NORMALIZE_MODE='series'


# ---- Helpers ----------------------------------------------------------
def _import_normalize_module():
    global normalize_mod
    try:
        normalize_mod = importlib.import_module(NORMALIZE_MODULE)
    except Exception as e:
        raise RuntimeError(f"Falha ao importar módulo de normalização '{NORMALIZE_MODULE}': {e}")


def _load_joblibs_if_available():
    """Carrega artefatos persistidos pelo normalize_pipeline (se existirem).

    O módulo define caminhos relativos: root_dir/operators/power_transformer.joblib e scaler.joblib.
    """
    global pt, scaler
    try:
        transformer_dir = getattr(normalize_mod, 'transformer_dir', None)
        scale_dir = getattr(normalize_mod, 'scale_dir', None)
        if transformer_dir and os.path.exists(transformer_dir):
            pt = joblib.load(transformer_dir)
        if scale_dir and os.path.exists(scale_dir):
            scaler = joblib.load(scale_dir)
    except Exception as e:
        # Não interrompe startup; apenas registra que não foi possível
        print(f"[WARN] Não foi possível carregar joblibs de normalização: {e}")
        pt, scaler = None, None


def _make_row(date_value: str, exog_values: Dict[str, float], categorical_values: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    row: Dict[str, object] = {DATE_COL: pd.to_datetime(date_value)}
    for c in EXOG_COLS:
        if c not in exog_values:
            raise KeyError(f"Exógena obrigatória ausente: '{c}'")
        row[c] = float(exog_values[c])
    for c in CATEG_COLS:
        if not categorical_values or c not in categorical_values:
            raise KeyError(f"Categórica obrigatória ausente: '{c}'")
        row[c] = categorical_values[c]
    return pd.DataFrame([row])


def _normalize_point_scaler_only(df_row: pd.DataFrame) -> pd.DataFrame:
    """Cria colunas *_norm para NORM_COLS usando os joblibs carregados.

    - Se 'pt' (PowerTransformer) estiver disponível, aplica **pt -> scaler**.
    - Se apenas 'scaler' estiver disponível, aplica **scaler** direto.
    - Se nenhum estiver disponível, lança erro (modelo foi treinado com normalização externa).
    """
    if scaler is None and pt is None:
        raise RuntimeError("Artefatos de normalização não encontrados. Certifique-se de que os joblibs foram gerados no treino e estão em data_pipelome/operators/.")

    out = df_row.copy()
    for c in NORM_COLS:
        if c not in out.columns:
            raise KeyError(f"Coluna para normalização não encontrada: '{c}'")
        x = np.asarray(out[c]).reshape(-1, 1).astype(float)
        x_t = x
        if pt is not None:
            try:
                x_t = pt.transform(x_t)
            except Exception as e:
                # Se pt não for compatível com dado pontual, degrade para apenas scaler
                print(f"[WARN] PowerTransformer.transform falhou em '{c}': {e}. Usando apenas scaler.")
                x_t = x
        if scaler is not None:
            x_t = scaler.transform(x_t)
        out[f"{c}_norm"] = x_t.ravel()
    return out


def _normalize_series(df_hist: pd.DataFrame, df_row: pd.DataFrame) -> pd.DataFrame:
    """Concatena histórico + ponto e chama normalize_time_series(isTrain=False).
    Retorna apenas a **última linha** com colunas *_norm.
    """
    if normalize_mod is None or not hasattr(normalize_mod, 'normalize_time_series'):
        raise RuntimeError("normalize_time_series não disponível no módulo importado.")

    df_cat = pd.concat([df_hist, df_row], ignore_index=True)
    df_norm, _ = normalize_mod.normalize_time_series(
        isTrain=False,
        df=df_cat,
        cols=NORM_COLS,
        config=None,
    )
    last = df_norm.tail(1).copy()
    # exige que *_norm existam
    for c in NORM_COLS:
        if f"{c}_norm" not in last.columns:
            raise RuntimeError(f"Coluna normalizada ausente após normalize_time_series: {c}_norm")
    return last


# ---- FastAPI lifecycle ------------------------------------------------
@app.on_event('startup')
def _startup():
    global model
    if not MODEL_URI:
        raise RuntimeError("Env MODEL_URI não definido. Ex.: runs:/<run_id>/model")
    # Carrega modelo do MLflow
    model = mlflow.sklearn.load_model(MODEL_URI)

    # Importa módulo de normalização e carrega joblibs (se existirem)
    _import_normalize_module()
    _load_joblibs_if_available()


# ---- Endpoint ---------------------------------------------------------
@app.post('/predict')
def predict(req: ScenarioRequest):
    try:
        # 1) monta a linha do cenário
        row = _make_row(req.date_value, req.exog_values, req.categorical_values)

        # 2) normaliza conforme modo
        if NORMALIZE_MODE == 'series':
            history = req.history or []
            if not history:
                raise HTTPException(status_code=400, detail="Modo 'series' requer 'history' no payload.")
            # valida que cada ponto do history tem Date + exógenas
            hist_rows = []
            for h in history:
                if 'Date' not in h:
                    raise HTTPException(status_code=400, detail="Cada item de 'history' deve conter 'Date'.")
                r = {DATE_COL: pd.to_datetime(h.get('Date'))}
                for c in EXOG_COLS:
                    if c not in h:
                        raise HTTPException(status_code=400, detail=f"History sem exógena obrigatória: '{c}'")
                    r[c] = float(h[c])
                for c in CATEG_COLS:
                    if c not in h:
                        raise HTTPException(status_code=400, detail=f"History sem categórica obrigatória: '{c}'")
                    r[c] = h[c]
                hist_rows.append(r)
            df_hist = pd.DataFrame(hist_rows).sort_values(DATE_COL)
            X = _normalize_series(df_hist, row)
        else:
            # scaler_only: usa joblibs carregados (pt/scaler) sem ADF/diff
            X = _normalize_point_scaler_only(row)

        # 3) seleciona as features na mesma ordem usada no treino
        feat_cols = [f"{c}_norm" for c in NORM_COLS]
        missing = [c for c in feat_cols if c not in X.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Colunas normalizadas ausentes: {missing}")

        y_hat = model.predict(X[feat_cols].values)
        return {"prediction": float(np.asarray(y_hat).reshape(-1)[0])}

    except HTTPException:
        raise
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro inesperado: {e}")
