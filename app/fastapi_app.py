# fastapi_app.py
# ------------------------------------------------------------
# API FastAPI para previsão one-step do boi_negociado via LSTM
# ------------------------------------------------------------
# ✔ Carrega modelo LSTM no startup usando Lifespan
# ✔ Carrega scalers individuais por feature
# ✔ Recebe 1 linha de cenário futur0 e devolve Y(t+1)
# ✔ Simples, limpo e produção-ready
# ------------------------------------------------------------

from pathlib import Path
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import json

from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from data_pipeline.dags.normalize_pipeline import normalize_time_series, PipelineConfig

# ---------------------- CONFIGURAÇÕES ------------------------

path = Path(os.path.abspath(__file__))
root_dir = path.parent.parent
model_path = os.path.join(root_dir, 'ml-pipeline' ,'models', "model_lstm.h5")
operators_path = os.path.join(root_dir, 'data_pipeline' ,'operators')

# ordem exata usada no treinamento
FEATURE_COLS = [
    "boi_futuro", "boi_dolar", "soja_real", "soja_dolar", "soja_futuro",
    "milho_futuro", "milho_real", "milho_dolar", "precip_total_mm",
    "selic", "cme", "week"
]

TARGET_COL = "boi_negociado"

# objetos globais carregados no startup
model = None
scalers = {}


# ----------------------- LIFESPAN APP ------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Substitui @app.on_event("startup").
    Tudo antes do yield ocorre no startup.
    Tudo após o yield ocorre no shutdown.
    """
    global model

    print("\n🚀 Inicializando API e carregando recursos...\n")

    # --- Carrega modelo ---
    try:
        print("🔄 Carregando modelo LSTM...")
        model = tf.keras.models.load_model(model_path)
        print("✅ Modelo carregado!")
    except Exception as e:
        print(f"❌ Erro carregando modelo: {e}")
        raise e

    # API está pronta
    print("\n🔥 API pronta para uso!\n")
    yield

    # --- SHUTDOWN (opcional) ---
    print("\n🧹 Encerrando API... limpeza final.\n")


app = FastAPI(
    title="API - Previsão Boi Negociado (One-Step)",
    lifespan=lifespan
)

# ---------------------- MODELOS DE REQUEST ---------------------

class PredictRequest(BaseModel):
    date: str            # data futura desejada
    scenario: dict       # ex: {"boi_futuro": 310.2, "soja_real": 145.3, ...}


# -------------------- FUNÇÕES AUXILIARES ----------------------

def scale_input_row(df_row: pd.DataFrame) -> np.ndarray:
    """
    Aplica o scaler individual de cada coluna e retorna:
        shape = (1, 1, n_features)
    Pois é um one-step LSTM sem janela histórica.
    """
    config = PipelineConfig()
    df_out, _ = normalize_time_series(isTrain=False, df=df_row, cols=FEATURE_COLS, config=config, artifact_dir=operators_path)

    X =df_out.values.reshape(1, 1, -1)
    return X

# ------------------------- ENDPOINT ---------------------------

@app.post("/predict")
def predict(req: PredictRequest):
    """
    Realiza a previsão one-step usando somente o cenário informado.
    Modelo espera timesteps=1 e n_features conforme treino.
    """
    try:
        # verifica features obrigatórias
        missing = [c for c in FEATURE_COLS if c not in req.scenario]
        if missing:
            raise HTTPException(status_code=400,
                                detail=f"Features ausentes no cenário: {missing}")

        # cria DataFrame com 1 linha
        row = pd.DataFrame([{col: req.scenario[col] for col in FEATURE_COLS}])

        # aplica normalização (um scaler por coluna)
        X = scale_input_row(row)

        # previsão
        y_pred = model.predict(X, verbose=0)
        y_pred_value = float(y_pred[0][0])

        return {
            "date": req.date,
            "prediction": y_pred_value
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

