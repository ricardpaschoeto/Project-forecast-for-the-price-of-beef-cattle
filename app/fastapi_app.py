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
import sys
# Caminho absoluto da raiz do projeto
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
import tensorflow as tf

# ---------------------- CONFIGURAÇÕES ------------------------

path = Path(os.path.abspath(__file__))
root_dir = path.parent.parent
model_path = os.path.join(root_dir, 'ml-pipeline' ,'models', "model_lstm.h5")
operators_path = os.path.join(root_dir, 'data_pipeline' ,'operators')


hist_path = os.path.join(root_dir, 'data_pipeline', 'sensors', 'dados_modelo_rfe.csv')
df_hist = pd.read_csv(hist_path, index_col=0, dayfirst=True)
df_hist.index = pd.to_datetime(df_hist.index, dayfirst=True)


# ordem exata usada no treinamento
FEATURE_COLS = [
    "Taxa_EUA_norm","milho_dolar_norm","dolar_norm","CovidPeriodFlag_norm","festas_juninas_flag_norm","sao_joao_flag_norm","sao_pedro_flag_norm","finados_weekend_flag_norm","ipca_norm","el_nino_encoded_norm"
]

TARGET_COL = "boi_negociado"

# objetos globais carregados no startup
model = None

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
        model = tf.keras.models.load_model(model_path, compile=False)
        print(model.input_shape)
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

# ------------------------- ENDPOINT ---------------------------

@app.post("/predict")
def predict(req: PredictRequest):
    """
    Realiza a previsão one-step usando somente o cenário informado.
    Modelo espera timesteps=1 e n_features conforme treino.
    """
    try:
        
        # 1. Construir janela de 60 steps
        scenario_df = pd.DataFrame(req.scenario)
        #window = np.array(req.scenario)

        # aplica normalização (um scaler por coluna)
        X = scenario_df.values.reshape(1, 60, len(FEATURE_COLS))  # shape (1, 60, 10)

        # previsão
        y_pred = model.predict(X)
        y_pred_value = float(y_pred[0][0])

        return {
            "date": req.date,
            "prediction": y_pred_value
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

