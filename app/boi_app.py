# app.py — versão profissional com:
# ✔ Previsão na SIDEBAR
# ✔ Abas: Backtest | Cenários | Dashboard
# ✔ Trading Terminal styling (dark)
# ✔ IC 95%
# ✔ Exportação CSV e PDF
# ✔ Cenários múltiplos (valores absolutos)
# ✔ Backtest rolling-origin
# -------------------------------------------------------------------------

from pathlib import Path
import os
import sys

# Caminho absoluto da raiz do projeto
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from typing import List, Dict
import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, date
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from datetime import date
from pydantic import BaseModel
import tensorflow as tf

import time
from data_pipeline.dags.normalize_pipeline import normalize_time_series, PipelineConfig, inverse_transform_column

# -------------------------------------------------------------------------
# CONFIGURAÇÕES
# -------------------------------------------------------------------------

API_URL = "http://localhost:8000/predict"

DEBOUNCE_SECONDS = 1.0  # intervalo mínimo entre chamadas

COLS_FLAGS = ['CovidPeriodFlag', 'festas_juninas_flag', 'sao_joao_flag', 'sao_pedro_flag', 'finados_weekend_flag', 'el_nino_encoded']
FEATURE_COLS = ['dolar', 'milho_dolar', 'Taxa_EUA', 'ipca']

path = Path(os.path.abspath(__file__))
root_dir = path.parent.parent
operators_path = os.path.join(root_dir, 'data_pipeline' ,'operators')
df_path = os.path.join(root_dir,'data_pipeline', 'sensors', 'dados_modelo_rfe.csv')
model_path = os.path.join(root_dir, 'ml-pipeline' ,'models', "model_lstm.h5")

# ========================
def _load_history():
    path = Path(os.path.abspath(__file__))
    root_dir = path.parent.parent
    df_path = os.path.join(root_dir,'data_pipeline', 'sensors', 'dados_modelo_rfe.csv')
    df = pd.read_csv(df_path, index_col=0, dayfirst=True)
    df.index = pd.to_datetime(df.index, dayfirst=True)
    df.drop(columns=["boi_negociado"], inplace=True)
    return df   

def load_scenarios(dates: list[str]):
    scenario = {}
    list_scenarios = []
    for date in dates:
        scenario[date] = _load_history().loc[date].to_dict()
        list_scenarios.append(scenario[date])

    return list_scenarios
# ========================

dates = ["03/01/2020", "29/03/2020", "17/07/2020", "20/07/2025", "11/08/2025"] # 196.7, 200.5, 222.55, 215.3, 227.2
scenarios = load_scenarios(dates)

# Inicializa estado
if "last_call" not in st.session_state:
    st.session_state.last_call = 0

def call_api_with_debounce(payload):
    current_time = time.time()
    if current_time - st.session_state.last_call < DEBOUNCE_SECONDS:
        return None  # bloqueia chamadas repetidas

    st.session_state.last_call = current_time  # atualiza o relógio

    # Agora é seguro chamar a API
    response = requests.post(API_URL, json=payload)
    return response
class PredictRequest(BaseModel):
    date: str            # data futura desejada
    scenario: List[Dict[str, float]]     # ex: {"boi_futuro": 310.2, "soja_real": 145.3, ...}

# ======================================================================
# CSS avançado
# ======================================================================

layout_css = """
<style>
/* Remoção de espaços brancos excessivos e scroll gaps */
.block-container {
    padding: 1.5rem 2rem 1rem 2rem !important;
    max-width: 100% !important;
}
/* Cards profissionais Dark */
.cowboy-card {
    background: #161A25;
    border: 1px solid #2B3040;
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 15px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
}
/* Compactar inputs do Sidebar (reduzir padding) */
div[data-testid="stSidebar"] {
    background-color: #0E1117 !important;
}
.css-1d391kg { padding-top: 1rem !important; }
div[data-testid="stMetricValue"] { font-size: 1.5rem !important; }

/* Customização para ocultar ou afinar barra de scroll global */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #0E1117;
}
::-webkit-scrollbar-thumb {
    background: #3B4252;
    border-radius: 3px;
}
</style>
"""
st.markdown(layout_css, unsafe_allow_html=True)

# -------------------------------------------------------------------------
# FUNÇÕES AUXILIARES
# -------------------------------------------------------------------------

@st.cache_data
def load_history():
    df = pd.read_csv(df_path, index_col=0, dayfirst=True)
    df.index = pd.to_datetime(df.index, dayfirst=True)
    return df

df_hist = load_history()

def _carrega_modelo():
    try:
        print("🔄 Carregando modelo LSTM...")
        model = tf.keras.models.load_model(model_path, compile=False)
        print(model.input_shape)
        print("✅ Modelo carregado!")
        return model
    except Exception as e:
        print(f"❌ Erro carregando modelo: {e}")
        raise e

model = _carrega_modelo()

def compute_sigma(df):
    """Desvio padrão dos retornos — usado para IC 95%."""
    diffs = df["boi_negociado"].diff().dropna()
    return diffs.std()


def plot_forecast(fig, ax, df_hist, data_prev, y_pred, ic95):
    """Gráfico histórico + previsão + IC95 + linha tracejada """
    
    #fig.patch.set_facecolor("#0E1117")
    #ax.set_facecolor("#0E1117")

    # Histórico
    ax.plot(df_hist.index,
            df_hist,
            color="#4EA8DE",
            label="Histórico",
            linewidth=2)

    # Ponto previsto
    forecast_date = pd.to_datetime(data_prev,format="mixed",dayfirst=True)
    ax.scatter(forecast_date,
               y_pred,
               color="#E63946",
               s=120,
               label="Previsão")

    # Linha tracejada do último valor até a previsão
    last_date = df_hist.index.max()
    last_value = df_hist.iloc[-1]

    ax.plot([last_date, forecast_date],
            [last_value, y_pred],
            linestyle="--",
            color="#E63946",
            linewidth=2,
            label="Tendência")

    # IC 95%
    ax.errorbar(forecast_date,
                y_pred,
                yerr=ic95,
                fmt="o",
                color="gray",
                capsize=8,
                label="IC 95%")

    return fig, ax

def gerar_pdf(preco_atual, y_pred, data_prev, tendencia, fig):
    """Gera um PDF completo contendo previsão + gráfico"""

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    w, h = A4

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(40, h - 40, "Relatório de Previsão — Boi Negociado")

    pdf.setFont("Helvetica", 11)
    pdf.drawString(40, h - 70, f"Gerado em: {datetime.now():%d/%m/%Y %H:%M}")

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, h - 110, "Resumo")

    pdf.setFont("Helvetica", 10)
    y = h - 140
    pdf.drawString(40, y, f"- Preço atual: {preco_atual:.2f}")
    y -= 15
    pdf.drawString(40, y, f"- Previsão ({data_prev}): {y_pred:.2f}")
    y -= 15
    pdf.drawString(40, y, f"- Tendência: {tendencia}")

    y -= 250
    buf_fig = io.BytesIO()
    fig.savefig(buf_fig, format="png")
    buf_fig.seek(0)
    pdf.drawImage(ImageReader(buf_fig), 40, y, width=500, preserveAspectRatio=True)

    pdf.save()
    buffer.seek(0)
    return buffer

def backtest(df, steps=200):
    """Rolling-origin cross-validation via API (one-step)."""
    rows = []

    for i in range(len(df) - steps, len(df)):
        row = df.iloc[i - 1]
        true_val = df.iloc[i]["boi_negociado"]

        scenario = {col: row[col] for col in FEATURE_COLS}
        payload = {
            "date": str(df.iloc[i]["data"].date()),
            "scenario": scenario
        }

        try:
            res = requests.post(API_URL, json=payload)
            if res.status_code != 200:
                continue

            pred = float(res.json()["prediction"])
            rows.append({
                "data": df.iloc[i]["data"],
                "y_true": true_val,
                "y_pred": pred,
                "erro": pred - true_val
            })

        except:
            continue

    return pd.DataFrame(rows)

# RSI calculation
def compute_rsi(series, window):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window).mean()
    avg_loss = pd.Series(loss).rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# MACD calculation
def compute_macd(series, fast, slow):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow

# Volatility calculation
def compute_volatility(series, window):
    return series.pct_change().rolling(window).std() * np.sqrt(252)

def scale_input_row(df_window: pd.DataFrame) -> np.ndarray:
    """
    Aplica o scaler individual de cada coluna e retorna:
        shape = (1, window_size, n_features)
    Pois é um one-step LSTM sem janela histórica.
    """
    config = PipelineConfig()
    df_out, _ = normalize_time_series(isTrain=False, 
                                      df=df_window, 
                                      cols=df_window.columns.to_list(),
                                      cols_flags=COLS_FLAGS,
                                      config=config, 
                                      artifact_dir=operators_path)

    return df_out


def build_lstm_window(future_row: pd.DataFrame) -> pd.DataFrame:
    """
    Monta a janela final de (60, 10):
    - 59 últimos passos reais do dataset
    - 1 linha de cenário futuro vinda do payload
    """
    # pega os últimos 59 steps da série histórica
    df_tail = df_hist[FEATURE_COLS + COLS_FLAGS].tail(59)

    # garante mesmas colunas e mesma ordem
    future_row = future_row.reindex(columns=FEATURE_COLS + COLS_FLAGS, fill_value=0)

    # concatena o step futuro
    df_window = pd.concat([df_tail, future_row], ignore_index=True)

    if df_window.shape != (60, len(FEATURE_COLS) + len(COLS_FLAGS)):
        raise ValueError(f"Janela incorreta: esperado (60,{len(FEATURE_COLS) + len(COLS_FLAGS)}), obtido {df_window.shape}")

    return df_window

def _test_create_predict_data(scenario):
    scenario_df = pd.DataFrame([scenario])
    df_window = build_lstm_window(scenario_df)
    df_out = scale_input_row(df_window)
    df_norm = df_out.filter(regex='_norm$')
    scenario_dict = df_norm.to_dict(orient="records")

    scenario_df = pd.DataFrame(scenario_dict)
    return scenario_df,scenario_dict

def _predict(req: PredictRequest):
    """
    Realiza a previsão one-step usando somente o cenário informado.
    Modelo espera timesteps=1 e n_features conforme treino.
    """
 
    try:
        
        # 1. Construir janela de 60 steps
        scenario_df = pd.DataFrame(req.scenario)

        # remove sufixo _norm das colunas
        scenario_df = scenario_df.rename(
            columns=lambda c: c.replace("_norm", "")
        )

        future_row = scenario_df.reindex(columns=df_hist.columns.drop("boi_negociado"), fill_value=0)
        # TODO: Normalizar toda a JANELA
        # window = build_lstm_window(future_row)

        # aplica normalização (um scaler por coluna)
        X = future_row.values.reshape(1, 60, len(FEATURE_COLS ) + len(COLS_FLAGS))  # shape (1, 60, 10)

        # previsão
        y_pred = model.predict(X)
        y_pred_value = float(y_pred[0][0])

        return {
            "date": req.date,
            "prediction": y_pred_value
        }
    
    except Exception as e:
        print(f"Erro na previsão: {e}")
        raise e

# =========================================================================
# SIDEBAR — PREVISÃO ONE-STEP
# =========================================================================

with st.sidebar:

    st.markdown("### 🧩 Configure os valores futuros e gere uma previsão.")

    with st.form("forecast_form"):
        # Grid responsivo para inputs
        st.markdown('<div class="sidebar-grid">', unsafe_allow_html=True)

        # Data
        selected_date = st.date_input("📅 Data da Previsão", value=date.today(), min_value=date.today())

        scenario = {}
        ultima_data = df_hist.index.max()
        st.session_state['ultima_data'] = ultima_data
        
        scenario["Taxa_EUA"] = st.number_input("Taxa Exportação EUA (%)", value=float(df_hist.loc[ultima_data, 'Taxa_EUA']), format="%.4f", help="Taxa de exportação total aplicada a carne bovina pelos EUA ao Brasil.")
        scenario["milho_dolar"] = st.number_input("Preço Saca Milho ($)", value=float(df_hist.loc[ultima_data, 'milho_dolar']), format="%.4f", help="Preço da saca de milho (60kg) convertido para dólares americanos.")
        scenario["dolar"] = st.number_input("Dólar EUA ($)", value=float(df_hist.loc[ultima_data, 'dolar']), format="%.4f", help="Cotação da moeda americana.")
        scenario["CovidPeriodFlag"] = st.number_input("COVID", value=int(df_hist.loc[ultima_data, 'CovidPeriodFlag']), format="%d", min_value=0, max_value=1, step=1,help="Variável exógena COVID com valores 1 ou 0 -  Com PANDEMIA e SEM PANDEMIA, respectivamente.")
        scenario["festas_juninas_flag"] = st.number_input("Festas juninas", value=int(df_hist.loc[ultima_data, 'festas_juninas_flag']), format="%d", min_value=0, max_value=1, step=1,help="Aplica-se 0 ou 1 em caso da data escolhida estar dentro da janela de festas juninas.")
        scenario["sao_joao_flag"] = st.number_input("Festa de São João", value=int(df_hist.loc[ultima_data, 'sao_joao_flag']), format="%d", min_value=0, max_value=1, step=1,help="Aplica-se 0 ou 1 em caso da data escolhida estar dentro da janela de festas de São João.")
        scenario["sao_pedro_flag"] = st.number_input("Festa de São Pedro", value=int(df_hist.loc[ultima_data, 'sao_pedro_flag']), format="%d", min_value=0, max_value=1, step=1, help="Aplica-se 0 ou 1 em caso da data escolhida estar dentro da janela de festas de São pedro.")
        scenario["finados_weekend_flag"] = st.number_input("Feriado de finados", value=int(df_hist.loc[ultima_data, 'finados_weekend_flag']), format="%d", min_value=0, max_value=1, step=1, help="Aplica-se 0 ou 1 em caso da data escolhida estar dentro da janela do feriado de Finados.")
        scenario["ipca"] = st.number_input("ipca (%)", value=float(df_hist.loc[ultima_data, 'ipca']), format="%.4f", help="O IPCA é o principal índice que mede a inflação no Brasil. Ele mostra quanto os preços de produtos e serviços aumentam (ou diminuem) ao longo do tempo.")
        scenario["el_nino_encoded"] = st.number_input("Fenômeno el nino", value=int(df_hist.loc[ultima_data, 'el_nino_encoded']), format="%d", min_value=0, max_value=1, step=1, help="El Niño e La Niña são fenômenos climáticos que acontecem no Oceano Pacífico e que alteram a temperatura do mar, causando mudanças importantes no clima do mundo, como secas ou chuvas intensas.")
        
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        run_forecast = st.form_submit_button("🚀 Gerar Previsão")

# =========================================================================
# TABS — BACKTEST | CENÁRIOS | DASHBOARD
# =========================================================================

tab_dashboard, tab_cenarios, tab_backtest   = st.tabs([
    "💹 Trading Dashboard",
    "🧪 Cenários Múltiplos",
    "📊 Backtest"        
])

# -------------------------------------------------------------------------
# TAB: DASHBOARD ESTILO TRADING TERMINAL
# -------------------------------------------------------------------------
with tab_dashboard:

    st.markdown("## 💹 Trading Dashboard — Boi Negociado")

    # Bloomberg-style sliders

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        window_rsi = st.slider("RSI Window", 5, 30, 14)

    with col2:
        window_macd_fast = st.slider("MACD Fast", 5, 20, 12)

    with col3:
        window_macd_slow = st.slider("MACD Slow", 10, 40, 26)

    with col4:
        window_vol = st.slider("Volatility Window", 5, 60, 20)

    # ================================
    # Calcular indicadores técnicos usando os dados históricos
    # ================================
    rsi = compute_rsi(df_hist["boi_negociado"], window_rsi)
    macd = compute_macd(df_hist["boi_negociado"], window_macd_fast, window_macd_slow)
    vol = compute_volatility(df_hist["boi_negociado"], window_vol)

    preco_atual = df_hist["boi_negociado"].iloc[-1]
    ultimos_30 = df_hist["boi_negociado"].tail(30)
    variacao_30d = ((ultimos_30.iloc[-1] - ultimos_30.iloc[0]) /
                     ultimos_30.iloc[0]) * 100

    tendencia = " "
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Preço Atual (R$)", f"{preco_atual:.2f}")
    c2.metric("Variação 30 dias (%)", f"{variacao_30d:.2f}%")


    data_prev = selected_date.strftime("%d/%m/%Y")
    pred = None
    ic95 = None
    st.markdown("### 📈 Gráficos - Histórico, RSI, MACD, Volatilidade")

    cutoff_date = df_hist.index.max() - pd.DateOffset(months=12)
    df_12m = df_hist[df_hist.index >= cutoff_date]

    # ================================
    # FIGURA 1 — Preço 12 meses
    # ================================
    st.markdown('<div class="cowboy-card">', unsafe_allow_html=True)
    grafico_container = st.empty()
    fig1, ax1 = plt.subplots(figsize=(20, 4))

    if "fig1" not in st.session_state:
        st.session_state["fig1"] = None        
        ax1.plot(df_12m.index, df_12m["boi_negociado"], color="#4EA8DE", linewidth=2, label="Preço do Boi")

        # --- Título e labels grandes ---
        ax1.set_title("Histórico — Boi Negociado (Últimos 12 meses)", color="black")
        ax1.set_xlabel("Data", fontsize=16, color="white")
        ax1.set_ylabel("Preço (R$)", fontsize=16, color="white")


        # --- Legenda correta ---
        ax1.legend(facecolor="#1E1E1E", labelcolor="white", fontsize=14)

        # --- Formatação de datas ---
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))   # marca mês a mês
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b/%Y"))   # formato: Jan/2025

        # --- Rotação para aparecer ---
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=12, color="black")

        # --- Estética ---
        ax1.tick_params(colors="black")
        ax1.grid(alpha=0.2)
    
        st.session_state["fig1"] = fig1
        grafico_container.pyplot(st.session_state["fig1"])
    else:
        grafico_container.pyplot(st.session_state["fig1"])

    st.markdown('</div>', unsafe_allow_html=True)

    # ================================
    # FIGURA 1 — Predição One-Step via API
    # ================================
       
    if True: # run_forecast:
        scenario_df, scenario_dict = _test_create_predict_data(scenarios[1])
        req = PredictRequest(date=data_prev, scenario=scenario_dict)
        res = _predict(req)

        # X = scenario_df.values.reshape(1, 60, len(FEATURE_COLS))

        # payload = {"date": str(selected_date), "scenario": scenario_dict}
        # res = call_api_with_debounce(payload)
        
        if res is None:
            st.warning("🔁 Aguarde antes de rodar a previsão novamente...")
            st.stop()
        #if res.status_code != 200:
            #st.error("Erro na API: " + res.text)
            #st.stop()
        else:

            # inversão da normalização para o valor real
            #y_pred_df = pd.DataFrame({'boi_negociado_norm': [float(res.json()["prediction"])]}, index=[pd.to_datetime( res.json()["date"], format="%Y-%m-%d")])
            y_pred_df = pd.DataFrame({'boi_negociado_norm': [float(res["prediction"])]}, index=[pd.to_datetime( res["date"], format="%d/%m/%Y")])

            y_pred_value = inverse_transform_column(
                df_norm = y_pred_df,
                cols= ['boi_negociado'],
                artifact_dir=operators_path,
                start_values=df_hist[['boi_negociado']],
            )

            pred = float(y_pred_value.to_numpy().squeeze())
            #data_prev = res.json()["date"]
            data_prev = res["date"]

            preco_atual = float(df_hist["boi_negociado"].iloc[-1])
            tendencia = "Alta" if pred > preco_atual else "Baixa"

            sigma = compute_sigma(df_hist)
            ic95 = 1.96 * sigma

            st.markdown("### 📌 Indicadores")

            c3.metric("Previsão", f"{pred:.2f}")
            c4.metric("Tendência", tendencia)
            fig, ax = plot_forecast(fig1, ax1, df_12m["boi_negociado"], data_prev, pred, ic95)
            
            # Título e labels
            ax.set_title("Histórico + Previsão com Tendência", color="white", fontsize=20)
            ax.set_xlabel("Data", color="white", fontsize=14)
            ax.set_ylabel("Preço (R$)", color="white", fontsize=14)

            # Formato de data
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b/%Y"))
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", color="black")

            ax.grid(alpha=0.2)
            ax.legend(facecolor="#1E1E1E", labelcolor="white")

            st.session_state["fig1"] = fig

            grafico_container.pyplot(st.session_state["fig1"])
    # ================================
    # FILTRAR ÚLTIMOS 12 MESES
    # ================================

    # Recalcular indicadores somente com os últimos 12 meses
    rsi = compute_rsi(df_12m["boi_negociado"], window_rsi)
    macd = compute_macd(df_12m["boi_negociado"], window_macd_fast, window_macd_slow)
    vol = compute_volatility(df_12m["boi_negociado"], window_vol)

    # ================================
    # FIGURA 2 — RSI (12 meses)
    # ================================
    st.markdown('<div class="cowboy-card">', unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(20, 4))
    ax2.plot(df_12m.index, rsi, color="orange", linewidth=2, label="RSI")

    # Título e labels maiores
    ax2.set_title("RSI (Últimos 12 meses)", fontsize=20, color="black")
    ax2.set_xlabel("Data", fontsize=16, color="white")
    ax2.set_ylabel("RSI", fontsize=16, color="white")

    # Legenda
    ax2.legend(facecolor="#1E1E1E", labelcolor="white", fontsize=14)

    # Formatação do eixo X
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b/%Y"))

    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", fontsize=12, color="black")

    ax2.tick_params(colors="black")
    ax2.grid(alpha=0.2)

    st.pyplot(fig2)
    st.markdown('</div>', unsafe_allow_html=True)

    # ================================
    # FIGURA 3 — MACD (12 meses)
    # ================================
    st.markdown('<div class="cowboy-card">', unsafe_allow_html=True)
    fig3, ax3 = plt.subplots(figsize=(20, 4))
    ax3.plot(df_12m.index, macd, color="cyan", linewidth=2, label="MACD")

    ax3.set_title("MACD (Últimos 12 meses)", fontsize=20, color="black")
    ax3.set_xlabel("Data", fontsize=16, color="white")
    ax3.set_ylabel("MACD", fontsize=16, color="white")

    ax3.legend(facecolor="#1E1E1E", labelcolor="white", fontsize=14)

    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b/%Y"))

    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right", fontsize=12, color="black")

    ax3.tick_params(colors="black")
    ax3.grid(alpha=0.2)

    st.pyplot(fig3)
    st.markdown('</div>', unsafe_allow_html=True)

    # ================================
    # FIGURA 4 — Volatilidade (12 meses)
    # ================================
    st.markdown('<div class="cowboy-card">', unsafe_allow_html=True)
    fig4, ax4 = plt.subplots(figsize=(20, 4))
    ax4.plot(df_12m.index, vol, color="red", linewidth=2, label="Volatilidade")

    ax4.set_title("Volatilidade (Últimos 12 meses)", fontsize=20, color="black")
    ax4.set_xlabel("Data", fontsize=16, color="white")
    ax4.set_ylabel("Volatilidade", fontsize=16, color="white")

    ax4.legend(facecolor="#1E1E1E", labelcolor="white", fontsize=14)

    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%b/%Y"))

    plt.setp(ax4.get_xticklabels(), rotation=45, ha="right", fontsize=12, color="black")

    ax4.tick_params(colors="black")
    ax4.grid(alpha=0.2)

    st.pyplot(fig4)
    st.markdown('</div>', unsafe_allow_html=True)
# -------------------------------------------------------------------------
# EXPORTAR - XLS ou PDF
# -------------------------------------------------------------------------
    st.markdown('<div class="cowboy-card">', unsafe_allow_html=True)
    st.markdown("### 💾 Exportação")

    df_exp = df_hist.copy()

    new_row = {
        "data": pd.to_datetime(data_prev, format="mixed", dayfirst=True)
    }

    for col in FEATURE_COLS:
        new_row[col] = scenario[col]

    if pred is not None:
        new_row["boi_negociado"] = pred
    else:
        new_row["boi_negociado"] = '-'

    # 3. Gera um DataFrame da nova linha com EXACTAMENTE o schema correto
    new_row_df = pd.DataFrame([new_row])

    # Reindexa o df da linha nova para combinar com df_exp
    new_row_df = new_row_df.reindex(columns=df_exp.columns)

    # 4. Concat seguro — AGORA SEM WARNING
    df_exp = pd.concat([df_exp, new_row_df], ignore_index=True)

    csv_bytes = df_exp.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Exportar CSV", csv_bytes, "previsao.csv")

    pdf_bytes = io.BytesIO()
    if pred is not None:
        pdf_bytes = gerar_pdf(preco_atual, pred, data_prev, tendencia, fig1)
        
    st.download_button("📘 Exportar PDF", pdf_bytes, "relatorio_previsao.pdf")

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------------------------------
# TAB: CENÁRIOS MÚLTIPLOS
# -------------------------------------------------------------------------
# data,Taxa_EUA,milho_dolar,dolar,CovidPeriodFlag,festas_juninas_flag,sao_joao_flag,sao_pedro_flag,finados_weekend_flag,ipca,el nino_encoded,boi_negociado
with tab_cenarios:
    st.header("🧪 Simulação de Cenários Múltiplos (Valores Absolutos)")

    num_cenarios = st.slider("Número de cenários", 1, 5, 2)

    cenarios = []
    for i in range(num_cenarios):
        st.subheader(f"Cenário {i+1}")
        cols2 = st.columns(3)
        scenario = {}
        scenario["Taxa_EUA"] = st.number_input(f"Taxa Exportação EUA (R$)  — cenário {i+1}", value=float(df_hist.loc[ultima_data, 'Taxa_EUA']), format="%.4f", help="Taxa de exportação total aplicada a carne bovina pelos EUA ao Brasil.")
        scenario["milho_dolar"] = st.number_input(f"Preço Saca Milho ($)  — cenário {i+1}", value=float(df_hist.loc[ultima_data, 'milho_dolar']), format="%.4f", help="Preço da saca de milho (60kg) convertido para dólares americanos.")
        scenario["dolar"] = st.number_input(f"Dólar EUA ($)  — cenário {i+1}", value=float(df_hist.loc[ultima_data, 'dolar']), format="%.4f", help="Cotação da moeda americana.")
        scenario["CovidPeriodFlag"] = st.number_input(f"COVID  — cenário {i+1}", value=int(df_hist.loc[ultima_data, 'CovidPeriodFlag']), format="%d", min_value=0, max_value=1, step=1,help="Variável exógena COVID com valores 1 ou 0 -  Com PANDEMIA e SEM PANDEMIA, respectivamente.")
        scenario["festas_juninas_flag"] = st.number_input(f"Festas juninas  — cenário {i+1}", value=int(df_hist.loc[ultima_data, 'festas_juninas_flag']), format="%d", min_value=0, max_value=1, step=1,help="Aplica-se 0 ou 1 em caso da data escolhida estar dentro da janela de festas juninas.")
        scenario["sao_joao_flag"] = st.number_input(f"Festa de São João  — cenário {i+1}", value=int(df_hist.loc[ultima_data, 'sao_joao_flag']), format="%d", min_value=0, max_value=1, step=1,help="Aplica-se 0 ou 1 em caso da data escolhida estar dentro da janela de festas de São João.")
        scenario["sao_pedro_flag"] = st.number_input(f"Festa de São Pedro  — cenário {i+1}", value=int(df_hist.loc[ultima_data, 'sao_pedro_flag']), format="%d", min_value=0, max_value=1, step=1, help="Aplica-se 0 ou 1 em caso da data escolhida estar dentro da janela de festas de São pedro.")
        scenario["finados_weekend_flag"] = st.number_input(f"Feriado de finados  — cenário {i+1}", value=int(df_hist.loc[ultima_data, 'finados_weekend_flag']), format="%d", min_value=0, max_value=1, step=1, help="Aplica-se 0 ou 1 em caso da data escolhida estar dentro da janela do feriado de Finados.")
        scenario["ipca"] = st.number_input(f"ipca (%)  — cenário {i+1}", value=float(df_hist.loc[ultima_data, 'ipca']), format="%.4f", help="O IPCA é o principal índice que mede a inflação no Brasil. Ele mostra quanto os preços de produtos e serviços aumentam (ou diminuem) ao longo do tempo.")
        scenario["el_nino_encoded"] = st.number_input(f"Fenômeno el nino  — cenário {i+1}", value=int(df_hist.loc[ultima_data, 'el_nino_encoded']), format="%d", min_value=0, max_value=1, step=1, help="El Niño e La Niña são fenômenos climáticos que acontecem no Oceano Pacífico e que alteram a temperatura do mar, causando mudanças importantes no clima do mundo, como secas ou chuvas intensas.")

        cenarios.append(scenario)

    if st.button("Simular Cenários"):
        resultados = []
        for idx, sc in enumerate(cenarios):
            scenario_df = pd.DataFrame([sc])
            df_window = build_lstm_window(scenario_df)
            df_out = scale_input_row(scenario_df)
            df_norm = df_out[df_out.columns[df_out.columns.str.endswith("_norm")]]
            scenario_dict = df_norm.to_dict(orient="records")[0]

            payload = {"date": str(selected_date), "scenario": scenario_dict}
            res = call_api_with_debounce(payload)
            #payload = {"date": str(selected_date), "scenario": sc}
            #res = requests.post(API_URL, json=payload)
            if res.status_code == 200:
                pred = float(res.json()["prediction"])
                resultados.append((f"Cenário {idx+1}", pred))

        df_res = pd.DataFrame(resultados, columns=["cenário", "previsão"])
        st.dataframe(df_res)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(df_res["cenário"], df_res["previsão"], color="#4EA8DE")
        ax.set_title("Comparação de Cenários", color="white")
        ax.tick_params(colors="white")
        st.pyplot(fig)

# -------------------------------------------------------------------------
# TAB: BACKTEST
# -------------------------------------------------------------------------
with tab_backtest:
    st.header("📊 Backtest Automático (rolling-origin)")

    if st.button("Executar Backtest"):
        with st.spinner("Executando backtest..."):
            df_bt = backtest(df_hist)

        if df_bt.empty:
            st.warning("Backtest não pôde ser executado.")
        else:
            st.dataframe(df_bt)

            fig_bt, ax_bt = plt.subplots(figsize=(10, 4))
            ax_bt.plot(df_bt["data"], df_bt["erro"], color="#E63946")
            ax_bt.axhline(0, color="white")
            ax_bt.set_title("Erro da Previsão", color="white")
            ax_bt.grid(alpha=0.2)
            ax_bt.tick_params(colors="white")
            st.pyplot(fig_bt)






