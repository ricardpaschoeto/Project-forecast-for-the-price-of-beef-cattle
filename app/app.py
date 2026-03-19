# app.py — versão profissional com:
# ✔ Previsão na SIDEBAR
# ✔ Abas: Backtest | Cenários | Dashboard
# ✔ Trading Terminal styling (dark)
# ✔ IC 95%
# ✔ Exportação CSV e PDF
# ✔ Cenários múltiplos (valores absolutos)
# ✔ Backtest rolling-origin
# -------------------------------------------------------------------------

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
from pathlib import Path
import os

# -------------------------------------------------------------------------
# CONFIGURAÇÕES
# -------------------------------------------------------------------------

API_URL = "http://localhost:8000/predict"

path = Path(os.path.abspath(__file__))
root_dir = path.parent.parent
df_path = os.path.join(root_dir,'data_pipeline', 'sensors', 'dados_modelo_lasso.csv')

FEATURE_COLS = [
    "boi_futuro", "boi_dolar",
    "soja_real", "soja_dolar", "soja_futuro",
    "milho_futuro", "milho_real", "milho_dolar",
    "precip_total_mm", "selic", "cme", "week"
]

# ======================================================================
# CSS avançado
# ======================================================================

layout_css = """
<style>

/* Remove padding lateral do app */
.block-container {
    padding-top: 1rem !important;
    padding-left: 0rem !important;
    padding-right: 0rem !important;
    padding-bottom: 0rem !important;
    max-width: 100% !important;
}

/* Tabs ocupando largura máxima */
div[data-baseweb="tab-list"] {
    width: 100% !important;
    justify-content: space-around !important;
}

/* Reduz padding vertical das seções */
section.main > div {
    padding-top: 0rem !important;
}

/* Cards dos gráficos: ocupam largura total e reduzem bordas */
.cowboy-card {
    padding: 5px 15px;
    margin-bottom: 10px;
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


def compute_sigma(df):
    """Desvio padrão dos retornos — usado para IC 95%."""
    diffs = df["boi_negociado"].diff().dropna()
    return diffs.std()


def plot_forecast(df_hist, data_prev, y_pred, ic95):
    """Gráfico histórico + previsão + IC95"""

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    ax.plot(df_hist["data"],
            df_hist["boi_negociado"],
            color="#4EA8DE",
            label="Histórico",
            linewidth=2)

    ax.scatter(pd.to_datetime(data_prev),
               y_pred,
               color="#E63946",
               s=100,
               label="Previsão")

    ax.errorbar(pd.to_datetime(data_prev),
                y_pred,
                yerr=ic95,
                fmt="o",
                color="gray",
                capsize=8,
                label="IC 95%")

    ax.axvline(pd.to_datetime(data_prev),
               linestyle="--",
               color="#6C757D",
               alpha=0.6)

    ax.set_title("Histórico + Previsão LSTM + IC 95%", color="white")
    ax.set_xlabel("Data", color="white")
    ax.set_ylabel("Preço (R$)", color="white")

    ax.tick_params(colors="white")
    ax.grid(alpha=0.2)
    ax.legend(facecolor="#1E1E1E", labelcolor="white")

    return fig


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


# =========================================================================
# SIDEBAR — PREVISÃO ONE-STEP
# =========================================================================

with st.sidebar:

    st.markdown("### 🧩 Configure os valores futuros e gere uma previsão.")

    # Grid responsivo para inputs
    st.markdown('<div class="sidebar-grid">', unsafe_allow_html=True)

    # Data
    selected_date = st.date_input("📅 Data da Previsão", value=date.today(), min_value=date.today())

    scenario = {}
    ultima_data = df_hist.index.max()

    scenario["boi_futuro"] = st.number_input("Preço Boi Futuro (R$)", value=float(df_hist.loc[ultima_data, 'boi_futuro']), format="%.4f", help="Preço do contrato futuro do boi gordo (B3) para o vencimento selecionado.")
    scenario["boi_dolar"] = st.number_input("Preço Boi ($)", value=float(df_hist.loc[ultima_data, 'boi_dolar']), format="%.4f", help="Preço da arroba do boi gordo convertido para dólares americanos.")
    scenario["soja_real"] = st.number_input("Preço Soja (R$)", value=float(df_hist.loc[ultima_data, 'soja_real']), format="%.4f", help="Preço atual da saca de soja (60kg) no mercado físico brasileiro em Reais.")
    scenario["soja_dolar"] = st.number_input("Preço Soja ($)", value=float(df_hist.loc[ultima_data, 'soja_dolar']), format="%.4f", help="Preço da saca de soja (60kg) convertido para dólares americanos.")
    scenario["soja_futuro"] = st.number_input("Preço Soja Futuro (R$)", value=float(df_hist.loc[ultima_data, 'soja_futuro']), format="%.4f", help="Cotação do contrato futuro da soja na CBOT ou B3.")
    scenario["milho_futuro"] = st.number_input("Preço Milho Futuro (R$)", value=float(df_hist.loc[ultima_data, 'milho_futuro']), format="%.4f", help="Preço do contrato futuro do milho (B3) para a data de liquidação.")
    scenario["milho_real"] = st.number_input("Preço Milho (R$)", value=float(df_hist.loc[ultima_data, 'milho_real']), format="%.4f", help="Preço atual da saca de milho (60kg) no mercado físico brasileiro em Reais.")
    scenario["milho_dolar"] = st.number_input("Preço Milho ($)", value=float(df_hist.loc[ultima_data, 'milho_dolar']), format="%.4f", help="Preço da saca de milho (60kg) convertido para dólares americanos.")
    scenario["precip_total_mm"] = st.number_input("Precipitação Total (mm)", value=float(df_hist.loc[ultima_data, 'precip_total_mm']), format="%.4f", help="Volume total de chuva acumulado (em milímetros) no período ou região.")
    scenario["selic"] = st.number_input("Taxa SELIC (%)", value=float(df_hist.loc[ultima_data, 'selic']), format="%.4f", help="Taxa básica de juros da economia brasileira (em % ao ano).")
    scenario["cme"] = st.number_input("Índice CME", value=float(df_hist.loc[ultima_data, 'cme']), format="%.4f", help="Cotação vinda da Chicago Mercantile Exchange.")
    scenario["boi_negociado"] = st.number_input("Boi Negociado (Vol)", value=float(df_hist.loc[ultima_data, 'boi_negociado']), format="%.4f", help="Volume total de cabeças de gado comercializadas no período.")

    scenario["week"] = selected_date.isocalendar().week

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    run_forecast = st.button("🚀 Gerar Previsão")

# =========================================================================
# RESULTADOS DA PREVISÃO (MOSTRADOS NO CORPO)
# =========================================================================

# ======================================================================
# RESULTADO DA PREVISÃO — Corpo Principal
# ======================================================================

if run_forecast:

    payload = {"date": str(selected_date), "scenario": scenario}
    res = requests.post(API_URL, json=payload)

    if res.status_code != 200:
        st.error("Erro na API: " + res.text)
    else:
        pred = float(res.json()["prediction"])
        data_prev = res.json()["date"]

        preco_atual = float(df_hist["boi_negociado"].iloc[-1])
        tendencia = "Alta" if pred > preco_atual else "Baixa"

        sigma = compute_sigma(df_hist)
        ic95 = 1.96 * sigma

        st.markdown("## 🔮 Resultado da Previsão")
        st.markdown("### 📌 Indicadores")

        c1, c2, c3 = st.columns(3)
        c1.metric("Preço Atual", f"{preco_atual:.2f}")
        c2.metric("Previsão", f"{pred:.2f}")
        c3.metric("Tendência", tendencia)

        st.markdown("### 📉 Gráfico da Previsão")
        fig = plot_forecast(df_hist, data_prev, pred, ic95)
        st.pyplot(fig)

        st.markdown("### 💾 Exportação")
        df_exp = df_hist.copy()
        df_exp.loc[len(df_exp)] = [pd.to_datetime(data_prev)] + \
            [scenario[c] for c in FEATURE_COLS] + [pred]

        csv_bytes = df_exp.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Exportar CSV", csv_bytes, "previsao.csv")

        pdf_bytes = gerar_pdf(preco_atual, pred, data_prev, tendencia, fig)
        st.download_button("📘 Exportar PDF", pdf_bytes, "relatorio_previsao.pdf")

st.markdown("---")


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

    # Compute indicators
    rsi = compute_rsi(df_hist["boi_negociado"], window_rsi)
    macd = compute_macd(df_hist["boi_negociado"], window_macd_fast, window_macd_slow)
    vol = compute_volatility(df_hist["boi_negociado"], window_vol)

    preco_atual = df_hist["boi_negociado"].iloc[-1]
    ultimos_30 = df_hist["boi_negociado"].tail(30)
    variacao_30d = ((ultimos_30.iloc[-1] - ultimos_30.iloc[0]) /
                     ultimos_30.iloc[0]) * 100

    c1, c2 = st.columns(2)
    c1.metric("Preço Atual (R$)", f"{preco_atual:.2f}")
    c2.metric("Variação 30 dias (%)", f"{variacao_30d:.2f}%")

    st.markdown("### 📈 Gráfico Histórico")

    # ================================
    # FILTRAR ÚLTIMOS 12 MESES
    # ================================
    cutoff_date = df_hist.index.max() - pd.DateOffset(months=12)
    df_12m = df_hist[df_hist.index >= cutoff_date]

    # Recalcular indicadores somente com os últimos 12 meses
    rsi = compute_rsi(df_12m["boi_negociado"], window_rsi)
    macd = compute_macd(df_12m["boi_negociado"], window_macd_fast, window_macd_slow)
    vol = compute_volatility(df_12m["boi_negociado"], window_vol)

    # ================================
    # FIGURA 1 — Preço 12 meses
    # ================================
    st.markdown('<div class="cowboy-card">', unsafe_allow_html=True)
    fig1, ax1 = plt.subplots(figsize=(20, 4))
    ax1.plot(df_12m.index, df_12m["boi_negociado"], color="#4EA8DE", linewidth=2, label="Preço do Boi")

    # --- Título e labels grandes ---
    ax1.set_title("Histórico — Boi Negociado (Últimos 12 meses)", color="white", fontsize=20)
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

    st.pyplot(fig1)
    st.markdown('</div>', unsafe_allow_html=True)

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
# TAB: CENÁRIOS MÚLTIPLOS
# -------------------------------------------------------------------------
with tab_cenarios:
    st.header("🧪 Simulação de Cenários Múltiplos (Valores Absolutos)")

    num_cenarios = st.slider("Número de cenários", 1, 5, 2)

    cenarios = []
    for i in range(num_cenarios):
        st.subheader(f"Cenário {i+1}")
        cols2 = st.columns(3)
        sc = {}
        for j, col in enumerate(FEATURE_COLS):
            with cols2[j % 3]:
                sc[col] = st.number_input(f"{col} — cenário {i+1}", value=0.0, format="%.4f")
        cenarios.append(sc)

    if st.button("Simular Cenários"):
        resultados = []
        for idx, sc in enumerate(cenarios):
            payload = {"date": str(selected_date), "scenario": sc}
            res = requests.post(API_URL, json=payload)
            if res.status_code == 200:
                pred = float(res.json()["prediction"])
                resultados.append((f"Cenário {idx+1}", pred))

        df_res = pd.DataFrame(resultados, columns=["cenário", "previsão"])
        st.dataframe(df_res)

        fig3, ax3 = plt.subplots(figsize=(8, 4))
        ax3.bar(df_res["cenário"], df_res["previsão"], color="#4EA8DE")
        ax3.set_title("Comparação de Cenários", color="white")
        ax3.tick_params(colors="white")
        st.pyplot(fig3)

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






