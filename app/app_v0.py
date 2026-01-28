# app.py
# Prototipo de Dashboard de Previsao do Preco do Boi de Corte (Streamlit)
# Autor: Ricardo (com suporte do M365 Copilot)
# Observacao: Este app usa dados e elasticidades sinteticas apenas para demonstracao.
# Substitua as funcoes de previsao pelo seu LSTM e carregue dados reais.

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

def generate_synthetic_data(n_days: int = 3*365, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    start_date = datetime.today() - timedelta(days=n_days)
    dates = pd.date_range(start_date, periods=n_days, freq='D')

    trend = np.linspace(0, 200, n_days)
    season = 50 * np.sin(2 * np.pi * np.arange(n_days) / 365)
    noise = np.random.normal(0, 15, n_days)
    price = 250 + trend + season + noise

    usd = 4.5 + 0.0005 * np.arange(n_days) + np.random.normal(0, 0.05, n_days)
    rain = 5 + 50 * np.maximum(0, np.sin(2 * np.pi * (np.arange(n_days) + 60) / 365)) + np.random.normal(0, 5, n_days)
    feed = 300 + 0.05 * np.arange(n_days) + np.random.normal(0, 5, n_days)

    df = pd.DataFrame({
        'data': dates,
        'preco_boi': price,
        'usd': usd,
        'chuva_mm': rain,
        'custo_racao': feed,
    })
    return df

def load_uploaded_data(file) -> pd.DataFrame:
    try:
        df = pd.read_csv(file)
    except Exception:
        file.seek(0)
        df = pd.read_excel(file, engine='openpyxl')
    return df

def _last_slope(series: pd.Series, window: int = 30) -> float:
    if len(series) < window:
        window = len(series)
    y = series.iloc[-window:].values
    x = np.arange(window)
    slope = np.polyfit(x, y, 1)[0]
    return slope

def forecast_placeholder(df: pd.DataFrame, horizon: int, elasticidades: dict, cenarios: dict) -> dict:
    hist = df['preco_boi'].copy()
    last_price = hist.iloc[-1]
    slope = _last_slope(hist, window=30)
    daily_drift = slope
    fcst = []
    level = last_price
    for t in range(1, horizon+1):
        level = level + daily_drift + np.random.normal(0, 3)
        fcst.append(level)
    fcst = np.array(fcst)

    adj_factor = 1.0
    for var, delta_pct in cenarios.items():
        beta = elasticidades.get(var, 0.0)
        adj_factor += beta * delta_pct

    fcst_adj = fcst * adj_factor

    resid = hist.diff().dropna()
    sigma = np.std(resid)
    ci95 = 1.96 * sigma
    lower = fcst_adj - ci95
    upper = fcst_adj + ci95

    future_dates = pd.date_range(df['data'].iloc[-1] + timedelta(days=1), periods=horizon, freq='D')
    return {
        'dates': future_dates,
        'point': fcst_adj,
        'lower': lower,
        'upper': upper,
    }

    # 1) Preparar X do LSTM a partir de df (janelas, features, etc.)
    # X_input = ...

    # 2) Chamar seu modelo LSTM já treinado
    # y_hat = lstm_model.predict(X_input)  # shape [horizon]

    # 3) Aplicar ajustes de cenário (se desejar manter)
    # adj_factor = 1.0 + elasticidades['usd']*cenarios['usd'] \
    #                      + elasticidades['chuva_mm']*cenarios['chuva_mm'] \
    #                      + elasticidades['custo_racao']*cenarios['custo_racao']
    # y_hat_adj = y_hat * adj_factor

    # 4) Calcular banda de confiança (ex.: desvio dos resíduos do histórico)
    # sigma = ...
    # lower = y_hat_adj - 1.96 * sigma
    # upper = y_hat_adj + 1.96 * sigma

    # 5) Datas futuras
    # future_dates = pd.date_range(df['data'].iloc[-1] + timedelta(days=1), periods=horizon, freq='D')
    # return {'dates': future_dates, 'point': y_hat_adj, 'lower': lower, 'upper': upper}


def baseline_naive_last_value(df: pd.DataFrame, horizon: int) -> dict:
    last_price = df['preco_boi'].iloc[-1]
    fcst = np.full(horizon, last_price)
    future_dates = pd.date_range(df['data'].iloc[-1] + timedelta(days=1), periods=horizon, freq='D')
    return {'dates': future_dates, 'point': fcst}

def backtest_mape(df: pd.DataFrame, horizon: int = 30) -> dict:
    if len(df) <= horizon + 30:
        return {'mape_lstm': np.nan, 'mape_baseline': np.nan}
    train = df.iloc[:-horizon]
    test = df.iloc[-horizon:]
    elasticidades = {'usd': 0.2, 'chuva_mm': -0.05, 'custo_racao': 0.1}
    cenarios = {'usd': 0.0, 'chuva_mm': 0.0, 'custo_racao': 0.0}
    fcst = forecast_placeholder(train, horizon=horizon, elasticidades=elasticidades, cenarios=cenarios)['point']
    base = baseline_naive_last_value(train, horizon=horizon)['point']
    y_true = test['preco_boi'].values
    mape_lstm = np.mean(np.abs((y_true - fcst) / y_true)) * 100
    mape_baseline = np.mean(np.abs((y_true - base) / y_true)) * 100
    return {'mape_lstm': mape_lstm, 'mape_baseline': mape_baseline}

def plot_timeseries(df: pd.DataFrame, fcst: dict, baseline: dict, title: str = 'Preco do boi: historico e previsao') -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['data'], df['preco_boi'], label='Historico', color='#1f77b4')
    ax.plot(fcst['dates'], fcst['point'], label='Previsao (modelo)', color='#e377c2')
    ax.fill_between(fcst['dates'], fcst['lower'], fcst['upper'], color='#e377c2', alpha=0.2, label='IC 95%')
    ax.plot(baseline['dates'], baseline['point'], label='Baseline (ultimo valor)', color='#ff7f0e', linestyle='--')
    ax.set_title(title)
    ax.set_xlabel('Data')
    ax.set_ylabel('Preco (R$)')
    ax.legend()
    ax.grid(alpha=0.3)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf

def plot_error_distribution(df: pd.DataFrame) -> io.BytesIO:
    diffs = df['preco_boi'].diff().dropna()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(diffs, bins=30, color='#2ca02c', alpha=0.7)
    ax.set_title('Distribuicao das variacoes diarias do preco')
    ax.set_xlabel('Delta Preco (R$)')
    ax.set_ylabel('Frequencia')
    ax.grid(alpha=0.3)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf

def plot_correlations(df: pd.DataFrame) -> io.BytesIO:
    cols = ['preco_boi', 'usd', 'chuva_mm', 'custo_racao']
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(corr.values, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha='right')
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)
    ax.set_title('Mapa de correlacao')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf

def build_pdf(kpis: dict, fig_ts: io.BytesIO, fig_err: io.BytesIO, fig_corr: io.BytesIO) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    c.setFont('Helvetica-Bold', 16)
    c.drawString(40, height - 40, 'Relatorio de Previsao do Preco do Boi de Corte')
    c.setFont('Helvetica', 10)
    c.drawString(40, height - 60, f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    c.setFont('Helvetica-Bold', 12)
    c.drawString(40, height - 90, 'Resumo de KPIs')
    c.setFont('Helvetica', 10)
    y = height - 110
    for k, v in kpis.items():
        c.drawString(40, y, f'- {k}: {v}')
        y -= 14

    def draw_img(image_bytes, x, y, w):
        img = ImageReader(image_bytes)
        c.drawImage(img, x, y, width=w, preserveAspectRatio=True, mask='auto')

    draw_img(fig_ts, 40, 280, 520)
    c.drawString(40, 270, 'Historico e previsao com IC 95%')
    draw_img(fig_err, 40, 140, 250)
    c.drawString(40, 130, 'Distribuicao das variacoes diarias')
    draw_img(fig_corr, 310, 140, 250)
    c.drawString(310, 130, 'Mapa de correlacao')
    c.showPage()
    c.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

def main():
    st.set_page_config(page_title='Previsao do Preco do Boi de Corte', layout='wide')
    st.title('Previsao do Preco do Boi de Corte')
    # st.caption('Prototipo visual — substitua pelo seu modelo LSTM e dados reais.')

    with st.sidebar:
        st.header('Configuracoes')
        uploaded = st.file_uploader('Carregar dados (.csv ou .xlsx)', type=['csv', 'xlsx'])
        use_synth = st.toggle('Usar dados sinteticos (demo)', value=(uploaded is None))
        horizonte = st.select_slider('Horizonte de previsao (dias)', options=[30, 60, 90], value=60)
        st.divider()
        st.subheader('Cenarios (Delta% em relacao ao ultimo valor)')
        delta_usd = st.slider('USD (Delta%)', -10.0, 10.0, 0.0, step=0.5)
        delta_chuva = st.slider('Chuva (Delta%)', -30.0, 30.0, 0.0, step=1.0)
        delta_racao = st.slider('Custo da racao (Delta%)', -10.0, 10.0, 0.0, step=0.5)
        st.info('Elasticidades ilustrativas: USD +0,20; Chuva -0,05; Racao +0,10 (apenas para demonstracao).')

    if uploaded and not use_synth:
        try:
            df_raw = load_uploaded_data(uploaded)
        except Exception as e:
            st.error(f'Erro ao carregar arquivo: {e}')
            st.stop()
        st.subheader('Mapeamento de colunas')
        cols = list(df_raw.columns)
        col_data = st.selectbox('Coluna de data', cols, index=0)
        col_price = st.selectbox('Coluna de preco do boi', cols, index=min(1, len(cols)-1))
        col_usd = st.selectbox('Coluna USD', cols, index=min(2, len(cols)-1))
        col_rain = st.selectbox('Coluna chuva (mm)', cols, index=min(3, len(cols)-1))
        col_feed = st.selectbox('Coluna custo racao', cols, index=min(4, len(cols)-1))
        df = pd.DataFrame({
            'data': pd.to_datetime(df_raw[col_data]),
            'preco_boi': pd.to_numeric(df_raw[col_price], errors='coerce'),
            'usd': pd.to_numeric(df_raw[col_usd], errors='coerce'),
            'chuva_mm': pd.to_numeric(df_raw[col_rain], errors='coerce'),
            'custo_racao': pd.to_numeric(df_raw[col_feed], errors='coerce'),
        }).dropna()
    else:
        df = generate_synthetic_data()

    df = df.sort_values('data')

    elasticidades = {'usd': 0.20, 'chuva_mm': -0.05, 'custo_racao': 0.10}
    cenarios = {'usd': delta_usd / 100.0, 'chuva_mm': delta_chuva / 100.0, 'custo_racao': delta_racao / 100.0}

    fcst = forecast_placeholder(df, horizonte, elasticidades, cenarios)
    base = baseline_naive_last_value(df, horizonte)

    preco_atual = df['preco_boi'].iloc[-1]
    preco_prev_medio = float(np.mean(fcst['point']))
    tendencia = 'Alta' if preco_prev_medio > preco_atual else 'Baixa'
    ic95_amplitude = float(np.mean(fcst['upper'] - fcst['lower']))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Preco atual (R$)', f'{preco_atual:,.2f}')
    c2.metric(f'Previsao media {horizonte}d (R$)', f'{preco_prev_medio:,.2f}')
    c3.metric('Tendencia', tendencia)
    c4.metric('Amplitude media IC 95% (R$)', f'{ic95_amplitude:,.2f}')

    fig_ts_buf = plot_timeseries(df, fcst, base)
    st.image(fig_ts_buf, caption='Historico e previsao com intervalo de confianca')

    tab1, tab2, tab3, tab4 = st.tabs(['Desempenho', 'Fatores', 'Cenarios', 'Relatorio'])

    with tab1:
        st.subheader('Comparativo de desempenho (backtest)')
        bt = backtest_mape(df, horizon=30)
        c1, c2 = st.columns(2)
        c1.metric('MAPE Modelo (placeholder)', f"{bt['mape_lstm']:.2f}%")
        c2.metric('MAPE Baseline (ultimo valor)', f"{bt['mape_baseline']:.2f}%")
        fig_err_buf = plot_error_distribution(df)
        st.image(fig_err_buf, caption='Distribuicao das variacoes diarias do preco')
        st.caption('Backtest simples apenas para demonstracao. Substitua pelo seu pipeline de avaliacao do LSTM.')

    with tab2:
        st.subheader('Correlacao entre variaveis')
        fig_corr_buf = plot_correlations(df)
        st.image(fig_corr_buf, caption='Mapa de correlacao (Pearson)')
        st.caption('Use SHAP com seu modelo LSTM real para interpretar importancias de features.')

    with tab3:
        st.subheader('Tabela de cenarios')
        cenarios_table = []
        for d_usd in [-0.05, 0.0, 0.05, 0.10]:
            for d_feed in [-0.05, 0.0, 0.05, 0.10]:
                cenario = {'usd': d_usd, 'chuva_mm': 0.0, 'custo_racao': d_feed}
                fc = forecast_placeholder(df, horizonte, elasticidades, cenario)
                cenarios_table.append({
                    'USD (Delta%)': f'{d_usd*100:.0f}%',
                    'Racao (Delta%)': f'{d_feed*100:.0f}%',
                    f'Preco medio {horizonte}d (R$)': np.mean(fc['point']),
                })
        st.dataframe(pd.DataFrame(cenarios_table))
        st.caption('Elasticidades ilustrativas aplicadas sobre a previsao.')

    with tab4:
        st.subheader('Exportar')
        kpis = {
            'Preco atual (R$)': f'{preco_atual:,.2f}',
            f'Previsao media {horizonte}d (R$)': f'{preco_prev_medio:,.2f}',
            'Tendencia': tendencia,
            'Amplitude media IC 95% (R$)': f'{ic95_amplitude:,.2f}',
        }
        fig_err_buf = plot_error_distribution(df)
        fig_corr_buf = plot_correlations(df)
        pdf_bytes = build_pdf(kpis, fig_ts_buf, fig_err_buf, fig_corr_buf)
        st.download_button('Baixar relatorio (PDF)', data=pdf_bytes, file_name=f'relatorio_preco_boi_{horizonte}d.pdf', mime='application/pdf')

        df_fcst = pd.DataFrame({
            'data': fcst['dates'],
            'previsao': fcst['point'],
            'ic_lower': fcst['lower'],
            'ic_upper': fcst['upper'],
        })
        csv_bytes = df_fcst.to_csv(index=False).encode('utf-8')
        st.download_button('Baixar previsao (CSV)', data=csv_bytes, file_name=f'previsao_{horizonte}d.csv', mime='text/csv')

    st.divider()
    st.markdown('''
    Notas:
    - Este e um prototipo visual usando dados e elasticidades sinteticas para fins de demonstracao.
    - Para producao: carregue dados reais, substitua forecast_placeholder por uma chamada ao seu modelo LSTM, e utilize SHAP/avaliacao robusta (cross-validation, backtesting com multiplas janelas).
    - Inclua alertas (e-mail/WhatsApp) quando a previsao indicar variacoes relevantes.
    ''')

if __name__ == '__main__':
    main()