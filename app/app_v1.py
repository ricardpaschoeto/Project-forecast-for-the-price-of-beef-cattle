# app.py
# Prototipo aprimorado de Dashboard de Previsao do Preco do Boi de Corte (Streamlit)
# Inclui: backtests avancados, explicabilidade (SHAP/correlacao), cenarios, alertas, exportacao e ganchos para LSTM.
# Observacao: Usa dados e elasticidades sinteticas para demonstracao. Substitua pelo seu LSTM e dados reais.

import io
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

try:
    import shap
except Exception:
    shap = None
try:
    import optuna
except Exception:
    optuna = None
try:
    import sklearn
    from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
except Exception:
    sklearn = None
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return float(np.mean(np.abs((y_true - y_pred) / np.maximum(1e-8, y_true))))
    def mean_squared_error(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return float(np.mean((y_true - y_pred)**2))
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model as keras_load_model
except Exception:
    tf = None
    keras_load_model = None
try:
    import torch
except Exception:
    torch = None

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

def ensure_datetime_sorted(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['data'] = pd.to_datetime(df['data'])
    df = df.sort_values('data')
    return df

def _last_slope(series: pd.Series, window: int = 30) -> float:
    if len(series) < window:
        window = len(series)
    y = series.iloc[-window:].values
    x = np.arange(window)
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)

def forecast_placeholder(df: pd.DataFrame, horizon: int, elasticidades: Dict[str, float], cenarios: Dict[str, float]) -> Dict[str, np.ndarray]:
    hist = df['preco_boi'].copy()
    last_price = float(hist.iloc[-1])
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
        adj_factor += beta * float(delta_pct)
    fcst_adj = fcst * adj_factor
    resid = hist.diff().dropna()
    sigma = float(np.std(resid))
    ci95 = 1.96 * sigma
    lower = fcst_adj - ci95
    upper = fcst_adj + ci95
    future_dates = pd.date_range(df['data'].iloc[-1] + timedelta(days=1), periods=horizon, freq='D')
    return {'dates': future_dates, 'point': fcst_adj, 'lower': lower, 'upper': upper}

def keras_predict_iterative(model, series_df: pd.DataFrame, feature_cols: List[str], target_col: str, timesteps: int, horizon: int, scale: bool = False) -> np.ndarray:
    data = series_df[feature_cols].values.astype(np.float32)
    if scale:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) + 1e-8
        data = (data - mean) / std
    seq = data[-timesteps:, :]
    preds = []
    for h in range(horizon):
        x = np.expand_dims(seq, axis=0)
        yhat = model.predict(x, verbose=0)
        yhat_val = float(np.squeeze(yhat))
        preds.append(yhat_val)
        new_row = seq[-1, :].copy()
        new_row[0] = yhat_val
        seq = np.vstack([seq[1:], new_row])
    return np.array(preds, dtype=np.float32)

def forecast_lstm(df: pd.DataFrame, horizon: int, framework: str, model_file, feature_cols: List[str], target_col: str, timesteps: int, scale: bool, elasticidades: Dict[str, float], cenarios: Dict[str, float]) -> Dict[str, np.ndarray]:
    try:
        if framework == 'Keras' and keras_load_model is not None:
            model = keras_load_model(model_file)
            preds = keras_predict_iterative(model, df, feature_cols, target_col, timesteps, horizon, scale=scale)
        elif framework == 'PyTorch' and torch is not None:
            preds = forecast_placeholder(df, horizon, elasticidades, {'usd':0,'chuva_mm':0,'custo_racao':0})['point']
        else:
            preds = forecast_placeholder(df, horizon, elasticidades, cenarios)['point']
    except Exception as e:
        st.warning('Falha ao usar o modelo carregado; usando previsao placeholder. Erro: ' + str(e))
        preds = forecast_placeholder(df, horizon, elasticidades, cenarios)['point']
    adj_factor = 1.0
    for var, delta_pct in cenarios.items():
        beta = elasticidades.get(var, 0.0)
        adj_factor += beta * float(delta_pct)
    preds_adj = preds * adj_factor
    resid = df[target_col].diff().dropna()
    sigma = float(np.std(resid))
    ci95 = 1.96 * sigma
    lower = preds_adj - ci95
    upper = preds_adj + ci95
    future_dates = pd.date_range(df['data'].iloc[-1] + timedelta(days=1), periods=horizon, freq='D')
    return {'dates': future_dates, 'point': preds_adj, 'lower': lower, 'upper': upper}

def baseline_naive_last_value(df: pd.DataFrame, horizon: int, target_col: str = 'preco_boi') -> Dict[str, np.ndarray]:
    last_price = float(df[target_col].iloc[-1])
    fcst = np.full(horizon, last_price)
    future_dates = pd.date_range(df['data'].iloc[-1] + timedelta(days=1), periods=horizon, freq='D')
    return {'dates': future_dates, 'point': fcst}

def backtest_rolling(df: pd.DataFrame, horizon_list: List[int], train_min: int = 365, step: int = 30, target_col: str = 'preco_boi') -> pd.DataFrame:
    rows = []
    n = len(df)
    for start in range(train_min, n - max(horizon_list), step):
        train = df.iloc[:start]
        for h in horizon_list:
            test = df.iloc[start:start+h]
            fc = forecast_placeholder(train, h, {'usd':0.2,'chuva_mm':-0.05,'custo_racao':0.1}, {'usd':0,'chuva_mm':0,'custo_racao':0})['point']
            base = baseline_naive_last_value(train, h, target_col)['point']
            y_true = test[target_col].values
            mape_fc = mean_absolute_percentage_error(y_true, fc) * 100.0
            rmse_fc = np.sqrt(mean_squared_error(y_true, fc))
            mape_base = mean_absolute_percentage_error(y_true, base) * 100.0
            rmse_base = np.sqrt(mean_squared_error(y_true, base))
            rows.append({'inicio': train['data'].iloc[-1], 'horizonte': h, 'MAPE_modelo': mape_fc, 'RMSE_modelo': rmse_fc, 'MAPE_baseline': mape_base, 'RMSE_baseline': rmse_base})
    return pd.DataFrame(rows)

def plot_timeseries(df: pd.DataFrame, fcst: Dict[str, np.ndarray], baseline: Dict[str, np.ndarray], title: str = 'Preco do boi: historico e previsao') -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['data'], df['preco_boi'], label='Historico', color='#1f77b4')
    ax.plot(fcst['dates'], fcst['point'], label='Previsao (modelo)', color='#e377c2')
    if 'lower' in fcst and 'upper' in fcst:
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

def plot_shap_bar(shap_values: np.ndarray, feature_names: List[str]) -> io.BytesIO:
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(mean_abs)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(feature_names)), mean_abs[order], color='#9467bd')
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels([feature_names[i] for i in order], rotation=45, ha='right')
    ax.set_title('Importancia media (|SHAP|)')
    ax.set_ylabel('Valor medio absoluto')
    ax.grid(alpha=0.3)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf

def build_pdf(kpis: Dict[str, str], fig_ts: io.BytesIO, fig_err: io.BytesIO, fig_corr: io.BytesIO, shap_buf: Optional[io.BytesIO]) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    c.setFont('Helvetica-Bold', 16)
    c.drawString(40, height - 40, 'Relatorio de Previsao do Preco do Boi de Corte')
    c.setFont('Helvetica', 10)
    c.drawString(40, height - 60, 'Gerado em: ' + datetime.now().strftime('%d/%m/%Y %H:%M'))
    c.setFont('Helvetica-Bold', 12)
    c.drawString(40, height - 90, 'Resumo de KPIs')
    c.setFont('Helvetica', 10)
    y = height - 110
    for k, v in kpis.items():
        c.drawString(40, y, '- ' + str(k) + ': ' + str(v))
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
    if shap_buf is not None:
        draw_img(shap_buf, 40, 20, 520)
        c.drawString(40, 10, 'Importancia de features (SHAP)')
    c.showPage()
    c.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

def main():
    st.set_page_config(page_title='Previsao do Preco do Boi de Corte', layout='wide')
    st.markdown('<style> .stMetric {background:#f9fafb;padding:10px;border-radius:8px;} </style>', unsafe_allow_html=True)
    st.title('Previsao do Preco do Boi de Corte')
    st.caption('Prototipo visual com recursos avancados — substitua pelo seu modelo LSTM e dados reais.')

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
        st.info('Elasticidades ilustrativas (demo): USD +0,20; Chuva -0,05; Racao +0,10.')
        st.divider()
        st.subheader('Modelo LSTM (opcional)')
        framework = st.selectbox('Framework', ['Placeholder', 'Keras', 'PyTorch'], index=0)
        model_file = st.file_uploader('Carregar modelo (.h5 para Keras, .pt para PyTorch)', type=['h5','pt'])
        timesteps = st.number_input('Timesteps (janela de entrada)', min_value=10, max_value=240, value=60, step=10)
        scale = st.toggle('Padronizar features (z-score)', value=False)

    if uploaded and not use_synth:
        try:
            df_raw = load_uploaded_data(uploaded)
        except Exception as e:
            st.error('Erro ao carregar arquivo: ' + str(e))
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
    df = ensure_datetime_sorted(df)

    elasticidades = {'usd': 0.20, 'chuva_mm': -0.05, 'custo_racao': 0.10}
    cenarios = {'usd': delta_usd / 100.0, 'chuva_mm': delta_chuva / 100.0, 'custo_racao': delta_racao / 100.0}
    feature_cols = ['preco_boi', 'usd', 'chuva_mm', 'custo_racao']
    target_col = 'preco_boi'
    if framework == 'Keras' and model_file is not None and keras_load_model is not None:
        fcst = forecast_lstm(df, horizonte, framework, model_file, feature_cols, target_col, timesteps, scale, elasticidades, cenarios)
    else:
        fcst = forecast_placeholder(df, horizonte, elasticidades, cenarios)
    base = baseline_naive_last_value(df, horizonte, target_col)

    preco_atual = float(df['preco_boi'].iloc[-1])
    preco_prev_medio = float(np.mean(fcst['point']))
    tendencia = 'Alta' if preco_prev_medio > preco_atual else 'Baixa'
    ic95_amplitude = float(np.mean(fcst['upper'] - fcst['lower'])) if 'upper' in fcst else 0.0

    st.subheader('Visao Geral')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Preco atual (R$)', '{:,.2f}'.format(preco_atual))
    c2.metric('Previsao media {}d (R$)'.format(horizonte), '{:,.2f}'.format(preco_prev_medio))
    c3.metric('Tendencia', tendencia)
    c4.metric('Amplitude media IC 95% (R$)', '{:,.2f}'.format(ic95_amplitude))
    fig_ts_buf = plot_timeseries(df, fcst, base)
    st.image(fig_ts_buf, caption='Historico e previsao com intervalo de confianca')

    tab1, tab2, tab3, tab4, tab5 = st.tabs(['Desempenho', 'Explicabilidade', 'Cenarios', 'Avancado', 'Relatorio'])

    with tab1:
        st.subheader('Comparativo de desempenho (backtest simples)')
        bt = backtest_rolling(df, horizon_list=[30], train_min=365, step=30, target_col=target_col)
        if len(bt) > 0:
            c1, c2 = st.columns(2)
            m1 = bt['MAPE_modelo'].mean()
            m2 = bt['MAPE_baseline'].mean()
            c1.metric('MAPE Modelo (media)', '{:.2f}%'.format(m1))
            c2.metric('MAPE Baseline (media)', '{:.2f}%'.format(m2))
        fig_err_buf = plot_error_distribution(df)
        st.image(fig_err_buf, caption='Distribuicao das variacoes diarias do preco')
        st.caption('Backtest rolling com horizonte 30 dias (demo). Para producao: avalie multiplas janelas e horizontes.')

    with tab2:
        st.subheader('Explicabilidade do modelo')
        shap_buf = None
        if shap is not None and framework == 'Keras' and model_file is not None and keras_load_model is not None:
            try:
                model = keras_load_model(model_file)
                X = df[feature_cols].tail(200).values.astype(np.float32)
                # Demo apenas: gere shap_values reais no seu pipeline
                shap_values = np.random.normal(0, 1, size=(X.shape[0]-min(60, X.shape[0]//2), X.shape[1]))
                shap_buf = plot_shap_bar(shap_values, feature_cols)
                st.image(shap_buf, caption='Importancia de features (demo)')
                st.warning('Demo SHAP: substitua pela chamada correta do Kernel/DeepExplainer do seu modelo.')
            except Exception as e:
                st.info('Nao foi possivel calcular SHAP automaticamente. Motivo: ' + str(e))
        if shap_buf is None:
            st.subheader('Correlacao entre variaveis (proxy)')
            fig_corr_buf = plot_correlations(df)
            st.image(fig_corr_buf, caption='Mapa de correlacao (Pearson)')

    with tab3:
        st.subheader('Tabela de cenarios')
        cenarios_table = []
        for d_usd in [-0.05, 0.0, 0.05, 0.10]:
            for d_feed in [-0.05, 0.0, 0.05, 0.10]:
                cenario = {'usd': d_usd, 'chuva_mm': 0.0, 'custo_racao': d_feed}
                fc = forecast_placeholder(df, horizonte, elasticidades, cenario)
                cenarios_table.append({
                    'USD (Delta%)': '{:.0f}%'.format(d_usd*100),
                    'Racao (Delta%)': '{:.0f}%'.format(d_feed*100),
                    'Preco medio {}d (R$)'.format(horizonte): float(np.mean(fc['point'])),
                })
        st.dataframe(pd.DataFrame(cenarios_table))
        st.caption('Elasticidades ilustrativas aplicadas sobre a previsao.')

    with tab4:
        st.subheader('Recursos avancados')
        with st.expander('Backtests avancados (rolling, multiplos horizontes)'):
            bt2 = backtest_rolling(df, horizon_list=[30,60,90], train_min=365, step=30, target_col=target_col)
            if len(bt2) > 0:
                st.dataframe(bt2)
                st.caption('Analise rolling-origin com horizontes 30/60/90 (demo).')
        with st.expander('Otimizacao de hiperparametros (Optuna)'):
            trials_file = st.file_uploader('Carregar resultados de trials (CSV)', type=['csv'])
            if trials_file is not None:
                trials_df = pd.read_csv(trials_file)
                st.dataframe(trials_df)
                if 'mape' in trials_df.columns:
                    best_row = trials_df.iloc[trials_df['mape'].idxmin()]
                    st.success('Melhor trial MAPE: ' + str(best_row.get('mape')))
            else:
                demo_trials = pd.DataFrame([{
                    'trial': i,
                    'params': json.dumps({'lr': round(10**np.random.uniform(-4,-2),5), 'units': int(np.random.choice([32,64,128]))}),
                    'mape': round(np.random.uniform(4.0, 9.0), 2),
                    'rmse': round(np.random.uniform(50.0, 120.0), 2)
                } for i in range(10)])
                st.dataframe(demo_trials)
                st.caption('Demo: substitua pelo CSV exportado do seu estudo Optuna.')
        with st.expander('Alertas'):
            threshold_pct = st.number_input('Limiar de variacao (%) para alerta', min_value=0.0, max_value=100.0, value=5.0, step=0.5)
            alerts = []
            for dt, p in zip(fcst['dates'], fcst['point']):
                change_pct = (p - preco_atual) / max(preco_atual, 1e-8) * 100.0
                if abs(change_pct) >= threshold_pct:
                    direction = 'Alta' if change_pct > 0 else 'Baixa'
                    alerts.append({'data': dt.date(), 'variacao_%': round(change_pct,2), 'direcao': direction, 'preco_previsto': round(float(p),2)})
            if alerts:
                st.dataframe(pd.DataFrame(alerts))
            else:
                st.info('Nenhum alerta para o limiar atual.')

    with tab5:
        st.subheader('Exportar')
        kpis = {
            'Preco atual (R$)': '{:,.2f}'.format(preco_atual),
            'Previsao media {}d (R$)'.format(horizonte): '{:,.2f}'.format(preco_prev_medio),
            'Tendencia': tendencia,
            'Amplitude media IC 95% (R$)': '{:,.2f}'.format(ic95_amplitude),
        }
        fig_err_buf = plot_error_distribution(df)
        fig_corr_buf = plot_correlations(df)
        shap_demo_buf = None
        pdf_bytes = build_pdf(kpis, fig_ts_buf, fig_err_buf, fig_corr_buf, shap_demo_buf)
        st.download_button('Baixar relatorio (PDF)', data=pdf_bytes, file_name='relatorio_preco_boi_{}d.pdf'.format(horizonte), mime='application/pdf')
        df_fcst = pd.DataFrame({'data': fcst['dates'], 'previsao': fcst['point']})
        if 'lower' in fcst and 'upper' in fcst:
            df_fcst['ic_lower'] = fcst['lower']
            df_fcst['ic_upper'] = fcst['upper']
        csv_bytes = df_fcst.to_csv(index=False).encode('utf-8')
        st.download_button('Baixar previsao (CSV)', data=csv_bytes, file_name='previsao_{}d.csv'.format(horizonte), mime='text/csv')

    st.divider()
    st.markdown('''
    Notas:
    - Este e um prototipo visual com dados e elasticidades sinteticas (demo).
    - Para producao: carregue dados reais, integre seu LSTM (Keras/PyTorch) e substitua os blocos de previsao/SHAP.
    - Use backtests rolling com multiplas janelas e horizontes; reporte MAPE/RMSE/MAE.
    - Inclua alertas automatizados via e-mail/WhatsApp quando variacoes relevantes forem previstas.
    ''')

if __name__ == '__main__':
    main()