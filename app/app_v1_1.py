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
from typing import List, Dict, Optional, Callable
from typing import Optional, Sequence, Dict, Any
from sklearn.preprocessing import MinMaxScaler

import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from sklearn.exceptions import NotFittedError
import joblib
import matplotlib.dates as mdates

from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

BASE = os.path.dirname(os.path.abspath(__file__))  # .../app
MODELS_DIR = os.path.abspath(os.path.join(BASE, "..", "ml-pipeline", "models"))

import tensorflow as tf
keras_load_model = tf.keras.models.load_model

try:
    import shap
except Exception:
    shap = None

def load_uploaded_data(file) -> pd.DataFrame:
    try:
        df = pd.read_csv(file)
    except Exception as e:
        print(e)
    return df

def ensure_datetime_sorted(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['data'] = pd.to_datetime(df['data'])
    df = df.sort_values('data')
    return df

def _build_initial_window(df:pd.DataFrame,
                          feature_cols: List[str],
                          timesteps:int) -> np.ndarray:
    if len(df) < timesteps:
        raise ValueError(f"Número de linhas ({len(df)}) < timesteps ({timesteps})")
    X = df[feature_cols].iloc[-timesteps:].values.astype(np.float64)
    
    return X

def marginal_elasticity(
    model,
    X_row: np.ndarray,
    timesteps: int,
    features_length: int,
    target_col:str,
    feature_index: int = 0,
    delta_pct: float = 0.01,
    scaler = MinMaxScaler(),
    ) -> float:
    """
    Calcula elasticidade marginal de um modelo (nao linear) em relacao a uma feature,
    usando uma perturbacao percentual pequena (delta_pct).

    Formula: E_x ~= ((y_mod - y_base)/y_base) / delta_pct
    Onde X_mod = X_row, com a feature multiplicada por (1 + delta_pct).

    Parametros:
      - model_predict: funcao que recebe X (shape [1, n_features] ou [1, t, n_features]) e retorna y (shape [1] ou [1,1])
      - X_row: vetor (ou ultima janela) representativo do estado atual
      - y_base: previsao base (se None, sera calculada)
      - feature_index: indice da feature a perturbar
      - delta_pct: variacao percentual (padrao = 1%)
    """
    # Garantir que X_row seja 2D (batch de 1)
    assert X_row.shape == (1, timesteps, features_length), f"Esperado (1,timesteps, features_length), obtido {X_row.shape}"

    # Perturbar a feature
    X_mod = X_row.copy().astype(np.float64)
    factor = (1.0 + delta_pct)
    X_mod[:,-1, feature_index] *= factor

    # Normaliza as duas para o modelo
    X_row_norm = scaler.fit_transform(np.squeeze(X_row))
    X_mod_norm = scaler.fit_transform(np.squeeze(X_mod))

    # Previsao base
    y0 = np.float64(model(np.expand_dims(X_row_norm, axis=0)))
    y1 = np.float64(model(np.expand_dims(X_mod_norm, axis=0)))

    # Elasticidade por saída: ((y1 - y0)/y0) / delta    
    elastic = np.round(((y1 - y0) / y0) / delta_pct, 6)

    result = {}
    result[target_col] = elastic
 
    return result

def marginal_elasticities_batch(
    model,
    X_row: np.ndarray,
    timesteps: int,
    target_cols: List[str],
    feature_cols: List[str],
    delta_pct: float = 0.01,
    ) -> Dict[str, float]:
    """
    Calcula elasticidade marginal para todas as features listadas.
    """
    results = {}
    features_lenght = len(feature_cols)
    for j, feat in enumerate(feature_cols):
        if feat is not target_cols:
            E = marginal_elasticity(model, X_row, timesteps, features_lenght, target_cols, feature_index=j, delta_pct=delta_pct)
            results[feat] = E
    return results

def forecast_placeholder(df: pd.DataFrame, horizon: int, elasticidades: Dict[str, float], cenarios: Dict[str, float]) -> Dict[str, np.ndarray]:
    hist = df['boi_real'].copy()
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
    return {'dates': future_dates, 'y_pred': fcst_adj, 'lower': lower, 'upper': upper}

def backtest_rolling(df: pd.DataFrame, target_cols: str, horizon_list: List[int], train_min: int = 365, step: int = 30) -> pd.DataFrame:
    rows = []
    n = len(df)
    for start in range(train_min, n - max(horizon_list), step):
        train = df.iloc[:start]
        for h in horizon_list:
            test = df.iloc[start:start+h]
            fc = forecast_placeholder(train, h, {'usd':0.2,'chuva_mm':-0.05,'custo_racao':0.1}, {'usd':0,'chuva_mm':0,'custo_racao':0})['y_pred']
            base = baseline_naive_last_value(train, h, target_cols)['y_pred']
            y_true = test[target_cols].values
            mape_fc = mean_absolute_percentage_error(y_true, fc) * 100.0
            rmse_fc = np.sqrt(mean_squared_error(y_true, fc))
            mape_base = mean_absolute_percentage_error(y_true, base) * 100.0
            rmse_base = np.sqrt(mean_squared_error(y_true, base))
            rows.append({'inicio': train['data'].iloc[-1], 'horizonte': h, 'MAPE_modelo': mape_fc, 'RMSE_modelo': rmse_fc, 'MAPE_baseline': mape_base, 'RMSE_baseline': rmse_base})
    return pd.DataFrame(rows)



def ensure_scaler_fitted(scaler, sample: np.ndarray, inverse: bool = False):
    """
    Verifica se o scaler está 'fitado'. Se não estiver, levanta uma mensagem clara.
    'sample' deve ter shape (n, d) compatível com transform/inverse_transform.
    """
    if scaler is None:
        return
    try:
        if inverse:
            scaler.inverse_transform(sample[:1])
        else:
            scaler.transform(sample[:1])
    except NotFittedError as e:
        raise RuntimeError("Scaler não está fitado. Carregue (joblib.load) ou faça fit com dados do passado.") from e
    except Exception:
        # Alguns scalers levantam Exception genérica quando não fitados
        raise RuntimeError("Scaler não está fitado.")

def _last_slope(series: pd.Series, window: int = 30) -> float:
    if len(series) < window:
        window = len(series)
    y = series.iloc[-window:].values
    x = np.arange(window)
    slope = np.polyfit(x, y, 1)[0]
    return slope

def _iterative_lstm_forecast(model,
                            last_window: np.ndarray,# shape (lookback, n_features)
                            horizon:int,
                            input_scaler: Optional[object],
                            target_scaler: Optional[object],
                            target_col_idx: int,
                            exog_next_raw_fn: Optional[callable] = None,
                            exog_strategy: str = "persistence",  # "persistence" | "zeros" | "custom"
                            ) -> np.ndarray:
    
    # 1) check scalers
    if input_scaler is not None:
        ensure_scaler_fitted(input_scaler, last_window)
        window = input_scaler.transform(last_window) # (lookback, n_features)
    else:
        window = last_window.copy()
    
    lookback, n_features = window.shape 
    preds = []

    # 2) Loop iterativo
    for idx in range(horizon):
        X_model = np.expand_dims(window, axis=0)
        yhat = model.predict(X_model, verbose=0) # shape (1,1)
        yhat_val = float(np.squeeze(yhat))
        preds.append(float(yhat_val))

        # 3) Prepara a próxima linha futura (em ESCALA ORIGINAL)        
        if exog_next_raw_fn is not None:
            next_raw = np.array(exog_next_raw_fn(last_window[-1], idx), dtype=float).reshape(1.-1)  # shape (1, n_features)
            if next_raw.shape[1] != n_features:
                raise ValueError(f"exog_next_raw_fn retornou shape {next_raw.shape}, esperado (1, {n_features})")
        else:
            if exog_strategy == "persistence":
                next_raw = last_window[-1].reshape(1, -1)  # repete última linha
            elif exog_strategy == "zeros":
                next_raw = np.zeros((1, n_features), dtype=float)
            else:
                raise ValueError(f"Estratégia exog_strategy desconhecida: {exog_strategy}")

        # 4) Transforma a próxima linha para o espaço do modelo (features escaladas)
        if input_scaler is not None:
            next_scaled = input_scaler.transform(next_raw)  # shape (1, n_features)
        else:
            next_scaled = next_raw
            
        # 5) Substitui alvo pela PREVISÃO (em espaço escalado do ALVO)
        next_scaled[0, target_col_idx] = yhat_val

        # 6) Atualiza a janela (shift left + append nova linha)
        window = np.vstack((window[1:], next_scaled))
    
        # 7) Atualiza a janela RAW para o próximo passo (para exog_strategy='persistence' funcionar corretamente)
        last_window = np.vstack((last_window[1:], next_raw))

    # 8) Destransforma predições para ESCALA ORIGINAL
    preds_scaled_arr = np.array(preds).reshape(-1, 1)
    if target_scaler is not None:
        ensure_scaler_fitted(target_scaler, preds_scaled_arr, inverse=True)
        preds = target_scaler.inverse_transform(preds_scaled_arr).ravel()
    else: 
        preds = preds_scaled_arr.flatten()

    return preds

def forecast_lstm(
                    df: pd.DataFrame,
                    feature_cols: Sequence[str],
                    target_col: str,
                    model,
                    lookback: int,
                    horizon: int = 90,
                    input_scaler: Optional[MinMaxScaler] = None,
                    target_scaler: Optional[MinMaxScaler] = None,
                    min_train_size: Optional[int] = None,
                    step: int = 1,
                    fit_scalers_strategy: str = "fit_initial",  # "fit_initial" | "fit_expanding" | "load"
                    exog_next_raw_fn: Optional[callable] = None,
                    exog_strategy: str = "persistence",
                    # --- opcionais de cenário e IC ---
                    scenario_deltas: Optional[Dict[str, float]] = None,           # {"cambio": +0.05, ...}
                    elasticidades: Optional[Dict[str, Dict[str, float]]] = None,  # {"cambio": {"boi_real": 0.8}, ...}
                    diffs_window: Optional[int] = None,
                    evaluate_adjusted: bool = False,  # métricas sobre y_pred_adj
                ) -> (pd.DataFrame, Dict[int, Dict[str, Any]]): # type: ignore

    
    df = df.sort_index() if not isinstance(df.index, pd.RangeIndex) else df
    dates = pd.date_range(df['data'].iloc[-1] + timedelta(days=1), periods=horizon, freq='D')
    X_all = df[feature_cols].values.astype(float)     # (n, n_features)
    y_all = df[[target_col]].values.astype(float)       # (n,1)
    n, n_features = X_all.shape
    idx_target = feature_cols.index(target_col) if target_col in feature_cols else None

    H_max = 90 # horizonte máximo de predição
    start = max(min_train_size or lookback, lookback)
    end = n - H_max
    if end <= start:
        raise ValueError("Série muito curta para o(s) horizonte(s) solicitado(s).")

    # --- Estratégia de fit dos scalers (sem vazamento) ---
    if input_scaler is not None:
        if fit_scalers_strategy == "fit_initial":
            input_scaler.fit(X_all[:start])
        elif fit_scalers_strategy == "load":
            ensure_scaler_fitted(target_scaler, y_all[:1], inverse=True)
    
    rows = []
    for t in range(start, end + 1, step):
        # Fit expansivo opcional (adapta ao drift, sem usar futuro)
        if fit_scalers_strategy == "fit_expanding":
            if input_scaler is not None:
                input_scaler.fit(X_all[:t])
            if target_scaler is not None:
                target_scaler.fit(y_all[:t])

        last_window = X_all[t - lookback:t, :]  # (lookback, n_features)

        # Predição iterativa até H_max (uma vez)
        y_pred_full = _iterative_lstm_forecast(
            model=model,
            last_window=last_window,
            horizon=horizon,
            input_scaler=input_scaler,
            target_scaler=target_scaler,
            target_col_idx=idx_target  if idx_target is not None else 0,
            exog_next_raw_fn=exog_next_raw_fn,
            exog_strategy=exog_strategy,
        )  # shape (H_max,)

        # Alvos verdadeiros (ESCALA ORIGINAL)
        y_target_full = y_all[t:t + H_max].ravel()

        # Estima sigma para IC a partir de diffs recentes do alvo
        window_for_sigma = diffs_window if diffs_window is not None else lookback
        start_sigma = max(t - window_for_sigma, 0)
        diffs = pd.Series(y_all[start_sigma:t].ravel()).diff().dropna()
        sigma = float(diffs.std()) if diffs.size > 0 else 0.0
        ci95 = 1.96 * sigma  # banda constante (pode aumentar com lead time, se quiser)

        y_pred = y_pred_full[:horizon]
        y_target = y_target_full[:horizon]

        # Ajuste de cenário (multiplicativo): 1 + Σ beta_i * Δ_i
        adj_factor = 1.0
        if scenario_deltas is not None and elasticidades is not None:
            for var, delta_pct in scenario_deltas.items():
                beta = float(elasticidades.get(var, {}).get(target_col, 0.0))
                adj_factor += beta * float(delta_pct)
        y_pred_adj = y_pred * adj_factor

        lower = (y_pred_adj if evaluate_adjusted else y_pred) - ci95
        upper = (y_pred_adj if evaluate_adjusted else y_pred) + ci95

        rows.append({
                "dates": pd.to_datetime(dates),
                "t_index": df.index[t],
                "horizon": horizon,
                "y_target": y_target.tolist(),
                "y_pred": y_pred.tolist(),
                "y_pred_adj": y_pred_adj.tolist(),
                "ci95": float(ci95),
                "ci_lower": lower.tolist(),
                "ci_upper": upper.tolist(),
                "adj_factor": float(adj_factor),
            })
        
    df_results = pd.DataFrame(rows)

    return df_results

def baseline_naive_last_value(df: pd.DataFrame, horizon: int, target_col: str = 'preco_boi') -> Dict[str, np.ndarray]:
    last_price = df[[target_col]].iloc[-1,:]
    fcst = np.full((horizon,1), last_price.values)
    future_dates = pd.date_range(df['data'].iloc[-1] + timedelta(days=1), periods=horizon, freq='D')
    return {'dates': future_dates, 'y_pred': fcst}


def plot_multiple_anchors(df_results, horizon, n=5, step=10):
    df_h = df_results[df_results["horizon"] == horizon].iloc[::step]  # subamostra
    n = min(n, len(df_h))
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.5*n), sharex=True)

    for i in range(n):
        row = df_h.iloc[i]
        #tgt = np.array(row["y_target"])
        pred = np.array(row["y_pred"])
        #naive = np.array(row["y_naive"])
        #H = len(tgt)
        H = horizon

        ax = axes[i] if n > 1 else axes
        #ax.plot(range(1, H+1), tgt, label="Alvo", color="black")
        ax.plot(range(1, H+1), pred, label="LSTM", color="tab:blue")
        #
        # ax.plot(range(1, H+1), naive, label="Naive", color="tab:orange", linestyle="--")
        ax.set_title(f"Âncora: {pd.to_datetime(row['t_index']).date()} | H={horizon}")
        ax.set_ylabel("Preço")
        ax.legend()

    axes[-1].set_xlabel("Lead time")
    plt.tight_layout()
    plt.show()

def plot_timeseries(df: pd.DataFrame, fcst: Dict[str, np.ndarray], baseline: Dict[str, np.ndarray], title: str = 'Preco do boi: historico e previsao') -> io.BytesIO:
    # --- History (ensure correct types) ---
    df_plot = df.copy()
    df_plot['data'] = pd.to_datetime(df_plot['data'], errors='coerce')
    y_hist = pd.to_numeric(df_plot['boi_real'], errors='coerce')
    hist_mask = df_plot['data'].notna() & y_hist.notna()

    # --- Latest forecast row ---
    # dates_raw: vector of future dates (strings, datetime64, or DatetimeIndex)
    dates_raw = fcst['dates'].iloc[-1]
    y_pred_raw = fcst['y_pred'].iloc[-1]

    # Normalize to Series
    dates = pd.Series(dates_raw)
    y_pred = pd.Series(y_pred_raw)

    # Convert to datetime and numeric; drop NaT and NaN coherently
    dates = pd.to_datetime(dates, errors='coerce')
    y_pred = pd.to_numeric(y_pred, errors='coerce')

 
    mask = dates.notna() & y_pred.notna()
    dates = dates[mask].reset_index(drop=True)
    y_pred = y_pred[mask].reset_index(drop=True)
   
    if len(dates) != len(y_pred):
        raise ValueError(f"[forecast] Length mismatch: dates={len(dates)} vs y_pred={len(y_pred)}")

    
    fig, ax = plt.subplots(figsize=(10, 5))

    # History
    ax.plot(
        df_plot.loc[hist_mask, 'data'].values,
        y_hist.loc[hist_mask].values,
        label='Histórico',
        color='#1f77b4'
    )


    # Forecast (modelo)
    ax.plot(dates.values, y_pred.values, label='Previsão (modelo)', color='#e377c2')


    # Baseline (último valor)
    #ax.plot(base_dates.values, base_pred.values, label='Baseline (último valor)', color='#ff7f0e', linestyle='--')


    # Date formatting
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

    ax.set_title(title)
    ax.set_xlabel('Data')
    ax.set_ylabel('Preço (R$)')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()


    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    return buf

def plot_error_distribution(df: pd.DataFrame) -> io.BytesIO:
    diffs = df[['boi_real']].diff().dropna()
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
    cols = df.columns.to_list()
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

    LOOKBACK = 180
    #HORIZONS = (30, 60, 90)
    MIN_TRAIN = 600

    model = keras_load_model(MODELS_DIR + "/lstm_model.keras")
    input_scaler = joblib.load(MODELS_DIR + "/input_scaler.gz")
    target_scaler = joblib.load(MODELS_DIR + "/target_scaler.gz")
    fit_strategy = "load"

    st.set_page_config(page_title='Previsão do Preço do Boi de Corte', layout='wide')
    st.markdown('<style> .stMetric {background:#f9fafb;padding:10px;border-radius:8px;} </style>', unsafe_allow_html=True)
    st.title('Previsão do Preço do Boi de Corte')
    st.caption('Prototipo visual com recursos avancados — substitua pelo seu modelo LSTM e dados reais.')

    #uploaded = "C:/Users/rmoreir/OneDrive/Documentos/PROJETOS_ML_DATA_SCIENCE/PROJETO_AGRO_GOIAS/data/df_feat_selected.csv"
    #uploaded_not_norm = "C:/Users/rmoreir/OneDrive/Documentos/PROJETOS_ML_DATA_SCIENCE/PROJETO_AGRO_GOIAS/data/df_selected.csv"
    uploaded = "D:/Users/Ricar/OneDrive/Documentos/PROJETOS_ML_DATA_SCIENCE/PROJETO_AGRO_GOIAS/data/df_feat_selected.csv"
    uploaded_not_norm = "D:/Users/Ricar/OneDrive/Documentos/PROJETOS_ML_DATA_SCIENCE/PROJETO_AGRO_GOIAS/data/df_selected.csv"
    deltas = []
    with st.sidebar:
        st.header('Configuracoes')
        horizonte = st.select_slider('Horizonte de previsao (dias)', options=[30, 60, 90], value=90)
        st.divider()
        st.subheader('Cenarios (Delta% em relacao ao ultimo valor)')
        #delta_covid = st.slider('COVID (1/0)', 0.0, 1.0, 0.0, step=1.0)
        #deltas.append(delta_covid)
        delta_temp = st.slider('Temperatura (Delta%)', 0.0, 75.0, 0.0, step=1.0)
        deltas.append(delta_temp)
        delta_ipca = st.slider('IPCA (Delta%)', 2.0, 11.0, 0.0, step=0.5)
        deltas.append(delta_ipca)
        delta_selic = st.slider('SELIC (Delta%)', 2.0, 15.0, 0.0, step=0.5)
        deltas.append(delta_selic)
        delta_PIB = st.slider('PIB (Delta%)', -6.0, 17.0, 0.0, step=0.5)
        deltas.append(delta_PIB)
        delta_cme = st.slider('CME (Delta%)', -10.0, 20.0, 0.0, step=0.5)
        deltas.append(delta_cme)
        delta_soja_real = st.slider('Soja (Delta% R$)', -10.0, 40.0, 0.0, step=0.5)
        deltas.append(delta_soja_real)
        delta_soja_dolar = st.slider('Soja (Delta% $)', -10.0, 40.0, 0.0, step=0.5)
        deltas.append(delta_soja_dolar)
        delta_soja_futuro = st.slider('Soja Futuro (Delta%)', -10.0, 40.0, 0.0, step=0.5)
        deltas.append(delta_soja_futuro)
        delta_milho_futuro = st.slider('Milho Futuro (Delta%)', -10.0, 40.0, 0.0, step=0.5)
        deltas.append(delta_milho_futuro)
        delta_milho_real = st.slider('Milho (Delta% R$)', -10.0, 40.0, 0.0, step=0.5)
        deltas.append(delta_milho_real)
        delta_milho_dolar = st.slider('Milho (Delta% $)', -10.0, 40.0, 0.0, step=0.5)
        deltas.append(delta_milho_dolar)
        delta_taxa = st.slider('Taxa (Delta%)', -10.0, 40.0, 0.0, step=0.5)
        deltas.append(delta_taxa)
        #delta_el_nino = st.slider('El Niño ()', -1.0, 1.0, 0.1, step=0.5)
        #deltas.append(delta_el_nino)
        st.info('Elasticidades')
        st.divider()
        st.subheader('Modelo LSTM')
        timesteps = st.number_input('Timesteps (janela de entrada)', min_value=MIN_TRAIN, max_value=MIN_TRAIN*2, value=MIN_TRAIN, step=100)

    try:
        df_not_norm = load_uploaded_data(uploaded_not_norm)
        df_not_norm = df_not_norm.drop('Unnamed: 0', axis=1)
        df_not_norm_ = ensure_datetime_sorted(df_not_norm)[timesteps:]
    except Exception as e:
        st.error('Erro ao carregar arquivo: ' + str(e))
        st.stop()

    try:        
        # 2) Rodar walk-forward
        feature_cols = ['boi_real', 'boi_dolar','CovidPeriodFlag', 'TEMPERATURA', 'ipca', 'selic', 'PIB', 'cme_valor', 'soja_real', 'soja_dolar', 'soja_futuro', 
                    'milho_futuro', 'milho_real', 'milho_dolar', 'taxa', 'el_nino_encoded']
    
        target_col = 'boi_real'

        X_row = np.expand_dims(_build_initial_window(df_not_norm_, feature_cols, timesteps), axis=0)
        elasticidades = marginal_elasticities_batch(model,
                                                    X_row,
                                                    timesteps,
                                                    target_col,
                                                    feature_cols)
        cenarios = {}
        for delta, col in zip(deltas, feature_cols[3:]):
            chave = col
            valor = delta / 100.0  # converter para pct
            cenarios[chave] = valor

        fcst = forecast_lstm(
                            df=df_not_norm_,
                            feature_cols=feature_cols,
                            target_col=target_col,
                            model=model,
                            lookback=LOOKBACK,
                            horizon=horizonte,
                            input_scaler=input_scaler,
                            target_scaler=target_scaler,
                            min_train_size=MIN_TRAIN,
                            step=7,
                            fit_scalers_strategy=fit_strategy,
                            exog_next_raw_fn=None,      # se tiver caminhos de exógenas futuras, passe função aqui
                            exog_strategy="persistence",
                            scenario_deltas=cenarios,       # ou {"cambio": 0.05, ...}
                            elasticidades=elasticidades,         # ou {"cambio": {"boi_real": 0.8}, ...}
                            diffs_window=LOOKBACK,
                            evaluate_adjusted=False,    # métricas sobre previsão pura do modelo
                        )
        base = baseline_naive_last_value(df_not_norm_, horizonte, target_col)

    except Exception as e:
        st.error(str(e))
        return e

    preco_atual = float(df_not_norm['boi_real'].iloc[-1])    

    preco_prev_medio = (pd.Series(fcst['y_pred'])).explode().astype(float).mean()
    tendencia = 'Alta' if preco_prev_medio > preco_atual else 'Baixa'
    ic95_amplitude = float(np.mean(fcst['upper'] - fcst['lower'])) if 'upper' in fcst else 0.0

    st.subheader('Visao Geral')

    # CSS: fundo e cores das métricas
    st.markdown("""
    <style>
    .custom-box {
        background-color: #e8f0fe; /* 🔵 azul claro; troque pela cor que preferir */
        padding: 12px;
        border-radius: 10px;
        /* opcional: sombra para destacar */
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    /* Cores dos textos dentro do st.metric */
    div[data-testid="stMetricLabel"] { color: #111111 !important; }
    div[data-testid="stMetricValue"] { color: #000000 !important; }
    div[data-testid="stMetricDelta"] { color: #333333 !important; }
    </style>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Preco atual (R$)', '{:,.2f}'.format(preco_atual))
    c2.metric('Previsao media {}d (R$)'.format(horizonte), '{:,.2f}'.format(preco_prev_medio))
    c3.metric('Tendencia', tendencia)
    c4.metric('Amplitude media IC 95% (R$)', '{:,.2f}'.format(ic95_amplitude))

    fig_ts_buf = plot_timeseries(df_not_norm, fcst, base)
    st.image(fig_ts_buf, caption='Historico e previsao com intervalo de confianca')

    tab1, tab2, tab3 = st.tabs(['Explicabilidade', 'Avancado', 'Relatorio'])

    with tab1:
        st.subheader('Explicabilidade do modelo')
        shap_buf = None
        if shap is not None:
            try:
                X = df_not_norm_[feature_cols].tail(200).values.astype(np.float32)
                # Demo apenas: gere shap_values reais no seu pipeline
                shap_values = np.random.normal(0, 1, size=(X.shape[0]-min(60, X.shape[0]//2), X.shape[1]))
                shap_buf = plot_shap_bar(shap_values, feature_cols)
                st.image(shap_buf, caption='Importancia de features (demo)')
                st.warning('Demo SHAP: substitua pela chamada correta do Kernel/DeepExplainer do seu modelo.')
            except Exception as e:
                st.info('Nao foi possivel calcular SHAP automaticamente. Motivo: ' + str(e))
        if shap_buf is None:
            st.subheader('Correlacao entre variaveis (proxy)')
            fig_corr_buf = plot_correlations(df_not_norm)
            st.image(fig_corr_buf, caption='Mapa de correlacao (Pearson)')

    with tab2:
        st.subheader('Recursos avancados')
        with st.expander('Backtests avancados (rolling, multiplos horizontes)'):
            bt2 = backtest_rolling(df_not_norm_, horizon_list=[30,60,90], train_min=365, step=30, target_cols=target_col)
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
            for dt, p in zip(fcst['dates'].iloc[-1], fcst['y_pred'].iloc[-1]):
                change_pct = (p - preco_atual) / max(preco_atual, 1e-8) * 100.0
                if abs(change_pct) >= threshold_pct:
                    direction = 'Alta' if change_pct > 0 else 'Baixa'
                    alerts.append({'data': dt.date(), 'variacao_%': np.round(change_pct,2), 'direcao': direction, 'preco_previsto': round(float(p),2)})
            if alerts:
                st.dataframe(pd.DataFrame(alerts))
            else:
                st.info('Nenhum alerta para o limiar atual.')

    with tab3:
        st.subheader('Exportar')
        kpis = {
            'Preco atual (R$)': '{:,.2f}'.format(preco_atual),
            'Previsao media {}d (R$)'.format(horizonte): '{:,.2f}'.format(preco_prev_medio),
            'Tendencia': tendencia,
            'Amplitude media IC 95% (R$)': '{:,.2f}'.format(ic95_amplitude),
        }
        fig_err_buf = plot_error_distribution(df_not_norm_)
        fig_corr_buf = plot_correlations(df_not_norm_)
        shap_demo_buf = None
        pdf_bytes = build_pdf(kpis, fig_ts_buf, fig_err_buf, fig_corr_buf, shap_demo_buf)
        st.download_button('Baixar relatorio (PDF)', data=pdf_bytes, file_name='relatorio_preco_boi_{}d.pdf'.format(horizonte), mime='application/pdf')
        df_fcst = pd.DataFrame({'data': fcst['dates'], 'previsao': fcst['y_pred']})
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