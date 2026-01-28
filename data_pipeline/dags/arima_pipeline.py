# -*- coding: utf-8 -*-
"""
ARIMA/SARIMA Pipeline Completo
- Carrega série temporal de CSV (date_col, value_col)
- Testa estacionariedade (ADF, KPSS) e sugere d (e D se sazonal)
- Plota ACF/PACF
- Faz grid search por AIC para ARIMA ou SARIMA
- Ajusta melhor modelo, valida em holdout e prevê horizonte futuro
Autor: você :)
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from math import sqrt
from typing import Tuple, Optional, Dict, Any, List

from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from scipy.stats import shapiro


# =========================
# Utilidades
# =========================

def load_series(
    csv_path: str,
    date_col: Optional[str] = None,
    value_col: Optional[str] = None,
    sep: str = ",",
    decimal: str = ".",
    tz_localize: Optional[str] = None
) -> pd.Series:
    """Carrega a série temporal de um CSV.
    - csv_path: caminho do arquivo
    - date_col: nome da coluna de data (se None, usa a 1ª coluna)
    - value_col: nome da coluna do valor (se None, usa a 2ª coluna)
    - sep, decimal: separador e decimal do CSV
    - tz_localize: ex. 'America/Sao_Paulo' (opcional)
    Retorna: pd.Series indexada por datetime
    """
    df = pd.read_csv(csv_path, sep=sep, decimal=decimal)
    if date_col is None:
        date_col = df.columns[0]
    if value_col is None:
        # procura a primeira coluna numérica diferente de date_col
        num_cols = [c for c in df.columns if c != date_col and pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            raise ValueError("Não encontrei coluna numérica para os valores. Informe --value-col.")
        value_col = num_cols[0]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[date_col, value_col]).copy()
    df = df.sort_values(date_col)
    if tz_localize:
        df[date_col] = df[date_col].dt.tz_localize(tz_localize, nonexistent="shift_forward", ambiguous="NaT")
    s = pd.Series(df[value_col].values, index=df[date_col].values).astype(float)
    s = s.asfreq(pd.infer_freq(s.index), method=None) if pd.infer_freq(s.index) else s
    s.name = value_col
    return s


def stationarity_tests(x: pd.Series, alpha: float = 0.05) -> Dict[str, Any]:
    """Executa ADF e KPSS e retorna métricas e interpretações."""
    results = {}

    # ADF: H0 = raiz unitária (não estacionária). Rejeitar H0 (p<alpha) indica estacionária.
    adf_stat, adf_p, _, _, adf_crit, _ = adfuller(x.dropna(), autolag="AIC")
    results["adf_stat"] = adf_stat
    results["adf_p"] = adf_p
    results["adf_stationary"] = adf_p < alpha
    results["adf_crit"] = adf_crit

    # KPSS: H0 = estacionária. Rejeitar H0 (p<alpha) indica não estacionária.
    try:
        kpss_stat, kpss_p, _, kpss_crit = kpss(x.dropna(), regression="c", nlags="auto")
    except Exception:
        # KPSS pode falhar em séries muito curtas; trata exceção
        kpss_stat, kpss_p, kpss_crit = np.nan, np.nan, {}
    results["kpss_stat"] = kpss_stat
    results["kpss_p"] = kpss_p
    results["kpss_stationary"] = (kpss_p > alpha) if pd.notna(kpss_p) else None
    results["kpss_crit"] = kpss_crit

    return results


def suggest_d(x: pd.Series, max_d: int = 2, alpha: float = 0.05) -> int:
    """Sugere o número de diferenciações d usando ADF+KPSS."""
    d = 0
    xt = x.copy()
    for _ in range(max_d + 1):
        tests = stationarity_tests(xt, alpha=alpha)
        adf_ok = tests["adf_stationary"]
        kpss_ok = tests["kpss_stationary"] if tests["kpss_stationary"] is not None else adf_ok
        if adf_ok and kpss_ok:
            return d
        xt = xt.diff().dropna()
        d += 1
        if d > max_d:
            break
    return min(d, max_d)


def suggest_D_seasonal(x: pd.Series, s: int, max_D: int = 1, alpha: float = 0.05) -> int:
    """Sugere D (diferença sazonal) para período s usando ADF+KPSS após diff sazonal."""
    if s <= 1:
        return 0
    D = 0
    xt = x.copy()
    for _ in range(max_D + 1):
        tests = stationarity_tests(xt, alpha=alpha)
        adf_ok = tests["adf_stationary"]
        kpss_ok = tests["kpss_stationary"] if tests["kpss_stationary"] is not None else adf_ok
        if adf_ok and kpss_ok:
            return D
        xt = xt.diff(s).dropna()
        D += 1
        if D > max_D:
            break
    return min(D, max_D)


def significant_lags(y: pd.Series, nlags: int = 40, alpha: float = 0.05) -> Tuple[List[int], List[int]]:
    """Retorna lags significativos (ACF e PACF) via limiar ~ N(0, 1/sqrt(N))."""
    y = y.dropna()
    N = len(y)
    if N < 10:
        return [], []
    thr = 1.96 / np.sqrt(N)
    acf_vals = acf(y, nlags=nlags, fft=True)
    pacf_vals = pacf(y, nlags=nlags, method="ywm")
    sig_acf = [lag for lag in range(1, len(acf_vals)) if abs(acf_vals[lag]) > thr]
    sig_pacf = [lag for lag in range(1, len(pacf_vals)) if abs(pacf_vals[lag]) > thr]
    return sig_acf, sig_pacf


def train_test_split_series(s: pd.Series, test_size: Optional[int] = None, seasonal_period: int = 1) -> Tuple[pd.Series, pd.Series]:
    """Divide em treino e teste (por padrão 20% ou 2 sazonais, o que for maior)."""
    n = len(s.dropna())
    if test_size is None:
        test_size = max( int(round(0.2 * n)), seasonal_period * 2 )
    test_size = min(test_size, n // 2 if n >= 10 else max(1, n - 1))
    train = s.iloc[:-test_size]
    test = s.iloc[-test_size:]
    return train, test


def model_fit_aic(
    y: pd.Series,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
    enforce_stationarity: bool = False,
    enforce_invertibility: bool = False
):
    """Tenta ajustar SARIMAX e retorna (modelo, resultado, AIC)."""
    try:
        model = SARIMAX(
            y,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility
        )
        res = model.fit(disp=False)
        return model, res, res.aic
    except Exception:
        return None, None, np.inf


def grid_search_arima(
    y: pd.Series,
    d: int,
    p_max: int = 5,
    q_max: int = 5,
    seasonal_period: int = 1,
    D: int = 0,
    P_max: int = 2,
    Q_max: int = 2
) -> Dict[str, Any]:
    """Busca por AIC em ARIMA (se s=1) ou SARIMA (se s>1) e retorna melhor encontrado."""
    best = {"aic": np.inf, "order": None, "seasonal_order": None, "res": None}
    if seasonal_period <= 1:
        for p in range(p_max + 1):
            for q in range(q_max + 1):
                _, res, aic = model_fit_aic(y, (p, d, q))
                if aic < best["aic"]:
                    best.update({"aic": aic, "order": (p, d, q), "seasonal_order": (0, 0, 0, 0), "res": res})
    else:
        for p in range(p_max + 1):
            for q in range(q_max + 1):
                for P in range(P_max + 1):
                    for Q in range(Q_max + 1):
                        _, res, aic = model_fit_aic(y, (p, d, q), (P, D, Q, seasonal_period))
                        if aic < best["aic"]:
                            best.update({"aic": aic, "order": (p, d, q), "seasonal_order": (P, D, Q, seasonal_period), "res": res})
    return best


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calcula métricas padrão."""
    err = y_pred - y_true
    mae = np.mean(np.abs(err))
    rmse = sqrt(np.mean(err**2))
    mape = np.mean(np.abs(err / np.where(y_true == 0, np.nan, y_true))) * 100.0
    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE_%": float(np.nan_to_num(mape))}


def plot_all(
    s: pd.Series,
    d: int,
    D: int,
    s_period: int,
    train: pd.Series,
    test: pd.Series,
    best: Dict[str, Any],
    forecast_steps: int,
    title_prefix: str = "ARIMA Pipeline"
):
    """Gera gráficos: série, ACF/PACF, resíduos e previsões."""
    sns.set(style="whitegrid", context="talk")

    # Série e diferenciações
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)
    s.plot(ax=axes[0], color="#1f77b4", lw=1.5)
    axes[0].set_title(f"{title_prefix} - Série original")
    axes[0].set_xlabel("Tempo"); axes[0].set_ylabel(s.name or "Valor")

    sd = s.diff(d).dropna() if d > 0 else s.dropna()
    (sd if d > 0 else s).plot(ax=axes[1], color="#ff7f0e", lw=1.2)
    axes[1].set_title(f"Série {'diferenciada' if d>0 else 'original'} (d={d})")

    if s_period > 1 and D > 0:
        sds = s.diff(s_period).dropna()
        (sds).plot(ax=axes[2], color="#2ca02c", lw=1.2)
        axes[2].set_title(f"Diferença sazonal (D={D}, s={s_period})")
    else:
        axes[2].axis("off")
    plt.tight_layout()

    # ACF / PACF
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_acf(sd, lags=min(40, len(sd)-1), alpha=0.05, ax=axes[0])
    plot_pacf(sd, lags=min(40, len(sd)-1), alpha=0.05, ax=axes[1], method="ywm")
    axes[0].set_title("ACF (após differencing)"); axes[1].set_title("PACF (após differencing)")
    plt.tight_layout()

    # Ajuste & Previsão
    res = best["res"]
    order = best["order"]
    seasonal_order = best["seasonal_order"]
    print(f"\n>> Melhor modelo: ARIMA{order} SARIMA{seasonal_order} | AIC={best['aic']:.2f}")

    # Previsão no teste
    steps = len(test)
    pred = res.get_forecast(steps=steps)
    pred_mean = pred.predicted_mean
    pred_ci = pred.conf_int()

    # Plot train/test + previsões
    fig, ax = plt.subplots(figsize=(14, 6))
    train.plot(ax=ax, label="Treino")
    test.plot(ax=ax, label="Teste", color="#d62728")
    pred_mean.index = test.index
    pred_ci.index = test.index
    pred_mean.plot(ax=ax, label="Previsão (teste)", color="#2ca02c")
    ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color="#2ca02c", alpha=0.2)
    ax.set_title(f"Previsão no conjunto de teste - ARIMA{order} SARIMA{seasonal_order}")
    ax.legend()
    plt.tight_layout()

    # Gráficos de resíduos
    resid = res.resid.dropna()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(resid, color="#9467bd"); axes[0, 0].set_title("Resíduos")
    sns.histplot(resid, kde=True, ax=axes[0, 1], color="#8c564b"); axes[0, 1].set_title("Histograma dos resíduos")
    plot_acf(resid, lags=min(40, len(resid)-1), alpha=0.05, ax=axes[1, 0]); axes[1, 0].set_title("ACF dos resíduos")
    from statsmodels.graphics.gofplots import qqplot
    qqplot(resid, line="s", ax=axes[1, 1]); axes[1, 1].set_title("Q-Q Plot dos resíduos")
    plt.tight_layout()

    # Ljung-Box e Normalidade
    lb = acorr_ljungbox(resid, lags=[10], return_df=True)
    jb_stat, jb_p, _, _ = jarque_bera(resid)
    sh_stat, sh_p = shapiro(resid) if len(resid) <= 5000 else (np.nan, np.nan)
    print("\n>> Diagnósticos dos resíduos:")
    print(lb)
    print(f"Jarque-Bera p-value: {jb_p:.4f} | Shapiro-Wilk p-value: {sh_p if not np.isnan(sh_p) else 'N/A'}")

    # Forecast futuro
    if forecast_steps and forecast_steps > 0:
        fut = res.get_forecast(steps=forecast_steps)
        fut_mean = fut.predicted_mean
        fut_ci = fut.conf_int()

        fig, ax = plt.subplots(figsize=(14, 6))
        s.plot(ax=ax, label="Histórico")
        fut_mean.plot(ax=ax, label="Forecast", color="#ff7f0e")
        ax.fill_between(fut_ci.index, fut_ci.iloc[:, 0], fut_ci.iloc[:, 1], color="#ff7f0e", alpha=0.2)
        ax.set_title(f"Forecast {forecast_steps} passos à frente")
        ax.legend()
        plt.tight_layout()

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Pipeline ARIMA/SARIMA completo")
    parser.add_argument("--csv", type=str, required=True, help="Caminho do CSV")
    parser.add_argument("--date-col", type=str, default=None, help="Nome da coluna de data (padrão: primeira)")
    parser.add_argument("--value-col", type=str, default=None, help="Nome da coluna de valor (padrão: primeira numérica)")
    parser.add_argument("--sep", type=str, default=",", help="Separador CSV (padrão ',')")
    parser.add_argument("--decimal", type=str, default=".", help="Separador decimal (padrão '.')")
    parser.add_argument("--seasonal-period", type=int, default=1, help="Período sazonal s (ex.: 12 mensal, 7 diário)")
    parser.add_argument("--max-p", type=int, default=5, help="p máximo")
    parser.add_argument("--max-q", type=int, default=5, help="q máximo")
    parser.add_argument("--max-P", type=int, default=2, help="P máximo (sazonal)")
    parser.add_argument("--max-Q", type=int, default=2, help="Q máximo (sazonal)")
    parser.add_argument("--max-d", type=int, default=2, help="d máximo")
    parser.add_argument("--max-D", type=int, default=1, help="D máximo sazonal")
    parser.add_argument("--alpha", type=float, default=0.05, help="Nível de signif. p/ testes (ADF/KPSS)")
    parser.add_argument("--test-size", type=int, default=None, help="Tamanho do conjunto de teste (passos)")
    parser.add_argument("--forecast-steps", type=int, default=12, help="Horizonte de forecast futuro")
    parser.add_argument("--tz", type=str, default=None, help="Timezone p/ localize (ex.: 'America/Sao_Paulo')")
    args = parser.parse_args()

    # 1) Carrega série
    s = load_series(
        args.csv,
        date_col=args.date_col,
        value_col=args.value_col,
        sep=args.sep,
        decimal=args.decimal,
        tz_localize=args.tz
    )
    print(f"Série carregada. Período inferido: {pd.infer_freq(s.index)} | Observações: {len(s)}")

    # 2) Testes de estacionariedade e sugestão de d/D
    d = suggest_d(s, max_d=args.max_d, alpha=args.alpha)
    D = suggest_D_seasonal(s, s=args.seasonal_period, max_D=args.max_D, alpha=args.alpha) if args.seasonal_period > 1 else 0
    print(f"Sugestão de diferenciação: d={d} | D={D} (s={args.seasonal_period})")

    tests0 = stationarity_tests(s, alpha=args.alpha)
    print(f"ADF p={tests0['adf_p']:.4f} (stationary={tests0['adf_stationary']}) | "
          f"KPSS p={tests0['kpss_p'] if tests0['kpss_p'] is not None else 'N/A'} "
          f"(stationary={tests0['kpss_stationary']})")

    # 3) ACF/PACF: lags significativos para sugerir ranges
    sd = s.diff(d).dropna() if d > 0 else s.dropna()
    sig_acf, sig_pacf = significant_lags(sd, nlags=40)
    p_max = min(args.max_p, max(sig_pacf, default=2))
    q_max = min(args.max_q, max(sig_acf, default=2))
    print(f"Sugestão de busca: p in [0..{p_max}], q in [0..{q_max}]")

    # 4) Split train/test
    train, test = train_test_split_series(s, test_size=args.test_size, seasonal_period=args.seasonal_period)
    print(f"Split: treino={len(train)} | teste={len(test)}")

    # 5) Grid search (AIC)
    best = grid_search_arima(
        y=train,
        d=d,
        p_max=p_max,
        q_max=q_max,
        seasonal_period=args.seasonal_period,
        D=D,
        P_max=args.max_P,
        Q_max=args.max_Q
    )
    if best["res"] is None:
        raise RuntimeError("Nenhum modelo convergiu. Tente reduzir ranges ou verificar a série.")

    # 6) Avaliar no conjunto de teste
    res = best["res"]
    steps = len(test)
    fc = res.get_forecast(steps=steps)
    pred = fc.predicted_mean
    pred.index = test.index
    perf = metrics(test.values, pred.values)
    print(f"\nMétricas no teste: MAE={perf['MAE']:.4f} | RMSE={perf['RMSE']:.4f} | MAPE={perf['MAPE_%']:.2f}%")

    # 7) Plots e diagnósticos
    plot_all(
        s=s,
        d=best["order"][1],
        D=best["seasonal_order"][1] if best["seasonal_order"] else 0,
        s_period=args.seasonal_period,
        train=train,
        test=test,
        best=best,
        forecast_steps=args.forecast_steps
    )


if __name__ == "__main__":
    main()