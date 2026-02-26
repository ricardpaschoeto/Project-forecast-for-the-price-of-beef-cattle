from __future__ import annotations

import warnings
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
import os

from pmdarima.arima import auto_arima
from scipy.stats import shapiro
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from statsmodels.tsa.stattools import adfuller

from pathlib import Path

caminho = Path(os.path.abspath(__file__))
root_dir = caminho.parent.parent
# transformer_dir = os.path.join(root_dir,'operators', 'power_transformer.joblib')
# scale_dir = os.path.join(root_dir,'operators', 'scaler.joblib')

# =========================
# Configurações e dataclasses
# =========================

@dataclass
class NormalizationReport:
    column: str
    is_constant: bool
    adf_pvalue_initial: Optional[float]
    stationary_initial: Optional[bool]
    differencing_applied: int
    adf_pvalue_final: Optional[float]
    stationary_final: Optional[bool]
    shapiro_pvalue: Optional[float]
    normality: Optional[bool]
    transformer: str  # "minmax" ou "yeo-johnson+minmax"
    errors: Tuple[str, ...]



@dataclass
class PipelineConfig:
    scaler_range: Tuple[float, float] = (0.0, 1.0)
    adf_alpha: float = 0.05
    max_diff_order: int = 6
    interpolate: bool = True  # preencher após diffs
    interpolation_method: str = "linear"
    shapiro_alpha: float = 0.05
    shapiro_sample_cap: int = 5000  # amostra para o teste se a série for muito grande


# =========================
# Paths utilitários
# =========================

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _col_artifact_paths(artifact_dir: Path, col: str) -> Tuple[Path, Path, Path]:

    """
    Retorna caminhos para (pt.joblib, scaler.joblib, meta.json) por coluna.
    """
    pt_path = artifact_dir / f"{col}_pt.joblib"
    scaler_path = artifact_dir / f"{col}_scaler.joblib"
    meta_path = artifact_dir / f"{col}_meta.json"

    return pt_path, scaler_path, meta_path


# =========================
# Testes e transformações básicas
# =========================

def _is_constant(series: pd.Series) -> bool:
    return series.nunique(dropna=False) <= 1


def _safe_adfuller(series: pd.Series) -> Optional[float]:
    """Retorna p-valor do ADF, ou None se falhar."""
    try:
        result = adfuller(series.astype(float).values)
        return float(result[1])
    except Exception as e:
        warnings.warn(f"ADF falhou: {e}")
        return None

def _make_stationary_train(
    series: pd.Series,
    cfg: PipelineConfig
) -> Tuple[pd.Series, int, Optional[float], Optional[bool]]:
    """
    Tenta tornar a série estacionária via differencing incremental.
    Retorna (serie_stationary, ordem_diff_aplicada, p_final, flag_estacionaria).
    """
    p_initial = _safe_adfuller(series)
    stationary_initial = (p_initial is not None) and (p_initial <= cfg.adf_alpha)

    if stationary_initial:
        return series, 0, p_initial, True

    # Diferenciação incremental
    s = series.copy()
    for d in range(1, cfg.max_diff_order + 1):
        s = s.diff(1)  # aplica diferença de ordem 1 a cada passo
        if cfg.interpolate:
            try:
                s = s.interpolate(method=cfg.interpolation_method, 
                                  limit_direction="both")
            except Exception as e:
                warnings.warn(f"Interpolação falhou (d={d}): {e}")

        p = _safe_adfuller(s)
        if p is not None and p <= cfg.adf_alpha:
            return s, d, p, True

    # Não estacionária mesmo após máximo de diferenças
    p_final = _safe_adfuller(s)
    is_stat = (p_final is not None) and (p_final <= cfg.adf_alpha)
    return s, cfg.max_diff_order, p_final, is_stat


def _difference_with_history(
    train_tail_original: Optional[pd.Series],
    test_series: pd.Series,
    d: int,
    cfg: PipelineConfig
) -> pd.Series:
    """
    Aplica differencing de ordem d no TESTE usando a "history" do TREINO como contexto.
    Necessário para evitar leakage e manter consistência.

    - Para d=0: retorna test_series tal qual.
    - Para d>0: concatena os últimos d valores originais do treino com o teste e aplica
      diff(1) repetidamente d vezes; depois descarta a parte "history" e mantém apenas o teste.
    """
    # TODO

def _fit_auto_arima(series: pd.Series, stationary_hint: Optional[bool], cfg: PipelineConfig):
    """Treina ARIMA e retorna resíduos, ou None se falhar."""
    try:
        model = auto_arima(
            series.astype(float).values,
            seasonal=cfg.seasonal,
            m=cfg.seasonal_period,
            stepwise=cfg.stepwise,
            trace=False,
            error_action="ignore",
            stationary=stationary_hint,
            suppress_warnings=True,
        )
        # auto_arima já faz fit internamente; .fit(series) não é necessário.
        resid = model.resid()
        return resid
    except Exception as e:
        warnings.warn(f"Falha ao ajustar ARIMA: {e}")
        return None


def _normality_transform(col: str,
                         isTrain: bool, 
                         series: pd.Series, 
                         scaler: MinMaxScaler, 
                         pt: PowerTransformer, 
                         shapiro_alpha: float) -> Tuple[np.ndarray, Optional[float], Optional[bool], str, Tuple[str, ...]]:
    """
    Aplica MinMax direto se resíduos forem normais; caso contrário Yeo-Johnson + MinMax.
    Retorna (valores_transformados, p_shapiro, normal, nome_transformer, erros).
    """
    errors: list[str] = []

    # Avalia normalidade nos valores atuais da série (não dos resíduos),
    # pois resíduos podem não existir ou serem inadequados quando ARIMA falha.
    try:
        # Shapiro requer vetor 1D; se tamanho > 5000, amostre para evitar erro/performance
        values = series.astype(float).values
        if values.shape[0] > 5000:
            rng = np.random.default_rng(42)
            idx = rng.choice(values.shape[0], 5000, replace=False)
            test_values = values[idx]
        else:
            test_values = values

        _, p_shapiro = shapiro(test_values)
    except Exception as e:
        warnings.warn(f"Shapiro-Wilk falhou: {e}")
        errors.append(f"shapiro_error:{e}")
        p_shapiro = None

    is_normal = (p_shapiro is not None) and (p_shapiro > shapiro_alpha)

    try:
        col_pt_path = Path(root_dir, "operators", f"{col}_pt.joblib")
        col_scaler_path = Path(root_dir, "operators", f"{col}_scaler.joblib")

        if is_normal:
            if isTrain:
                transformed = scaler.fit_transform(values.reshape(-1, 1))
                joblib.dump(pt, col_pt_path)
                joblib.dump(scaler, col_scaler_path)
            else:
                pt = joblib.load(col_pt_path)
                scaler = joblib.load(col_scaler_path)
                transformed = scaler.transform(values.reshape(-1, 1))

            transformer_name = "minmax"
        else:
            if isTrain:
                transformed_serie = pt.fit_transform(values.reshape(-1, 1))
                transformed = scaler.fit_transform(transformed_serie)
            else:
                transformed_serie = pt.transform(values.reshape(-1, 1))
                transformed = scaler.transform(transformed_serie)

            transformer_name = "yeo-johnson+minmax"


    except Exception as e:
        errors.append(f"transform_error:{e}")
        warnings.warn(f"Falha na transformação: {e}")
        transformed = np.full((series.shape[0], 1), np.nan)
        transformer_name = "failed"

    return transformed, p_shapiro, is_normal, transformer_name, tuple(errors)




# =========================
# Pipeline principal
# =========================

def normalize_time_series(
    isTrain: bool,
    df: pd.DataFrame,
    cols: Iterable[str],
    config: Optional[PipelineConfig] = None
) -> Tuple[pd.DataFrame, Dict[str, NormalizationReport]]:
    """
    Normaliza séries históricas em `cols` usando:
      - Teste ADF (estacionaridade)
      - Diferenciação incremental (se necessário)
      - Teste Shapiro-Wilk (normalidade)
      - MinMaxScaler ou Yeo-Johnson + MinMax

    Retorna:
      df_out: DataFrame com colunas transformadas (sufixo "_norm")
      report: dicionário de relatórios por coluna original
    """
    cfg = config or PipelineConfig()
    scaler = MinMaxScaler(feature_range=cfg.scaler_range)
    pt = PowerTransformer(method="yeo-johnson")
    df_out = df.copy()
    reports: Dict[str, NormalizationReport] = {}

    for col in cols:
        errors: list[str] = []

        if col not in df_out.columns:
            warnings.warn(f"Coluna '{col}' não encontrada no DataFrame.")
            reports[col] = NormalizationReport(
                column=col, is_constant=False,
                adf_pvalue_initial=None, stationary_initial=None,
                differencing_applied=0, adf_pvalue_final=None, stationary_final=None,
                shapiro_pvalue=None, normality=None, transformer="missing", errors=tuple(["missing_column"])
            )
            continue

        series = df_out[col].astype(float)

        # Trata colunas constantes
        is_const = _is_constant(series)
        if is_const:
            warnings.warn(f"Coluna '{col}' é constante; será apenas escalonada para o valor único.")
            unique_val = series.iloc[0] if len(series) > 0 else 0.0
            try:
                if isTrain:
                    transformed = scaler.fit_transform(np.full((series.shape[0], 1), unique_val))
                else:
                    transformed = scaler.transform(np.full((series.shape[0], 1), unique_val))
            except Exception as e:
                errors.append(f"scaler_const_error:{e}")
                transformed = np.full((series.shape[0], 1), np.nan)

            df_out[f"{col}_norm"] = transformed.ravel()
            reports[col] = NormalizationReport(
                column=col, is_constant=True,
                adf_pvalue_initial=None, stationary_initial=None,
                differencing_applied=0, adf_pvalue_final=None, stationary_final=None,
                shapiro_pvalue=None, normality=None, transformer="minmax(constant)", errors=tuple(errors)
            )
            continue

        # Estacionaridade
        stationary_series, d_order, p_final, is_stationary = _make_stationary(series, cfg)
        p_initial = _safe_adfuller(series)
        stationary_initial = (p_initial is not None) and (p_initial <= cfg.adf_alpha)

        # (Opcional) Ajuste ARIMA para resíduos — útil para diagnóstico (não obrigatório para transformação)
        resid = _fit_auto_arima(stationary_series, stationary_hint=is_stationary, cfg=cfg)
        if resid is None:
            errors.append("arima_fit_failed")

        # Normalização/transformação
        transformed, p_shapiro, is_normal, transformer_name, err = _normality_transform(isTrain, stationary_series, scaler, pt, cfg.shapiro_alpha)
        errors.extend(list(err))

        # Atribui coluna transformada com sufixo para preservar original
        df_out[f"{col}_norm"] = transformed.ravel()

        reports[col] = NormalizationReport(
            column=col,
            is_constant=False,
            adf_pvalue_initial=(None if p_initial is None else round(p_initial, 4)),
            stationary_initial=stationary_initial,
            differencing_applied=d_order,
            adf_pvalue_final=(None if p_final is None else round(p_final, 4)),
            stationary_final=is_stationary,
            shapiro_pvalue=(None if p_shapiro is None else round(p_shapiro, 4)),
            normality=is_normal,
            transformer=transformer_name,
            errors=tuple(errors)
        )

    return df_out, {k: v for k, v in reports.items()}


# =========================
# Execução direta (exemplo)
# =========================

# def _usage():
    # Exemplo de uso: carregue seu CSV e selecione colunas
    # df = pd.read_csv("seu_arquivo.csv")
    # cols = df.columns.to_list()
    # df_norm, report = normalize_time_series(df, cols)
    # print(df_norm.tail())
    # print({k: asdict(v) for k, v in report.items()})

# if __name__ == "__main__":
#     warnings.filterwarnings("ignore", category=FutureWarning)
#     _usage()