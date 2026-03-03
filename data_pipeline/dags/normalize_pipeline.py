from __future__ import annotations


import json
import os
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, List

import joblib
import numpy as np
import pandas as pd
from pmdarima import auto_arima
from scipy.stats import shapiro
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from statsmodels.tsa.stattools import adfuller

caminho = Path(os.path.abspath(__file__))
root_dir = caminho.parent.parent

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
    if d == 0 or train_tail_original is None or len(train_tail_original) < d:
        s = test_series.copy()
        if cfg.interpolate:
            try:
                s = s.interpolate(method=cfg.interpolation_method, limit_direction="both")
            except Exception as e:
                warnings.warn(f"Interpolação falhou (d={d} sem history): {e}")
        return s

    # Concatena history do treino com teste
    concat_original = pd.concat([train_tail_original, test_series], axis=0)
    s = concat_original.copy()
    for _ in range(d):
        s = s.diff(1)
        if cfg.interpolate:
            try:
                s = s.interpolate(method=cfg.interpolation_method, limit_direction="both")
            except Exception as e:
                warnings.warn(f"Interpolação falhou (d={d} com history): {e}")
    # Retorna apenas a parte do teste (descarta history)
    return s.iloc[d:]

def _shapiro_pvalue(values: np.ndarray, cap: int) -> Optional[float]:

    """Calcula p-valor do teste de Shapiro-Wilk, com amostragem se necessário."""
    try:
        if values.shape[0] > cap:
            rng = np.random.default_rng(42)
            idx = rng.choice(values.shape[0], cap, replace=False)
            test_values = values[idx]
        else:
            test_values = values

        _, p_shapiro = shapiro(test_values)
        return float(p_shapiro)
    except Exception as e:
        warnings.warn(f"Shapiro-Wilk falhou: {e}")
        return None


# =========================
# Transformação / Inversão
# =========================
def _fit_and_transform_column_train(col: str, 
                                    series_stationary: pd.Series, 
                                    cfg: PipelineConfig,
                                    artifact_dir: Path
                                    ) -> Tuple[np.ndarray, str, Optional[float], Optional[bool], Tuple[str, ...]]:


    """
    Treina (fit) e aplica a transformação (PowerTransformer opcional + MinMax) para a coluna,
    salvando artefatos por coluna.
    """
    errors: list[str] = []
    pt = PowerTransformer(method="yeo-johnson")
    scaler = MinMaxScaler(feature_range=cfg.scaler_range)

    values = series_stationary.astype(float).values.reshape(-1, 1)

    # Avalia normalidade para decidir transformação
    p_shapiro = _shapiro_pvalue(values.ravel(), cfg.shapiro_sample_cap)
    is_normal = (p_shapiro is not None) and (p_shapiro > cfg.shapiro_alpha)

    try:
        if is_normal:
            transformed = scaler.fit_transform(values)
            transformer_name = "minmax"
        else:
            intermediate = pt.fit_transform(values)
            transformed = scaler.fit_transform(intermediate)
            transformer_name = "yeo-johnson+minmax"
    except Exception as e:
        errors.append(f"transform_error:{e}")
        warnings.warn(f"Falha na transformação: {e}")
        transformed = np.full((series_stationary.shape[0], 1), np.nan)
        transformer_name = "failed"

    # Salva artefatos
    pt_path, scaler_path, _ = _col_artifact_paths(artifact_dir, col)
    try:
        joblib.dump(pt, pt_path)
        joblib.dump(scaler, scaler_path)
    except Exception as e:
        errors.append(f"artifact_save_error:{e}")
        warnings.warn(f"Falha ao salvar artefatos: {e}")

    return transformed.ravel(), transformer_name, p_shapiro, is_normal, tuple(errors)

def _transform_column_infer(col: str,
                            series_stationary: pd.Series,
                            cfg: PipelineConfig,
                            artifact_dir: Path
                            ) -> Tuple[np.ndarray, str, Optional[float], Optional[bool], Tuple[str, ...]]:
    
    """Aplica transformação usando artefatos salvos do treino. Retorna (valores_transformados, nome_transformer, p_shapiro, is_normal, erros)."""
    errors: list[str] = []
    pt_path, scaler_path, _ = _col_artifact_paths(artifact_dir, col)

    try:
        pt: PowerTransformer = joblib.load(pt_path)
        scaler: MinMaxScaler= joblib.load(scaler_path)
    except Exception as e:
        errors.append(f"artifact_load_error:{e}")
        warnings.warn(f"Falha ao carregar artefatos: {e}")
        values = series_stationary.astype(float).values.reshape(-1, 1)
        return np.full((series_stationary.shape[0], 1), np.nan), "failed", None, None, tuple(errors)

    values = series_stationary.astype(float).values.reshape(-1, 1)
    transformer_name = "yeo-johnson+minmax"  # default se falhar o teste de normalidade
    try:
        if getattr(pt, 'lambdas_', None) is None:
            # Treino decidiu apenas MinMax
            out = scaler.transform(values)
            transformer_name = "minmax"
            p_shapiro = _shapiro_pvalue(values.ravel(), cfg.shapiro_sample_cap)
            is_normal = (p_shapiro is not None) and (p_shapiro > cfg.shapiro_alpha)
        else:
            # Treino aplicou Yeo-Johnson + MinMax
            intermediate = pt.transform(values)
            out = scaler.transform(intermediate)
            transformer_name = "yeo-johnson+minmax"
            p_shapiro = _shapiro_pvalue(values.ravel(), cfg.shapiro_sample_cap)
            is_normal = (p_shapiro is not None) and (p_shapiro > cfg.shapiro_alpha)

    except Exception as e:
        errors.append(f"transform_error:{e}")
        warnings.warn(f"Falha na transformação (infer) para '{col}': {e}")
        out = np.full(values.shape[0], np.nan)

    return out.ravel(), transformer_name, p_shapiro, is_normal, tuple(errors)

def inverse_transform_column(
        df_norm: pd.DataFrame,
        cols: Iterable[str],
        artifact_dir: Optional[str | Path] = None,
        start_values: Optional[pd.DataFrame] = None
        ) -> pd.DataFrame:

    """
    Inverte a normalização das colunas informadas, incluindo reversão do differencing
    quando possível.

    Args:
        df_norm: DataFrame com colunas normalizadas (sufixo "_norm" será buscado).
        cols: colunas originais (sem sufixo).
        artifact_dir: diretório onde os artefatos por coluna foram salvos.
        start_values: DataFrame contendo as últimas observações ORIGINAIS de treino
                      (necessárias para reverter differencing do teste/predição).
                      Deve ter pelo menos 'd' linhas para cada coluna (d = ordem de diff).
    Returns:
        df_inv: DataFrame com colunas invertidas (sem sufixo).
    """
    artifact_dir = Path(artifact_dir or Path("operators"))
    df_inv = pd.DataFrame(index=df_norm.index.copy())

    for col in cols:
        pt_path, scaler_path, meta_path = _col_artifact_paths(artifact_dir, col)

        try:
            with open(meta_path, "rb") as f:
                meta = json.load(f)
        except Exception as e:
            warnings.warn(f"Falha ao carregar meta para '{col}': {e}")
            df_inv[col] = np.nan
            continue

        d = int(meta.get("differencing_applied", 0))
        is_constant = bool(meta.get("is_constant", False))

        if is_constant:
            # Para colunas constantes, a inversão é trivial: basta preencher com o valor único original.
            const_val = meta.get("constant_value", 0.0)
            df_inv[col] = const_val
            continue

        # 1) inverter escalas (scaler/pt)
        try:
            pt: PowerTransformer = joblib.load(pt_path)
            scaler: MinMaxScaler = joblib.load(scaler_path)
        except Exception as e:
            warnings.warn(f"Falha ao carregar artefatos para '{col}': {e}")
            df_inv[col] = np.nan
            continue

        norm_col_name = f"{col}_norm"
        if norm_col_name not in df_norm.columns:
            warnings.warn(f"Coluna normalizada '{norm_col_name}' não encontrada para inversão.")
            df_inv[col] = np.nan
            continue

        norm_values = df_norm[norm_col_name].astype(float).values.reshape(-1, 1)
        try:
            if getattr(pt, 'lambdas_', None) is None:
                # Treino aplicou apenas MinMax
                inv_scaled = scaler.inverse_transform(norm_values)
                base_vals = inv_scaled.ravel()
            else:
                # Treino aplicou Yeo-Johnson + MinMax
                inv_scaled = scaler.inverse_transform(norm_values)
                base_vals = pt.inverse_transform(inv_scaled).ravel()
        except Exception as e:
            warnings.warn(f"Falha na inversão da transformação para '{col}': {e}")
            df_inv[col] = np.nan
            continue

        # 2) inverter differencing (se aplicável)
        if d > 0:
            if start_values is None or col not in start_values.columns:
                warnings.warn(f"Valores iniciais para reversão de differencing de '{col}' não fornecidos.")
                df_inv[col] = base_vals # Retorna os valores transformados sem inverter o diff
            else:
                hist = start_values[col].astype(float)
                # Precisamos dos últimos d valores originais do treino
                if len(hist) < d:
                    warnings.warn(f"Valores iniciais para '{col}' insuficientes para reversão de differencing (d={d}).")
                    df_inv[col] = base_vals # Retorna os valores transformados sem inverter o diff
                else:
                    # reconstrução cumulativa: para d vezes, acumular as diferenças
                    recon = base_vals.copy()
                    # Faça a reconstrução em d passos
                    for order in range(d):
                        # ponto inicial para a reconstrução é o último valor original do treino naquele nível
                        s0 = hist.iloc[-(d - order)]
                        recon = np.r_[s0, np.cumsum(recon)]
                        recon = recon[1:]  # remove o primeiro elemento (s0) que foi adicionado apenas para acumular
                    df_inv[col] = recon
        else:
            df_inv[col] = base_vals

        return df_inv

# =========================
# Pipeline principal
# =========================


def normalize_time_series(
    isTrain: bool,
    df: pd.DataFrame,
    cols: Iterable[str],
    config: Optional[PipelineConfig] = None,
    artifact_dir: Optional[str | Path] = None,
    history_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, NormalizationReport]]:
    """
    Normaliza séries históricas em `cols` usando:
      - Teste ADF (estacionaridade) + differencing incremental (se necessário)
      - (Opcional) Interpolação pós-diff
      - Teste Shapiro-Wilk (normalidade)
      - MinMaxScaler ou Yeo-Johnson + MinMax
    -> Sem data leakage: no treino faz FIT e salva artefatos por coluna;
       no teste carrega e TRANSFORMA usando 'history_df' para differencing.

    Args:
        isTrain: True (ajusta e salva) | False (apenas transforma, carregando artefatos).
        df: DataFrame com as colunas a normalizar.
        cols: colunas a normalizar.
        config: PipelineConfig (parâmetros).
        artifact_dir: diretório para armazenar/ler artefatos por coluna (default: ./operators).
        history_df: obrigatório quando isTrain=False e houver differencing (d>0);
                    deve conter, no mínimo, os últimos 'd' valores ORIGINAIS de cada coluna.
    Returns:
        (df_out, reports)
          - df_out: DataFrame original + colunas "<col>_norm"
          - reports: dict {col: NormalizationReport}
    """
    cfg = config or PipelineConfig()
    artifact_dir_path = Path(artifact_dir or Path("data_pipeline/operators"))
    _ensure_dir(artifact_dir_path)

    df_out = df.copy()
    reports: Dict[str, NormalizationReport] = {}

    for col in cols:
        errors: List[str] = []

        if col not in df_out.columns:
            warnings.warn(f"Coluna '{col}' não encontrada no DataFrame.")
            reports[col] = NormalizationReport(
                column=col,
                is_constant=False,
                adf_pvalue_initial=None,
                stationary_initial=None,
                differencing_applied=0,
                adf_pvalue_final=None,
                stationary_final=None,
                shapiro_pvalue=None,
                normality=None,
                transformer="missing",
                errors=("missing_column",),
            )
            continue

        series_orig = df_out[col].astype(float)

        # CONSTANTE?
        if _is_constant(series_orig):
            const_val = float(series_orig.iloc[0]) if len(series_orig) > 0 else 0.0
            # Ajuste/Transformação consistente
            pt_path, scaler_path, meta_path = _col_artifact_paths(artifact_dir_path, col)

            if isTrain:
                scaler = MinMaxScaler(feature_range=cfg.scaler_range)
                try:
                    scaler.fit(np.full((1, 1), const_val))
                    joblib.dump(scaler, scaler_path)
                    # "pt" não é necessário; salvar um "pt vazio" apenas por compatibilidade
                    pt = PowerTransformer(method="yeo-johnson")
                    joblib.dump(pt, pt_path)
                except Exception as e:
                    warnings.warn(f"Falha ao salvar scaler/pt para coluna constante '{col}': {e}")
                    errors.append(f"save_const_error:{e}")

                meta = {
                    "is_constant": True,
                    "constant_value": const_val,
                    "differencing_applied": 0,
                }
                try:
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    warnings.warn(f"Falha ao salvar meta da coluna constante '{col}': {e}")
                    errors.append(f"save_meta_const_error:{e}")

                # Saída normalizada para coluna constante: scaler.transform do mesmo valor
                try:
                    norm_vals = scaler.transform(np.full((len(series_orig), 1), const_val)).ravel()
                except Exception:
                    norm_vals = np.full(len(series_orig), 0.0)
                df_out[f"{col}_norm"] = norm_vals

                reports[col] = NormalizationReport(
                    column=col,
                    is_constant=True,
                    adf_pvalue_initial=None,
                    stationary_initial=None,
                    differencing_applied=0,
                    adf_pvalue_final=None,
                    stationary_final=None,
                    shapiro_pvalue=None,
                    normality=None,
                    transformer="constant",
                    errors=tuple(errors),
                )
                continue

            else:
                # Inferência: apenas carregar meta e aplicar scaler no valor constante
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    const_val_meta = float(meta.get("constant_value", series_orig.iloc[0]))
                except Exception as e:
                    warnings.warn(f"Falha ao carregar meta de coluna constante '{col}': {e}")
                    const_val_meta = float(series_orig.iloc[0])

                try:
                    scaler: MinMaxScaler = joblib.load(scaler_path)
                    norm_vals = scaler.transform(np.full((len(series_orig), 1), const_val_meta)).ravel()
                except Exception as e:
                    warnings.warn(f"Falha ao aplicar scaler em coluna constante '{col}': {e}")
                    norm_vals = np.full(len(series_orig), 0.0)

                df_out[f"{col}_norm"] = norm_vals

                reports[col] = NormalizationReport(
                    column=col,
                    is_constant=True,
                    adf_pvalue_initial=None,
                    stationary_initial=None,
                    differencing_applied=0,
                    adf_pvalue_final=None,
                    stationary_final=None,
                    shapiro_pvalue=None,
                    normality=None,
                    transformer="constant",
                    errors=tuple(errors),
                )
                continue

        # NÃO CONSTANTE
        if isTrain:
            # 1) ADF + differencing com fit no TREINO
            p_initial = _safe_adfuller(series_orig)
            stationary_initial = (p_initial is not None) and (p_initial <= cfg.adf_alpha)

            s_stat, d, p_final, stationary_final = _make_stationary_train(series_orig, cfg)

            # 2) Fit + Transform (salvando artefatos por coluna)
            transformed, transformer_name, p_shapiro, is_normal, err = _fit_and_transform_column_train(
                col, s_stat, cfg, artifact_dir_path
            )
            errors.extend(list(err))

            # 3) salvar meta com informações necessárias para inferência e inversão
            pt_path, scaler_path, meta_path = _col_artifact_paths(artifact_dir_path, col)
            meta = {
                "is_constant": False,
                "differencing_applied": d,
                "adf_pvalue_initial": None if p_initial is None else round(p_initial, 6),
                "stationary_initial": stationary_initial,
                "adf_pvalue_final": None if p_final is None else round(p_final, 6),
                "stationary_final": stationary_final,
                "transformer": transformer_name,
                "shapiro_pvalue": None if p_shapiro is None else round(p_shapiro, 6),
                "normality": is_normal,
                # Guardar últimos d valores originais para facilitar inversão posterior
                "last_train_values": (
                    series_orig.tail(d).tolist() if d > 0 else []
                ),
                "scaler_range": cfg.scaler_range,
            }
            try:
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
            except Exception as e:
                warnings.warn(f"Falha ao salvar meta de '{col}': {e}")
                errors.append(f"save_meta_error:{e}")

            # 4) Persistir a coluna normalizada no df_out
            df_out[f"{col}_norm"] = transformed

            reports[col] = NormalizationReport(
                column=col,
                is_constant=False,
                adf_pvalue_initial=meta["adf_pvalue_initial"],
                stationary_initial=stationary_initial,
                differencing_applied=d,
                adf_pvalue_final=meta["adf_pvalue_final"],
                stationary_final=stationary_final,
                shapiro_pvalue=meta["shapiro_pvalue"],
                normality=is_normal,
                transformer=transformer_name,
                errors=tuple(errors),
            )

        else:
            # INFERÊNCIA (teste/produção): carregar artefatos e transformar SEM ajuste
            pt_path, scaler_path, meta_path = _col_artifact_paths(artifact_dir_path, col)
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception as e:
                warnings.warn(f"Falha ao carregar meta de '{col}' no modo inferência: {e}")
                # fallback: copia a coluna e retorna
                df_out[f"{col}_norm"] = series_orig.values
                reports[col] = NormalizationReport(
                    column=col,
                    is_constant=False,
                    adf_pvalue_initial=None,
                    stationary_initial=None,
                    differencing_applied=0,
                    adf_pvalue_final=None,
                    stationary_final=None,
                    shapiro_pvalue=None,
                    normality=None,
                    transformer="failed",
                    errors=("load_meta_error",),
                )
                continue

            d = int(meta.get("differencing_applied", 0))
            # preparar "history" com últimos d valores originais do treino
            train_tail = None
            if d > 0:
                if history_df is not None and col in history_df.columns and len(history_df[col]) >= d:
                    train_tail = history_df[col].astype(float).tail(d)
                else:
                    # fallback: usar os valores salvos na meta (últimos do treino)
                    last_vals = meta.get("last_train_values", [])
                    if len(last_vals) >= d:
                        train_tail = pd.Series(last_vals[-d:], index=[-i for i in range(d, 0, -1)], dtype=float)
                    else:
                        train_tail = None  # sem contexto

            # 1) aplicar differencing com history
            s_stat = _difference_with_history(train_tail, series_orig, d, cfg)

            # 2) transformar usando artefatos existentes
            transformed, transformer_name, p_shapiro, is_normal, err = _transform_column_infer(
                col, s_stat, cfg, artifact_dir_path
            )
            errors.extend(list(err))

            df_out[f"{col}_norm"] = transformed

            reports[col] = NormalizationReport(
                column=col,
                is_constant=False,
                adf_pvalue_initial=meta.get("adf_pvalue_initial"),
                stationary_initial=meta.get("stationary_initial"),
                differencing_applied=d,
                adf_pvalue_final=meta.get("adf_pvalue_final"),
                stationary_final=meta.get("stationary_final"),
                shapiro_pvalue=p_shapiro,
                normality=is_normal,
                transformer=transformer_name,
                errors=tuple(errors),
            )

    return df_out, reports



# =========================
# Execução direta (exemplo)
# =========================

# def _usage():
#     # Exemplo de uso: carregue seu CSV e selecione colunas
#     caminho = Path(os.path.abspath(__file__))
#     root_dir = caminho.parent.parent
#     df_path =os.path.join(root_dir,'sensors/dados_modelo_lasso.csv')
#     #df_path =os.path.join(root_dir,'sensord/dados_modelo_rfe.csv')

#     df = pd.read_csv(df_path, index_col=0, parse_dates=True)
#     cols = df.columns.to_list()
#     df_norm, report = normalize_time_series(isTrain=True, df=df, cols=cols)
#     df_norm.to_csv(os.path.join(root_dir,'sensors/dados_modelo_lasso_norm.csv'))

#     print(df_norm.tail())
#     print({k: asdict(v) for k, v in report.items()})

# if __name__ == "__main__":
#     warnings.filterwarnings("ignore", category=FutureWarning)
#     _usage()