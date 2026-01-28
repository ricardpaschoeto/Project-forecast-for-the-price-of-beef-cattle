# data-pipeline/modules/master/master_builder.py

from __future__ import annotations

import datetime as dt
import numpy as np
import pandas as pd

from utils.date_utils import parse_date_br, ensure_date_range_df
from modules.market import get_market_data, fill_beef_trade_and_tariffs
from modules.macro import get_macro_data
from modules.climate import get_climate_data
from modules.seasonality import get_calendar_features


# ---- Lista completa de colunas esperadas no dataframe final ----
_EXPECTED_COLUMNS = [
    # índice / data
    "data",

    # tarifas & exportações
    "Taxa_EUA",
    "Volume_exportação_EUA",
    "Taxa_UE",
    "Volume_exportação_EUROPA",
    "Taxa_china",
    "Volume_exportação_ASIA",

    # mercado (USD/BRL, proxies commodities)
    "boi_real", "boi_dolar",
    "soja_real", "soja_dolar",
    "milho_real", "milho_dolar",
    "dolar",
    "cme",

    # calendário / sazonalidade
    "CovidPeriodFlag", "ciclo", "aspecto",
    "ano_novo_flag",
    "carnaval_core_flag", "carnaval_window_flag",
    "pascoa_flag", "pascoa_window_flag",
    "dias_maes_flag", "dias_maes_weekend_flag",
    "festas_juninas_flag",
    "santo_antonio_flag", "sao_joao_flag", "sao_pedro_flag",
    "ferias_midyear_flag", "ferias_verao_flag",
    "dias_pais_flag", "dias_pais_weekend_flag",
    "independencia_flag", "independencia_window_flag",
    "dia_criancas_flag",
    "finados_flag", "finados_weekend_flag",
    "natal_flag", "natal_window_flag",
    "fim_ano_window_flag",
    "sazonal_carne_weight",

    # clima
    "scpdsi", "el nino", "precip_total_mm", "TEMPERATURA",

    # macro
    "selic", "ipca", "pib_brasil", "pib_agro",
]


def _ensure_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Garante a existência de todas as colunas, preenchendo com NaN as ausentes."""
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols]


def _build_climate_block(
    start: dt.date,
    end: dt.date,
    config: dict | None,
) -> pd.DataFrame:
    """
    Constrói o bloco de clima. Se faltarem parâmetros no config, devolve
    DataFrame com as colunas climáticas em NaN para não quebrar o pipeline.
    """
    climate_cols = ["precip_total_mm", "TEMPERATURA", "scpdsi", "el nino"]

    if not config:
        return pd.DataFrame(index=pd.date_range(start, end, freq="D"), columns=climate_cols)

    estacoes = (config.get("climate", {}) or {}).get("estacoes_inmet")
    bbox = (config.get("climate", {}) or {}).get("terraclimate_bbox")
    nc_path = (config.get("climate", {}) or {}).get("terraclimate_scpdsi_nc")

    # Se algum dos parâmetros for None, retorna blocos vazios (NaN)
    if not estacoes or not bbox or not nc_path:
        return pd.DataFrame(index=pd.date_range(start, end, freq="D"), columns=climate_cols)

    return get_climate_data(
        start=start,
        end=end,
        estacoes_inmet=estacoes,
        terraclimate_bbox=bbox,
        terraclimate_scpdsi_nc=nc_path,
    )


def build_master_dataframe(
    data_inicial: str,
    data_final: str,
    *,
    config: dict | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Constrói o dataframe diário consolidado com todas as fontes:

    Parâmetros
    ----------
    data_inicial : str
        Data inicial no formato 'dd-mm-aaaa'.
    data_final : str
        Data final no formato 'dd-mm-aaaa'.
    config : dict | None
        Parâmetros opcionais, por exemplo:
        {
          "climate": {
              "estacoes_inmet": ["A123", "A456"],
              "terraclimate_bbox": {"min_lon": -60, "max_lon": -40, "min_lat": -25, "max_lat": -5},
              "terraclimate_scpdsi_nc": "/path/TerraClimate_scpdsi_2024.nc"
          }
        }
    force_refresh : bool
        Se True, força re-download nas fontes de mercado (quando suportado).

    Retorno
    -------
    pd.DataFrame
        Dataframe diário com todas as colunas esperadas (_EXPECTED_COLUMNS).
    """
    start = parse_date_br(data_inicial)
    end = parse_date_br(data_final)

    # -------- Bloco de Mercado --------
    df_market = get_market_data(start, end, force_refresh=force_refresh)
    # (opcional) reforçar as tarifas/volumes via função dedicada
    df_market = fill_beef_trade_and_tariffs(df_market, start, end)
    df_market = ensure_date_range_df(df_market, start, end)

    # -------- Bloco Macro --------
    df_macro = get_macro_data(start, end)
    df_macro = ensure_date_range_df(df_macro, start, end)

    # -------- Bloco Clima --------
    df_clima = _build_climate_block(start, end, config)
    df_clima = ensure_date_range_df(df_clima, start, end)

    # -------- Bloco Sazonalidade --------
    df_cal = get_calendar_features(start, end)
    df_cal = ensure_date_range_df(df_cal, start, end)

    # -------- Join geral --------
    df = df_market.join(df_macro, how="outer")
    df = df.join(df_clima, how="outer")
    df = df.join(df_cal, how="outer")

    # Range e ordenação final
    df = ensure_date_range_df(df, start, end)

    # Coluna 'data' (datetime): mantém compatibilidade com seu script original
    df["data"] = df.index  # se quiser string: df["data"] = df.index.strftime("%d-%m-%Y")

    # Garante todas as colunas esperadas
    df = _ensure_columns(df, _EXPECTED_COLUMNS)

    return df