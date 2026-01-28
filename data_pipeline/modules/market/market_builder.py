import datetime as dt
import pandas as pd
import numpy as np

from utils.date_utils import ensure_date_range_df
from modules.market.nasdaq_api import fetch_nasdaq_timeseries, fetch_nasdaq_first_available
from modules.market.alpha_api import fetch_alpha_fx_daily, fetch_alpha_commodity
from modules.market.comex_api import comex_get_block_country_codes, comex_query_exports_monthly
from modules.market.wits_api import (
    wits_country_map,
    wits_code_by_iso3,
    wits_fetch_tariffs_trn,
    tariff_effective_min_mfn_pref,
)

def to_daily_ffill(df, start, end):
    idx = pd.date_range(start=start, end=end, freq="D")
    if df is None or df.empty:
        return pd.DataFrame(index=idx)
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return df.reindex(idx).ffill()


def get_market_data(start: dt.date, end: dt.date, force_refresh=False) -> pd.DataFrame:
    idx = pd.date_range(start=start, end=end, freq="D")
    market = pd.DataFrame(index=idx)

    # 1) Dólar — Alpha Vantage
    fx = fetch_alpha_fx_daily("USD", "BRL", start, end, force_refresh)
    market["dolar"] = to_daily_ffill(fx, start, end)["value"]

    # 2) Milho USD
    milho = fetch_alpha_commodity("CORN", "daily", start, end, force_refresh)
    if milho.empty:
        milho = fetch_alpha_commodity("CORN", "monthly", start, end, force_refresh)
    market["milho_dolar"] = to_daily_ffill(milho, start, end)["value"]

    # 3) Soja USD — Nasdaq Data Link
    soja_candidates = ["ODA/PSOYB_USD", "FRED/PSOYBUSDQ"]
    soja = fetch_nasdaq_first_available(soja_candidates, start, end, force_refresh)
    market["soja_dolar"] = to_daily_ffill(soja, start, end).get("value", np.nan)

    # 4) Boi USD — Nasdaq Data Link
    boi_candidates = ["ODA/PBEEF_USD", "FRED/PBEEFUSDQ"]
    boi = fetch_nasdaq_first_available(boi_candidates, start, end, force_refresh)
    market["boi_dolar"] = to_daily_ffill(boi, start, end).get("value", np.nan)

    # Converte para BRL:
    market["soja_real"] = market["soja_dolar"] * market["dolar"]
    market["milho_real"] = market["milho_dolar"] * market["dolar"]
    market["boi_real"] = market["boi_dolar"] * market["dolar"]

    # Placeholder para cme / futuros
    for col in ["cme", "boi_futuro", "soja_futuro", "milho_futuro"]:
        market[col] = np.nan

    # Exportações + Tarifas
    market = fill_beef_trade_and_tariffs(market, start, end)
    market = ensure_date_range_df(market, start, end)
    return market


def fill_beef_trade_and_tariffs(market: pd.DataFrame, start: dt.date, end: dt.date) -> pd.DataFrame:
    sh4_beef = ["0201", "0202", "0206", "1602"]

    # Regiões via blocos Comex
    ue = comex_get_block_country_codes("União Europeia")
    usa = comex_get_block_country_codes("Estados Unidos")
    china = comex_get_block_country_codes("China")
    hk = comex_get_block_country_codes("Hong Kong")
    asia = comex_get_block_country_codes("Ásia")
    oriente_medio = comex_get_block_country_codes("Oriente Médio")

    asia = sorted(set(asia) - set(oriente_medio))
    asia = sorted(set(asia) | set(china) | set(hk))

    region_map = {
        "EUA": usa,
        "EUROPA": ue,
        "ASIA": asia,
        "CHINA": china,
    }

    def region_monthly_volume(countries):
        if not countries:
            return pd.Series(dtype=float)
        df = comex_query_exports_monthly(start, end, sh4_beef, countries)
        if df.empty:
            return pd.Series(dtype=float)
        return df.groupby("date")["kg"].sum()

    vol_usa = region_monthly_volume(region_map["EUA"])
    vol_eu = region_monthly_volume(region_map["EUROPA"])
    vol_asia = region_monthly_volume(region_map["ASIA"])

    market["Volume_exportação_EUA"] = to_daily_ffill(vol_usa, start, end)
    market["Volume_exportação_EUROPA"] = to_daily_ffill(vol_eu, start, end)
    market["Volume_exportação_ASIA"] = to_daily_ffill(vol_asia, start, end)

    # HS6 list derivada do NCM
    df_all = comex_query_exports_monthly(
        start,
        end,
        sh4_beef,
        sorted(set(usa + ue + asia)),
        include_ncm_detail=True,
    )

    hs6_list = []
    if not df_all.empty and df_all["ncm"].notna().any():
        ncm = df_all["ncm"].astype(str).str.replace(r"\D", "", regex=True)
        hs6_list = sorted(set(ncm.str[:6].dropna().unique()))
        hs6_list = [x for x in hs6_list if len(x) == 6]

    if not hs6_list:
        # Se nada encontrado, não quebra pipeline.
        market["Taxa_EUA"] = np.nan
        market["Taxa_UE"] = np.nan
        market["Taxa_china"] = np.nan
        return market

    # Mapa WITS
    cmap = wits_country_map()
    partner_bra = wits_code_by_iso3(cmap, "BRA")
    reporter_usa = wits_code_by_iso3(cmap, "USA")
    reporter_chn = wits_code_by_iso3(cmap, "CHN")

    try:
        reporter_eu = wits_code_by_iso3(cmap, "EUU")
    except:
        reporter_eu = wits_code_by_iso3(cmap, "DEU")

    years = list(range(start.year, end.year + 1))

    def tariff_series(reporter):
        out = {}
        for y in years:
            df_tar = wits_fetch_tariffs_trn(reporter, partner_bra, hs6_list, y)
            eff = tariff_effective_min_mfn_pref(df_tar)
            out[pd.Timestamp(year=y, month=1, day=1)] = float(eff.mean()) if len(eff) else np.nan
        s = pd.Series(out).sort_index()
        return s

    taxa_eua_ann = tariff_series(reporter_usa)
    taxa_ue_ann = tariff_series(reporter_eu)
    taxa_china_ann = tariff_series(reporter_chn)

    daily_eua = to_daily_ffill(taxa_eua_ann, start, end)
    daily_ue = to_daily_ffill(taxa_ue_ann, start, end)
    daily_china = to_daily_ffill(taxa_china_ann, start, end)

    # Regras EUA (10% → 50% → isenções a partir nov/2025)
    APR_START = pd.Timestamp("2025-04-01")
    AUG_START = pd.Timestamp("2025-08-01")
    NOV_EXEMPT = pd.Timestamp("2025-11-13")

    base = daily_eua.copy()
    base.loc[base.index >= APR_START] = 10.0
    base.loc[(base.index >= AUG_START) & (base.index < NOV_EXEMPT)] = 50.0

    # Lista simplificada de HS6 isentos (exemplo)
    US_EXEMPT = {
        "020110", "020120", "020130", "020210", "020220",
        "020230", "020610", "020621", "020622", "020629", "160250"
    }

    df_usa = comex_query_exports_monthly(start, end, sh4_beef, usa, include_ncm_detail=True)

    if not df_usa.empty:
        ncm = df_usa["ncm"].astype(str).str.replace(r"\D", "", regex=True)
        df_usa["hs6"] = ncm.str[:6]

        w = (
            df_usa.groupby(["date", "hs6"])["kg"]
            .sum()
            .reset_index()
        )
        tot = w.groupby("date")["kg"].sum().rename("kg_total").reset_index()
        w = w.merge(tot, on="date")
        w["share"] = w["kg"] / w["kg_total"]

        share_isento_m = (
            w[w["hs6"].isin(US_EXEMPT)]
            .groupby("date")["share"]
            .sum()
            .sort_index()
        )
        share_isento_d = to_daily_ffill(share_isento_m, start, end).fillna(0.0)

        mask = base.index >= NOV_EXEMPT
        base.loc[mask] = (
            share_isento_d.loc[mask] * 10.0
            + (1 - share_isento_d.loc[mask]) * 50.0
        )

    market["Taxa_EUA"] = base
    market["Taxa_UE"] = daily_ue
    market["Taxa_china"] = daily_china

    return market