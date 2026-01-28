import datetime as dt
import numpy as np
import pandas as pd
import requests


WITS_COUNTRIES_URL = "https://wits.worldbank.org/API/V1/wits/datasource/trn/country/ALL"


def wits_country_map() -> pd.DataFrame:
    """Retorna tabela com countrycode e iso3Code do WITS/TRN."""
    r = requests.get(WITS_COUNTRIES_URL, timeout=60)
    r.raise_for_status()
    return pd.DataFrame(r.json())


def wits_code_by_iso3(df_map: pd.DataFrame, iso3: str) -> int:
    row = df_map.loc[df_map["iso3Code"].astype(str).str.upper() == iso3.upper()]
    if row.empty:
        raise ValueError(f"ISO3 não encontrado no WITS: {iso3}")
    return int(row.iloc[0]["countrycode"])


def wits_sdmx_to_rows(js: dict) -> pd.DataFrame:
    """
    Parser genérico SDMX-JSON.
    Retorna DF com colunas das dimensões e OBS_VALUE.
    """
    dims = ((js.get("structure") or {}).get("dimensions") or {}).get("series") or []
    dataset = js.get("dataSets", [{}])[0]
    series = dataset.get("series", {})
    rows = []

    dim_ids = [d.get("id") for d in dims]
    dim_vals_lists = [[v.get("id") for v in (d.get("values") or [])] for d in dims]

    for key, payload in series.items():
        idxs = [int(x) for x in key.split(":")]
        dim_kv = {}

        for dim_id, vals, i in zip(dim_ids, dim_vals_lists, idxs):
            if dim_id is not None and 0 <= i < len(vals):
                dim_kv[dim_id] = vals[i]

        obs = payload.get("observations") or {}
        if not obs:
            continue

        first_key = sorted(obs.keys(), key=lambda x: int(x))[0]
        value_raw = obs[first_key]
        value = value_raw[0] if isinstance(value_raw, list) else value_raw
        dim_kv["OBS_VALUE"] = value

        rows.append(dim_kv)

    return pd.DataFrame(rows)


def wits_fetch_tariffs_trn(reporter, partner, hs6_list, year) -> pd.DataFrame:
    hs6 = ";".join(sorted(set(hs6_list)))
    url = (
        "https://wits.worldbank.org/API/V1/SDMX/V21/datasource/TRN/"
        f"reporter/{reporter}/partner/{partner}/product/{hs6}/"
        f"year/{year}/datatype/reported?format=JSON"
    )

    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return wits_sdmx_to_rows(r.json())


def tariff_effective_min_mfn_pref(df_tar: pd.DataFrame) -> pd.Series:
    if df_tar.empty:
        return pd.Series(dtype=float)

    col_prod = next((c for c in df_tar.columns if c.upper() == "PRODUCT"), None)
    col_type = next((c for c in df_tar.columns if c.upper() == "TARIFFTYPE"), None)
    col_val = next((c for c in df_tar.columns if c.upper().startswith("OBS")), None)

    if not (col_prod and col_type and col_val):
        return pd.Series(dtype=float)

    tmp = df_tar[[col_prod, col_type, col_val]].copy()
    tmp[col_val] = pd.to_numeric(tmp[col_val], errors="coerce")
    tmp[col_prod] = tmp[col_prod].astype(str)
    tmp[col_type] = tmp[col_type].astype(str).str.upper()

    piv = tmp.pivot_table(
        index=col_prod, columns=col_type, values=col_val, aggfunc="mean"
    )

    mfn = piv.get("MFN")
    pref = piv.get("PREF")

    if mfn is None and pref is None:
        return pd.Series(dtype=float)
    if mfn is None:
        return pref
    if pref is None:
        return mfn

    eff = pd.concat([mfn, pref], axis=1).min(axis=1)
    eff.name = "tariff"
    return eff
