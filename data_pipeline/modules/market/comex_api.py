import datetime as dt
import numpy as np
import pandas as pd
import requests

COMEX_GENERAL_URL = "https://api-comexstat.mdic.gov.br/general?language=pt"
COMEX_ECOBLOCKS_URL = "https://api-comexstat.mdic.gov.br/tables/economic-blocks"


def comex_get_block_country_codes(search: str) -> list[int]:
    """
    Retorna a lista de países (coCountry) pertencentes a um bloco econômico.
    Ex: "União Europeia", "Ásia", "Estados Unidos".
    """
    params = {"language": "pt", "add": "country", "search": search}
    r = requests.get(COMEX_ECOBLOCKS_URL, params=params, timeout=60)
    r.raise_for_status()

    js = r.json()
    rows = (js.get("data") or {}).get("list") or []
    codes = []

    for row in rows:
        cc = row.get("coCountry")
        if cc is not None:
            try:
                codes.append(int(cc))
            except:
                pass

    return sorted(set(codes))


def comex_query_exports_monthly(
    start: dt.date,
    end: dt.date,
    sh4_list: list[str],
    country_codes: list[int],
    include_ncm_detail: bool = True,
) -> pd.DataFrame:
    """
    Consulta exportações do ComexStat (flow=export), filtrando SH4 e países.
    Retorna DataFrame mensal com colunas:
    - date (primeiro dia do mês)
    - kg
    - country_code
    - ncm (opcional)
    """
    details = ["country", "month"]
    if include_ncm_detail:
        details.append("ncm")

    body = {
        "flow": "export",
        "monthDetail": True,
        "period": {"from": start.strftime("%Y-%m"), "to": end.strftime("%Y-%m")},
        "filters": [
            {"filter": "sh4", "values": sh4_list},
            {"filter": "country", "values": country_codes},
        ],
        "details": details,
        "metrics": ["metricKG"],
    }

    r = requests.post(COMEX_GENERAL_URL, json=body, timeout=120)
    r.raise_for_status()

    js = r.json()
    rows = (js.get("data") or {}).get("list") or []
    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Mapeamento robusto (nomes variam por versão da API)
    col_ano = next((c for c in df.columns if c.lower() in ("coano", "ano")), None)
    col_mes = next((c for c in df.columns if c.lower() in ("comes", "mes")), None)
    col_kg = next((c for c in df.columns if c.lower() in ("metrickg", "peso_liquido_kg", "kg")), None)
    col_pais = next((c for c in df.columns if c.lower() in ("copais", "cocountry", "country")), None)
    col_ncm = next((c for c in df.columns if c.lower() in ("concm", "ncm")), None)

    if not (col_ano and col_mes and col_kg and col_pais):
        return pd.DataFrame()

    df[col_ano] = pd.to_numeric(df[col_ano], errors="coerce").astype("Int64")
    df[col_mes] = pd.to_numeric(df[col_mes], errors="coerce").astype("Int64")
    df[col_kg] = pd.to_numeric(df[col_kg], errors="coerce")

    df["date"] = pd.to_datetime(dict(year=df[col_ano], month=df[col_mes], day=1))
    df = df.dropna(subset=["date"])

    df = df.rename(columns={col_pais: "country_code", col_kg: "kg"})

    if include_ncm and col_ncm:
        df = df.rename(columns={col_ncm: "ncm"})
    else:
        df["ncm"] = np.nan

    df = df[["date", "kg", "country_code", "ncm"]].sort_values("date")
    return df
