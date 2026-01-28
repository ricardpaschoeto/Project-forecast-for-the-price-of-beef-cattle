import datetime as dt
import pandas as pd
import numpy as np
import requests


def get_inmet_daily_station(cod_estacao: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    Retorna dados diários da estação INMET:
    precip, tmin, tmax, tmed.
    """

    url = (
        f"https://https://tempo.inmet.gov.br/estacao/diaria/"
        f"{start.strftime('%Y-%m-%d')}/"
        f"{end.strftime('%Y-%m-%d')}/"
        f"{cod_estacao}"
    )

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    df["date"] = pd.to_datetime(df["DT_MEDICAO"]).dt.date
    df = df.set_index("date")

    precip_col = "CHUVA" if "CHUVA" in df.columns else ("PREC" if "PREC" in df.columns else None)
    if precip_col is None:
        raise ValueError("Coluna de precipitação não encontrada no retorno INMET.")

    df["precip"] = pd.to_numeric(df[precip_col], errors="coerce")
    df["tmin"] = pd.to_numeric(df.get("TMIN", np.nan), errors="coerce")
    df["tmax"] = pd.to_numeric(df.get("TMAX", np.nan), errors="coerce")

    if "T_MED" in df.columns:
        df["tmed"] = pd.to_numeric(df["T_MED"], errors="coerce")
    else:
        df["tmed"] = (df["tmin"] + df["tmax"]) / 2.0

    return df[["precip", "tmin", "tmax", "tmed"]]


def get_inmet_precip_temp(start: dt.date, end: dt.date, estacoes: list[str]) -> pd.DataFrame:
    """
    Agrega várias estações INMET, retornando:
    precip_total_mm, TEMPERATURA (média tmed).
    """
    frames = []

    for cod in estacoes:
        df_est = get_inmet_daily_station(cod, start, end)
        if df_est.empty:
            continue

        idx = pd.date_range(start=start, end=end, freq="D")
        df_est = df_est.reindex(idx)
        df_est.index.name = "date"
        frames.append(df_est)

    idx = pd.date_range(start=start, end=end, freq="D")

    if not frames:
        df = pd.DataFrame(index=idx, columns=["precip_total_mm", "TEMPERATURA"])
        return df

    panel = pd.concat(frames, axis=0, keys=estacoes, names=["estacao", "date"])
    daily = panel.groupby("date").mean()

    daily.rename(
        columns={
            "precip": "precip_total_mm",
            "tmed": "TEMPERATURA",
        },
        inplace=True,
    )

    daily = daily.reindex(idx)
    daily.index.name = "date"

    return daily[["precip_total_mm", "TEMPERATURA"]]

