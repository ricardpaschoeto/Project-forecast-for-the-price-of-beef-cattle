import datetime as dt
import pandas as pd
import requests


def get_bcb_series_raw(series_id: int, start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    Consulta diretamente uma série do SGS (Banco Central do Brasil).
    Retorna DataFrame indexado por date, com a coluna 'value'.
    """

    url = (
        f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_id}/dados"
        f"?formato=json&dataInicial={start.strftime('%d/%m/%Y')}"
        f"&dataFinal={end.strftime('%d/%m/%Y')}"
    )

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if not data:
        return pd.DataFrame(columns=["value"])

    df = pd.DataFrame(data)
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["valor"] = pd.to_numeric(df["valor"].str.replace(",", "."), errors="coerce")

    df = df.set_index("data").rename(columns={"valor": "value"})
    df.index.name = "date"
    return df