import io
import datetime as dt
import pandas as pd
import requests


def get_noaa_oni(start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    Retorna o índice ONI (El Niño) diário utilizando
    forward-fill do último valor disponível.
    """

    url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    df = pd.read_fwf(io.StringIO(resp.text), header=None)

    df.columns = ["season", "year", "total", "oni"]

    # Remove linha de cabeçalho presente nos dados
    df = df[df["year"] != "YR"].copy()

    df["season"] = df["season"].astype(str).str.strip()

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["oni"] = pd.to_numeric(df["oni"], errors="coerce")

    season_map = {
        "DJF": 1,
        "JFM": 2,
        "FMA": 3,
        "MAM": 4,
        "AMJ": 5,
        "MJJ": 6,
        "JJA": 7,
        "JAS": 8,
        "ASO": 9,
        "SON": 10,
        "OND": 11,
        "NDJ": 12,
    }

    df["month"] = df["season"].map(season_map)

    df = df.dropna(subset=["year", "month", "oni"])

    df["date"] = pd.to_datetime(
        {
            "year": df["year"].astype(int),
            "month": df["month"].astype(int),
            "day": 15,
        }
    )

    df = (
        df[["date", "oni"]]
        .set_index("date")
        .sort_index()
    )

    last_available = df.index.max()

    print(f"Última data disponível NOAA: {last_available:%Y-%m-%d}")

    # Gera série diária completa
    daily_index = pd.date_range(
        start=df.index.min(),
        end=max(pd.Timestamp(end), last_available),
        freq="D",
    )

    df_daily = (
        df.resample("D")
          .ffill()
          .reindex(daily_index)
          .ffill()
    )

    # Seleciona apenas o período solicitado
    df_daily = df_daily.loc[pd.Timestamp(start):pd.Timestamp(end)]

    df_daily.index.name = "date"

    df_daily.rename(
        columns={"oni": "el_nino"},
        inplace=True
    )

    return df_daily[["el_nino"]]


def main():
    start = dt.date(2026, 9, 1)
    end = dt.date(2026, 9, 30)

    df = get_noaa_oni(start, end)

    print(df)

if __name__ == "__main__":
    main()