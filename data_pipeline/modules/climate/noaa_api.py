import io
import datetime as dt
import pandas as pd
import requests


def get_noaa_oni(start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    Retorna o índice ONI (El Niño) diário (ffill).
    """

    url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    raw = resp.text
    df = pd.read_fwf(io.StringIO(raw), header=None)
    df.columns = ["year", "month", "oni"]

    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=15))
    df = df.set_index("date").sort_index()
    df = df.loc[start:end]

    idx = pd.date_range(start=start, end=end, freq="D")
    df_daily = df.resample("D").ffill().reindex(idx)
    df_daily.index.name = "date"

    df_daily.rename(columns={"oni": "el nino"}, inplace=True)

    return df_daily[["el nino"]]

def main():
    start = dt.date(2026, 9, 1)
    end = dt.date(2026, 9, 30)
    df = get_noaa_oni(start, end)
    print(df)

if __name__ == "__main__":
    main()