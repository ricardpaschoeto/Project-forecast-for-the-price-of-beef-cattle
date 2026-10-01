import io
import datetime as dt
from typing import Any

import pandas as pd
import requests


class NoaaClient:
    """Cliente para consultar e preparar o índice ONI da NOAA."""

    _URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
    _SEASON_MONTHS = {
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

    def __init__(self, http_client: Any = requests) -> None:
        self.http_client = http_client

    def get_oni(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        """Retorna o índice ONI diário para o período solicitado."""
        if start > end:
            raise ValueError("A data inicial deve ser anterior à data final.")

        data = self._fetch_data()
        return self._to_daily_dataframe(data, start, end)

    def _fetch_data(self) -> str:
        response = self.http_client.get(self._URL, timeout=30)
        response.raise_for_status()
        return response.text

    def _parse_data(self, content: str) -> pd.DataFrame:
        data = pd.read_csv(
            io.StringIO(content),
            sep=r"\s+",
            header=None,
            names=["season", "year", "total", "oni"],
        )
        data = data[data["year"] != "YR"].copy()
        data["season"] = data["season"].astype(str).str.strip()
        data["year"] = pd.to_numeric(data["year"], errors="coerce")
        data["oni"] = pd.to_numeric(data["oni"], errors="coerce")
        data["month"] = data["season"].map(self._SEASON_MONTHS)
        data = data.dropna(subset=["year", "month", "oni"])
        data["date"] = pd.to_datetime(
            {
                "year": data["year"].astype(int),
                "month": data["month"].astype(int),
                "day": 15,
            }
        )
        return data[["date", "oni"]].set_index("date").sort_index()

    def _to_daily_dataframe(
        self,
        content: str,
        start: dt.date,
        end: dt.date,
    ) -> pd.DataFrame:
        data = self._parse_data(content)
        last_available = data.index.max()
        print(f"Última data disponível NOAA: {last_available:%Y-%m-%d}")

        daily_index = pd.date_range(
            start=data.index.min(),
            end=max(pd.Timestamp(end), last_available),
            freq="D",
        )
        daily = data.resample("D").ffill().reindex(daily_index).ffill()
        result = daily.loc[pd.Timestamp(start):pd.Timestamp(end)].copy()
        result.index.name = "date"
        return result.rename(columns={"oni": "el_nino"})[["el_nino"]]


def get_noaa_oni(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Mantém a API funcional usada pelo pipeline de clima."""
    return NoaaClient().get_oni(start, end)


def main():
    start = dt.date(2020, 1, 1)
    end = dt.date(2026, 1, 10)

    df = get_noaa_oni(start, end)

    print(df)

if __name__ == "__main__":
    main()