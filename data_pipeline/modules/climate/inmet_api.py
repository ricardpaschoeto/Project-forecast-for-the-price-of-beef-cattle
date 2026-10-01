import datetime as dt
from typing import Any

import pandas as pd
import requests


class InmetClient:
    """Cliente para consultar temperatura diária no Open-Meteo Archive."""

    _URL = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(self, http_client: Any = requests) -> None:
        self.http_client = http_client

    def get_daily(
        self,
        latitude: float,
        longitude: float,
        start: dt.date,
        end: dt.date,
    ) -> pd.DataFrame:
        """Retorna a temperatura média diária no ponto solicitado."""
        if start > end:
            raise ValueError("A data inicial deve ser anterior à data final.")

        response = self._request(latitude, longitude, start, end)
        data = response.json()
        if "daily" not in data:
            return pd.DataFrame()

        return self._to_dataframe(data["daily"])

    def _build_params(
        self,
        latitude: float,
        longitude: float,
        start: dt.date,
        end: dt.date,
    ) -> dict[str, Any]:
        return {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "daily": ["temperature_2m_mean"],
            "timezone": "America/Sao_Paulo",
        }

    def _request(
        self,
        latitude: float,
        longitude: float,
        start: dt.date,
        end: dt.date,
    ) -> Any:
        response = self.http_client.get(
            self._URL,
            params=self._build_params(latitude, longitude, start, end),
            timeout=60,
        )
        response.raise_for_status()
        return response

    def _to_dataframe(self, daily: dict[str, Any]) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(daily["time"]),
                "tmed": daily["temperature_2m_mean"],
            }
        )
        return frame.set_index("date")


def get_openmeteo_daily(
    latitude: float,
    longitude: float,
    start: dt.date,
    end: dt.date,
) -> pd.DataFrame:
    """Mantém a API funcional usada pelo pipeline de clima."""
    return InmetClient().get_daily(latitude, longitude, start, end)

def main():

    latitude = -16.6869
    longitude = -49.2648
    start = dt.date(2026, 9, 1)
    end = dt.date(2026, 9, 29)

    df = get_openmeteo_daily(latitude, longitude, start, end)
    print(df)

if __name__ == "__main__":
    main()

