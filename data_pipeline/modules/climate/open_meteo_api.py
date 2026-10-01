import datetime as dt
from typing import Any

import pandas as pd
import requests


class OpenMeteoClient:
    """Cliente para consultar precipitação diária no Open-Meteo."""

    _URL = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(self, http_client: Any = requests) -> None:
        self.http_client = http_client

    def get_precipitation(
        self,
        lat: float,
        lon: float,
        start_date: Any,
        end_date: Any,
    ) -> pd.DataFrame:
        """Retorna a precipitação diária em milímetros."""
        response = self._request(lat, lon, start_date, end_date)
        return self._to_dataframe(response.json())

    def _build_params(
        self,
        lat: float,
        lon: float,
        start_date: Any,
        end_date: Any,
    ) -> dict[str, Any]:
        return {
            "latitude": lat,
            "longitude": lon,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "daily": "precipitation_sum",
            "timezone": "UTC",
        }

    def _request(
        self,
        lat: float,
        lon: float,
        start_date: Any,
        end_date: Any,
    ) -> Any:
        response = self.http_client.get(
            self._URL,
            params=self._build_params(lat, lon, start_date, end_date),
            timeout=60,
        )
        response.raise_for_status()
        return response

    def _to_dataframe(self, data: dict) -> pd.DataFrame:
        daily = data["daily"]
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(daily["time"]),
                "precipitation_mm": daily["precipitation_sum"],
            }
        )
        return frame.set_index("date")


def get_precipitation(lat, lon, start_date, end_date):
    """Mantém a API funcional usada pelo pipeline de clima."""
    return OpenMeteoClient().get_precipitation(lat, lon, start_date, end_date)

def main():
    lat = -15.7801  # Latitude for Brasília, Brazil
    lon = -47.9292  # Longitude for Brasília, Brazil
    start_date = dt.date(2020, 1, 1)
    end_date = dt.date(2026, 10, 1)

    df_precipitation = get_precipitation(lat, lon, start_date, end_date)
    print(df_precipitation)

if __name__ == "__main__":
    main()