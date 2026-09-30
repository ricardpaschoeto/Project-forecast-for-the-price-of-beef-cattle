import datetime as dt
import pandas as pd
import requests


def get_openmeteo_daily(
    latitude: float,
    longitude: float,
    start: dt.date,
    end: dt.date,
) -> pd.DataFrame:
    """
    Retorna dados diários do Open-Meteo:
    precip, tmin, tmax, tmed.
    """

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "daily": [
           "temperature_2m_mean",
        ],
        "timezone": "America/Sao_Paulo",
    }

    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()

    data = resp.json()

    if "daily" not in data:
        return pd.DataFrame()

    daily = data["daily"]

    df = pd.DataFrame({
        "date": pd.to_datetime(daily["time"]),
        "tmed": daily["temperature_2m_mean"],
    })

    df.set_index("date", inplace=True)

    return df

def main():

    latitude = -16.6869
    longitude = -49.2648
    start = dt.date(2026, 9, 1)
    end = dt.date(2026, 9, 29)

    df = get_openmeteo_daily(latitude, longitude, start, end)
    print(df)

if __name__ == "__main__":
    main()

