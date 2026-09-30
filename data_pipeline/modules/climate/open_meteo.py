import pandas as pd
import requests


def get_precipitation(lat, lon, start_date, end_date):

    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}"
        f"&longitude={lon}"
        f"&start_date={start_date}"
        f"&end_date={end_date}"
        "&daily=precipitation_sum"
        "&timezone=UTC"
    )

    r = requests.get(url, timeout=60)
    r.raise_for_status()

    data = r.json()

    df = pd.DataFrame({
        "date": pd.to_datetime(data["daily"]["time"]),
        "precipitation_mm": data["daily"]["precipitation_sum"]
    })

    return df.set_index("date")