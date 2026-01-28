import os
import pandas as pd
from utils.cache_utils import cache_file, read_cache, write_cache
from utils.http_utils import http_get_json

NASDAQ_API_KEY = os.getenv("NASDAQ_DATA_LINK_API_KEY", "").strip()
NASDAQ_BASE = "https://data.nasdaq.com/api/v3/datasets"

def fetch_nasdaq_timeseries(code, start, end, force_refresh=False):
    if not NASDAQ_API_KEY:
        return pd.DataFrame()

    cache_path = cache_file("nasdaq", code, start, end)
    if not force_refresh:
        cached = read_cache(cache_path)
        if cached is not None and not cached.empty:
            return cached

    url = f"{NASDAQ_BASE}/{code}.json"
    params = {
        "api_key": NASDAQ_API_KEY,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "order": "asc",
    }

    js = http_get_json(url, params)
    ds = js.get("dataset", {})
    cols = ds.get("column_names", [])
    data = ds.get("data", [])

    if not cols or not data:
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=cols)
    df[cols[0]] = pd.to_datetime(df[cols[0]])
    df = df.set_index(cols[0]).sort_index()
    df = df.rename(columns={cols[1]: "value"})[["value"]]

    write_cache(df, cache_path)
    return df


def fetch_nasdaq_first_available(candidates, start, end, force_refresh=False):
    for code in candidates:
        try:
            df = fetch_nasdaq_timeseries(code, start, end, force_refresh)
            if df is not None and not df.empty:
                return df
        except:
            continue
    return pd.DataFrame()