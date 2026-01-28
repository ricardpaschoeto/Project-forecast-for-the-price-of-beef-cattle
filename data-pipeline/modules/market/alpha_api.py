import os
import pandas as pd
from utils.cache_utils import cache_file, read_cache, write_cache
from utils.http_utils import http_get_json

ALPHA_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
ALPHA_BASE = "https://www.alphavantage.co/query"

def fetch_alpha_fx_daily(from_ccy, to_ccy, start, end, force_refresh=False):
    if not ALPHA_API_KEY:
        return pd.DataFrame()

    key = f"FX_{from_ccy}_{to_ccy}"
    cache_path = cache_file("alphav", key, start, end)

    if not force_refresh:
        cached = read_cache(cache_path)
        if cached is not None and not cached.empty:
            return cached

    params = {
        "function": "FX_DAILY",
        "from_symbol": from_ccy,
        "to_symbol": to_ccy,
        "outputsize": "full",
        "apikey": ALPHA_API_KEY,
    }

    js = http_get_json(ALPHA_BASE, params)
    ts_key = next((k for k in js.keys() if "Time Series FX" in k), None)
    if not ts_key:
        return pd.DataFrame()

    ts = js[ts_key]
    df = pd.DataFrame(ts).T
    df.index = pd.to_datetime(df.index)
    close = next(c for c in df.columns if "close" in c.lower())
    df = df.rename(columns={close: "value"})[["value"]]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.loc[(df.index.date >= start) & (df.index.date <= end)]
    write_cache(df, cache_path)
    return df