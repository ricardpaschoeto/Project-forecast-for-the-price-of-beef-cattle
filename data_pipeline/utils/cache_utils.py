import os
import hashlib
import pandas as pd

CACHE_DIR = "sensors"

def cache_file(provider: str, key: str, start, end) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    h = hashlib.md5(f"{provider}{key}{start}{end}".encode("utf-8")).hexdigest()
    safe = "".join([c if c.isalnum() or c in ("_", "-", ".", "/") else "_" for c in key])
    safe = safe.replace("/", "__")
    return os.path.join(CACHE_DIR, f"{provider}_{safe}_{h}.parquet")

def read_cache(path: str):
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except:
            return None
    return None

def write_cache(df: pd.DataFrame, path: str):
    try:
        df.to_parquet(path)
    except:
        pass