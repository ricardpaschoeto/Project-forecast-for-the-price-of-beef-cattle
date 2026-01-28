import datetime as dt
import pandas as pd

def parse_date_br(date_str: str) -> dt.date:
    return dt.datetime.strptime(date_str, "%d-%m-%Y").date()

def ensure_date_range_df(df: pd.DataFrame, start: dt.date, end: dt.date) -> pd.DataFrame:
    idx = pd.date_range(start=start, end=end, freq="D")
    df = df.copy()
    df = df.reindex(idx)
    df.index.name = "date"
    return df.sort_index()