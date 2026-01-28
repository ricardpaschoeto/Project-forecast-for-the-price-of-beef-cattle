
import sys, os
import time
import hashlib
import datetime as dt
import numpy as np
import pandas as pd
from pathlib import Path
import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.market.nasdaq_api import (fetch_nasdaq_first_available, fetch_nasdaq_timeseries)


start = dt.date(2026, 1, 1)
end   = dt.date(2026, 1, 27)

base_idx = pd.date_range(start=start, end=end, freq="D", name="date")
market = pd.DataFrame(index=base_idx)

spgsci = fetch_nasdaq_timeseries("FRED/SPGSCI", start, end, force_refresh=True)
print(spgsci)
