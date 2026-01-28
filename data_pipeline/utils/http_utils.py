import time
import requests

def http_get_json(url: str, params: dict, max_tries=6, backoff_s=2.0):
    last_exc = None
    for attempt in range(1, max_tries + 1):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff_s * attempt)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_exc = e
            time.sleep(backoff_s * attempt)
    raise last_exc