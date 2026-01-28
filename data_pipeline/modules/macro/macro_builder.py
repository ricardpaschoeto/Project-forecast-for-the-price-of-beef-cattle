import datetime as dt
import numpy as np
import pandas as pd

from utils.date_utils import ensure_date_range_df
from modules.macro.bcb_api import get_bcb_series_raw


def get_macro_data(start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    Constrói dataframe macroeconômico diário contendo:

    - selic (SGS 432)  → meta Selic ao ano
    - ipca (SGS 433)   → variação mensal do IPCA
    - pib_brasil       → stub (NaN)
    - pib_agro         → stub (NaN)

    Usa ensure_date_range_df para expandir e ffill.
    """

    base_idx = pd.date_range(start=start, end=end, freq="D")
    macro = pd.DataFrame(index=base_idx)
    macro.index.name = "date"

    # IDs comuns das séries SGS (validar se quiser atualizar)
    series_map = {
        "selic": 432,   # Meta Selic
        "ipca": 433,    # IPCA var. mensal
    }

    for name, series_id in series_map.items():
        try:
            df_raw = get_bcb_series_raw(series_id, start, end)
        except Exception:
            df_raw = pd.DataFrame(columns=["value"])

        if df_raw.empty:
            macro[name] = np.nan
        else:
            df = df_raw.rename(columns={"value": name})
            df = ensure_date_range_df(df, start, end)
            macro[name] = df[name]

    # Stubs: você pode substituir por PIB trimestral (IBGE) interpolado
    macro["pib_brasil"] = np.nan
    macro["pib_agro"] = np.nan

    macro = ensure_date_range_df(macro, start, end)
    return macro