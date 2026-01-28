import datetime as dt
import pandas as pd
import numpy as np

from modules.climate.inmet_api import get_inmet_precip_temp
from modules.climate.terraclimate_api import get_terraclimate_scpdsi
from modules.climate.noaa_api import get_noaa_oni


def get_climate_data(
    start: dt.date,
    end: dt.date,
    estacoes_inmet: list[str],
    terraclimate_bbox: dict,
    terraclimate_scpdsi_nc: str,
) -> pd.DataFrame:

    df_inmet = get_inmet_precip_temp(start, end, estacoes_inmet)
    df_scpdsi = get_terraclimate_scpdsi(start, end, terraclimate_bbox, terraclimate_scpdsi_nc)
    df_elnino = get_noaa_oni(start, end)

    df = df_inmet.join(df_scpdsi, how="outer")
    df = df.join(df_elnino, how="outer")

    for col in ["precip_total_mm", "TEMPERATURA", "scpdsi", "el nino"]:
        if col not in df.columns:
            df[col] = np.nan

    return df[["precip_total_mm", "TEMPERATURA", "scpdsi", "el nino"]]