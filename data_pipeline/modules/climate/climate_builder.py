import datetime as dt
import pandas as pd
import numpy as np

from modules.climate.inmet_api import get_openmeteo_daily
from modules.climate.terraclimate_api import get_terraclimate_scpdsi
from modules.climate.noaa_api import get_noaa_oni
from modules.climate.open_meteo import get_precipitation


def get_climate_data(
    start: dt.date,
    end: dt.date,
    estacoes_inmet: list[str],
    terraclimate_bbox: dict,
    terraclimate_scpdsi_nc: str,
) -> pd.DataFrame:

    df_inmet = get_openmeteo_daily(start, end, estacoes_inmet)
    df_scpdsi = get_terraclimate_scpdsi(start, end, terraclimate_bbox, terraclimate_scpdsi_nc)
    df_elnino = get_noaa_oni(start, end)
    df_precipitation = get_precipitation(
        lat=(terraclimate_bbox["min_lat"] + terraclimate_bbox["max_lat"]) / 2,
        lon=(terraclimate_bbox["min_lon"] + terraclimate_bbox["max_lon"]) / 2,
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
    )

    df = df_inmet.join(df_scpdsi, how="outer")
    df = df.join(df_elnino, how="outer")
    df = df.join(df_precipitation, how="outer")

    for col in ["precip_total_mm", "TEMPERATURA", "scpdsi", "el nino"]:
        if col not in df.columns:
            df[col] = np.nan

    return df[["precip_total_mm", "TEMPERATURA", "scpdsi", "el nino"]]