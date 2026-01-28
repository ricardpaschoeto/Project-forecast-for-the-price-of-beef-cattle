import datetime as dt
import pandas as pd
import xarray as xr


def get_terraclimate_scpdsi(start: dt.date, end: dt.date, bbox: dict, scpdsi_nc_path: str) -> pd.DataFrame:
    """
    Retorna o índice scPDSI diário interpolado (mensal → diário via ffill)
    a partir de arquivo TerraClimate nc.
    bbox = { 'min_lon': , 'max_lon': , 'min_lat': , 'max_lat': }
    """

    ds = xr.open_dataset(scpdsi_nc_path)

    ds_reg = ds.sel(
        lon=slice(bbox["min_lon"], bbox["max_lon"]),
        lat=slice(bbox["min_lat"], bbox["max_lat"]),
    )

    scpdsi_reg = ds_reg["scpdsi"].mean(dim=("lat", "lon"))

    df_monthly = scpdsi_reg.to_dataframe().reset_index()
    df_monthly["time"] = pd.to_datetime(df_monthly["time"])
    df_monthly = df_monthly.set_index("time").sort_index()
    df_monthly = df_monthly.loc[start:end]

    idx_daily = pd.date_range(start=start, end=end, freq="D")
    df_daily = df_monthly.resample("D").ffill().reindex(idx_daily)
    df_daily.index.name = "date"

    df_daily.rename(columns={"scpdsi": "scpdsi"}, inplace=True)

    return df_daily[["scpdsi"]]