import datetime as dt
from pathlib import Path

import pandas as pd
import requests
import xarray as xr


def get_terraclimate_scpdsi(
    start: dt.date,
    end: dt.date,
    bbox: dict,
    data_dir: str = "data_pipeline/sensors",
) -> pd.DataFrame:
    """
    Retorna scPDSI diário do TerraClimate.

    Caso o período solicitado seja posterior ao último dado
    disponível, o último valor é propagado via forward fill.
    """

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    datasets = []

    current_year = dt.date.today().year

    print("Localizando arquivos disponíveis...")

    for year in range(start.year - 1, current_year + 1):

        url = (
            "https://climate.northwestknowledge.net/"
            f"TERRACLIMATE-DATA/TerraClimate_pdsi_{year}.nc"
        )

        nc_file = data_path / f"TerraClimate_pdsi_{year}.nc"

        # se já baixado
        if nc_file.exists():
            datasets.append(xr.open_dataset(nc_file, engine="netcdf4"))
            continue

        try:

            r = requests.head(url, timeout=30, allow_redirects=True)

            if r.status_code >= 400:
                print(f"Ano {year} indisponível.")
                continue

            print(f"Baixando {year}...")

            r = requests.get(
                url,
                stream=True,
                timeout=300,
            )

            r.raise_for_status()

            with open(nc_file, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)

            datasets.append(xr.open_dataset(nc_file, engine="netcdf4"))

        except Exception as e:
            print(f"Erro ao baixar {year}: {e}")

    if not datasets:
        raise RuntimeError(
            "Nenhum arquivo TerraClimate foi encontrado."
        )

    # concatena todos os anos disponíveis
    ds = xr.concat(datasets, dim="time")

    print("Variáveis encontradas:")
    print(list(ds.data_vars))

    # normalmente TerraClimate usa pdsi
    var_name = "PDSI"

    ds_reg = ds.sel(
        lon=slice(bbox["min_lon"], bbox["max_lon"]),
        lat=slice(bbox["max_lat"], bbox["min_lat"]),
    )

    pdsi_reg = ds_reg[var_name].mean(
        dim=("lat", "lon")
    )

    df_monthly = (
        pdsi_reg.to_dataframe()
        .reset_index()
    )

    df_monthly["time"] = pd.to_datetime(
        df_monthly["time"]
    )

    df_monthly = (
        df_monthly
        .set_index("time")
        .sort_index()
    )

    last_available = df_monthly[var_name].dropna().index.max()

    print(
        f"Última data disponível TerraClimate: "
        f"{last_available:%Y-%m-%d}"
    )

    # cria série diária até a maior data entre
    # o pedido do usuário e o último dado disponível
    daily_index = pd.date_range(
        start=df_monthly.index.min(),
        end=max(
            pd.Timestamp(end),
            last_available,
        ),
        freq="D",
    )

    df_daily = (
        df_monthly
        .resample("D")
        .ffill()
        .reindex(daily_index)
        .ffill()
    )

    # somente o período solicitado
    df_daily = df_daily.loc[
        pd.Timestamp(start):pd.Timestamp(end)
    ]

    df_daily.index.name = "date"

    df_daily.rename(
        columns={
            var_name: "scpdsi"
        },
        inplace=True,
    )

    return df_daily[["scpdsi"]]


def main():

    start = dt.date(2026, 9, 1)
    end = dt.date(2026, 9, 30)

    bbox = {
        "min_lon": -53.5,
        "max_lon": -45.8,
        "min_lat": -19.5,
        "max_lat": -12.0,
    }

    df = get_terraclimate_scpdsi(
        start=start,
        end=end,
        bbox=bbox,
    )

    print(df)


if __name__ == "__main__":
    main()