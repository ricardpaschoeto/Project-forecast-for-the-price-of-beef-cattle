import datetime as dt
import os
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
import xarray as xr


class TerraClimateClient:
    """Cliente para baixar e preparar o índice scPDSI do TerraClimate."""

    _BASE_URL = "https://climate.northwestknowledge.net/TERRACLIMATE-DATA"
    _VARIABLE = "PDSI"

    def __init__(
        self,
        data_dir: str = "data_pipeline/sensors",
        http_client: Any = requests,
        today: Optional[dt.date] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.http_client = http_client
        self.today = today or dt.date.today()

    def get_scpdsi(
        self,
        start: dt.date,
        end: dt.date,
        bbox: dict,
    ) -> pd.DataFrame:
        """Retorna o scPDSI diário para o período e a região solicitados."""
        if start > end:
            raise ValueError("A data inicial deve ser anterior à data final.")

        self.data_dir.mkdir(parents=True, exist_ok=True)
        datasets = self._load_available_years(start, bbox)
        if not datasets:
            raise RuntimeError("Nenhum arquivo TerraClimate foi encontrado.")

        return self._to_daily_dataframe(xr.concat(datasets, dim="time"), start, end)

    def _load_available_years(self, start: dt.date, bbox: dict) -> list[xr.DataArray]:
        datasets = []
        print("Localizando arquivos disponíveis...")

        for year in range(start.year - 1, self.today.year + 1):
            try:
                datasets.append(self._load_year(year, bbox))
            except Exception as error:
                print(f"Erro ao baixar {year}: {error}")

        return datasets

    def _load_year(self, year: int, bbox: dict) -> xr.DataArray:
        nc_file = self.data_dir / f"TerraClimate_pdsi_{year}.nc"
        downloaded = False

        if not nc_file.exists():
            url = self._build_url(year)
            response = self.http_client.head(url, timeout=30, allow_redirects=True)
            if response.status_code >= 400:
                print(f"Ano {year} indisponível.")
                raise FileNotFoundError(year)

            print(f"Baixando {year}...")
            response = self.http_client.get(url, stream=True, timeout=300)
            response.raise_for_status()
            with nc_file.open("wb") as file:
                for chunk in response.iter_content(8192):
                    file.write(chunk)
            downloaded = True

        try:
            return self._read_region(nc_file, bbox)
        finally:
            if downloaded:
                try:
                    os.remove(nc_file)
                except OSError as error:
                    print(f"Erro ao remover {nc_file}: {error}")

    def _build_url(self, year: int) -> str:
        return f"{self._BASE_URL}/TerraClimate_pdsi_{year}.nc"

    def _read_region(self, nc_file: Path, bbox: dict) -> xr.DataArray:
        dataset = xr.open_dataset(nc_file, engine="netcdf4")
        try:
            regional = dataset.sel(
                lon=slice(bbox["min_lon"], bbox["max_lon"]),
                lat=slice(bbox["max_lat"], bbox["min_lat"]),
            )
            return regional[self._VARIABLE].mean(dim=("lat", "lon")).load()
        finally:
            dataset.close()

    def _to_daily_dataframe(
        self,
        data: xr.DataArray,
        start: dt.date,
        end: dt.date,
    ) -> pd.DataFrame:
        df_monthly = data.to_dataframe(name=self._VARIABLE).reset_index()
        df_monthly["time"] = pd.to_datetime(df_monthly["time"])
        df_monthly = df_monthly.set_index("time").sort_index()
        last_available = df_monthly[self._VARIABLE].dropna().index.max()

        print(f"Última data disponível TerraClimate: {last_available:%Y-%m-%d}")
        daily_index = pd.date_range(
            start=df_monthly.index.min(),
            end=max(pd.Timestamp(end), last_available),
            freq="D",
        )
        df_daily = (
            df_monthly.resample("D").ffill().reindex(daily_index).ffill()
        )
        result = df_daily.loc[pd.Timestamp(start):pd.Timestamp(end)].copy()
        result.index.name = "date"
        return result.rename(columns={self._VARIABLE: "scpdsi"})[["scpdsi"]]


def get_terraclimate_scpdsi(
    start: dt.date,
    end: dt.date,
    bbox: dict,
    data_dir: str = "data_pipeline/sensors",
) -> pd.DataFrame:
    """Mantém a API funcional usada pelos pipelines existentes."""
    return TerraClimateClient(data_dir=data_dir).get_scpdsi(start, end, bbox)


def main():

    start = dt.date(2020, 1, 1)
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