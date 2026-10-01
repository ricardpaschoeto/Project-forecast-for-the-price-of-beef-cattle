import datetime as dt
from typing import Optional

import numpy as np
import pandas as pd

if __package__:
    from .inmet_api import InmetClient
    from .noaa_api import NoaaClient
    from .open_meteo_api import OpenMeteoClient
    from .terraclimate_api import TerraClimateClient
else:
    from inmet_api import InmetClient
    from noaa_api import NoaaClient
    from open_meteo_api import OpenMeteoClient
    from terraclimate_api import TerraClimateClient


class ClimateBuilder:
    """Orquestra as fontes e monta o DataFrame climático consolidado."""

    CLIMATE_COLUMNS = [
        "precip_total_mm",
        "TEMPERATURA",
        "scpdsi",
        "el nino",
    ]

    def __init__(
        self,
        inmet_client: Optional[InmetClient] = None,
        terraclimate_client: Optional[TerraClimateClient] = None,
        noaa_client: Optional[NoaaClient] = None,
        precipitation_client: Optional[OpenMeteoClient] = None,
        logger: Optional[Any] = None,
    ) -> None:
        self.inmet_client = inmet_client or InmetClient()
        self.terraclimate_client = terraclimate_client or TerraClimateClient()
        self.noaa_client = noaa_client or NoaaClient()
        self.precipitation_client = precipitation_client or OpenMeteoClient()
        self.logger = logger

    def build(
        self,
        start: dt.date,
        end: dt.date,
        estacoes_inmet: list[str],
        terraclimate_bbox: dict[str, float],
        terraclimate_scpdsi_nc: Optional[str] = None,
    ) -> pd.DataFrame:
        """Coleta, normaliza e combina os dados climáticos diários."""
        self._validate_period(start, end)
        latitude, longitude = self._get_bbox_center(terraclimate_bbox)

        frames = [
            self._get_temperature(start, end, latitude, longitude),
            self.terraclimate_client.get_scpdsi(
                start=start,
                end=end,
                bbox=terraclimate_bbox,
            ),
            self.noaa_client.get_oni(start, end),
            self.precipitation_client.get_precipitation(
                lat=latitude,
                lon=longitude,
                start_date=start.strftime("%Y-%m-%d"),
                end_date=end.strftime("%Y-%m-%d"),
            ),
        ]
        renamed_frames = [
            self._prepare_frame(frames[0], {"tmed": "TEMPERATURA"}),
            self._prepare_frame(frames[1], {}),
            self._prepare_frame(frames[2], {"el_nino": "el nino"}),
            self._prepare_frame(
                frames[3], {"precipitation_mm": "precip_total_mm"}
            ),
        ]
        return self._merge_frames(renamed_frames, start, end)

    def _get_temperature(
        self,
        start: dt.date,
        end: dt.date,
        latitude: float,
        longitude: float,
    ) -> pd.DataFrame:
        return self.inmet_client.get_daily(
            latitude=latitude,
            longitude=longitude,
            start=start,
            end=end,
        )

    def _prepare_frame(
        self,
        frame: pd.DataFrame,
        column_map: dict[str, str],
    ) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))

        prepared = frame.rename(columns=column_map).copy()
        prepared.index = pd.to_datetime(prepared.index)
        prepared.index.name = "date"
        return prepared.sort_index()

    def _merge_frames(
        self,
        frames: list[pd.DataFrame],
        start: dt.date,
        end: dt.date,
    ) -> pd.DataFrame:
        date_index = pd.date_range(start=start, end=end, freq="D", name="date")
        result = pd.DataFrame(index=date_index)
        for frame in frames:
            result = result.join(frame, how="left")

        for column in self.CLIMATE_COLUMNS:
            if column not in result.columns:
                result[column] = np.nan

        return result[self.CLIMATE_COLUMNS]

    def _get_bbox_center(self, bbox: dict[str, float]) -> tuple[float, float]:
        required_keys = {"min_lat", "max_lat", "min_lon", "max_lon"}
        missing_keys = required_keys.difference(bbox)
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"Bounding box sem campos obrigatórios: {missing}")

        latitude = (bbox["min_lat"] + bbox["max_lat"]) / 2
        longitude = (bbox["min_lon"] + bbox["max_lon"]) / 2
        return latitude, longitude

    def _validate_period(self, start: dt.date, end: dt.date) -> None:
        if start > end:
            raise ValueError("A data inicial deve ser anterior à data final.")


def get_climate_data(
    start: dt.date,
    end: dt.date,
    estacoes_inmet: list[str],
    terraclimate_bbox: dict[str, float],
    terraclimate_scpdsi_nc: Optional[str] = None,
) -> pd.DataFrame:
    """Fachada compatível para construir o bloco climático consolidado.

    ``estacoes_inmet`` e ``terraclimate_scpdsi_nc`` são mantidos para
    compatibilidade com o master builder; os clientes atuais usam o centro
    do bounding box e suas próprias configurações de dados.
    """
    return ClimateBuilder().build(
        start=start,
        end=end,
        estacoes_inmet=estacoes_inmet,
        terraclimate_bbox=terraclimate_bbox,
        terraclimate_scpdsi_nc=terraclimate_scpdsi_nc,
    )

def main():
    from utils.logger import PipelineLogger
    logger = PipelineLogger()
    start = dt.date(2020, 1, 1)
    end = dt.date(2026, 9, 30)

    bbox = {
        "min_lon": -53.5,
        "max_lon": -45.8,
        "min_lat": -19.5,
        "max_lat": -12.0,
    }

    df = get_climate_data(
        start=start,
        end=end,
        estacoes_inmet=["A123"],
        terraclimate_bbox=bbox,
        logger=logger,
    )

    print(df)

if __name__ == "__main__":
    main()