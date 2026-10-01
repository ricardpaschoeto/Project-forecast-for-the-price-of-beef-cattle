import datetime as dt

import pandas as pd
import pytest

from data_pipeline.modules.climate.climate_builder import ClimateBuilder


BBOX = {
    "min_lat": -20.0,
    "max_lat": -10.0,
    "min_lon": -50.0,
    "max_lon": -40.0,
}


class FakeInmetClient:
    def __init__(self):
        self.calls = []

    def get_daily(self, **kwargs):
        self.calls.append(kwargs)
        return pd.DataFrame(
            {"tmed": [24.0, 25.0]},
            index=pd.to_datetime(["2026-01-01", "2026-01-02"]),
        )


class FakeTerraClimateClient:
    def get_scpdsi(self, **kwargs):
        return pd.DataFrame(
            {"scpdsi": [-1.0, -0.5]},
            index=pd.to_datetime(["2026-01-01", "2026-01-02"]),
        )


class FakeNoaaClient:
    def get_oni(self, start, end):
        return pd.DataFrame(
            {"el_nino": [0.2, 0.3]},
            index=pd.to_datetime(["2026-01-01", "2026-01-02"]),
        )


class FakePrecipitationClient:
    def get_precipitation(self, **kwargs):
        return pd.DataFrame(
            {"precipitation_mm": [5.0, 0.0]},
            index=pd.to_datetime(["2026-01-01", "2026-01-02"]),
        )


def test_build_composes_sources_and_normalizes_schema():
    inmet_client = FakeInmetClient()
    builder = ClimateBuilder(
        inmet_client=inmet_client,
        terraclimate_client=FakeTerraClimateClient(),
        noaa_client=FakeNoaaClient(),
        precipitation_client=FakePrecipitationClient(),
    )

    result = builder.build(
        start=dt.date(2026, 1, 1),
        end=dt.date(2026, 1, 3),
        estacoes_inmet=["A123"],
        terraclimate_bbox=BBOX,
    )

    assert list(result.columns) == builder.CLIMATE_COLUMNS
    assert result.index.equals(
        pd.date_range("2026-01-01", "2026-01-03", freq="D", name="date")
    )
    assert result.loc["2026-01-01"].to_dict() == {
        "precip_total_mm": 5.0,
        "TEMPERATURA": 24.0,
        "scpdsi": -1.0,
        "el nino": 0.2,
    }
    assert result.loc["2026-01-03"].isna().all()
    assert inmet_client.calls == [
        {
            "latitude": -15.0,
            "longitude": -45.0,
            "start": dt.date(2026, 1, 1),
            "end": dt.date(2026, 1, 3),
        }
    ]


def test_build_keeps_schema_when_a_source_is_empty():
    empty_client = FakeTerraClimateClient()
    empty_client.get_scpdsi = lambda **kwargs: pd.DataFrame()
    builder = ClimateBuilder(
        inmet_client=FakeInmetClient(),
        terraclimate_client=empty_client,
        noaa_client=FakeNoaaClient(),
        precipitation_client=FakePrecipitationClient(),
    )

    result = builder.build(
        dt.date(2026, 1, 1),
        dt.date(2026, 1, 2),
        [],
        BBOX,
    )

    assert list(result.columns) == builder.CLIMATE_COLUMNS
    assert result["scpdsi"].isna().all()


def test_build_rejects_invalid_period_and_bbox():
    builder = ClimateBuilder()

    with pytest.raises(ValueError, match="A data inicial deve ser anterior"):
        builder.build(dt.date(2026, 1, 2), dt.date(2026, 1, 1), [], BBOX)

    with pytest.raises(ValueError, match="Bounding box sem campos obrigatórios"):
        builder.build(
            dt.date(2026, 1, 1),
            dt.date(2026, 1, 2),
            [],
            {"min_lat": -20.0},
        )
