import datetime as dt

import numpy as np
import pandas as pd
import xarray as xr

from data_pipeline.modules.climate.terraclimate_api import TerraClimateClient


class FakeResponse:
    status_code = 200

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        return [b"netcdf-content"]


class FakeHttpClient:
    def __init__(self):
        self.head_urls = []
        self.get_urls = []

    def head(self, url, **kwargs):
        self.head_urls.append(url)
        return FakeResponse()

    def get(self, url, **kwargs):
        self.get_urls.append(url)
        return FakeResponse()


def test_to_daily_dataframe_forward_fills_monthly_values():
    data = xr.DataArray(
        np.array([2.0, 5.0]),
        coords={
            "time": pd.to_datetime(["2020-01-01", "2020-02-01"]),
        },
        dims="time",
    )

    result = TerraClimateClient()._to_daily_dataframe(
        data,
        dt.date(2020, 1, 2),
        dt.date(2020, 2, 2),
    )

    assert result.index[0] == pd.Timestamp("2020-01-02")
    assert result.index[-1] == pd.Timestamp("2020-02-02")
    assert result.loc["2020-01-15", "scpdsi"] == 2.0
    assert result.loc["2020-02-02", "scpdsi"] == 5.0


def test_get_scpdsi_rejects_reversed_period():
    client = TerraClimateClient()

    try:
        client.get_scpdsi(
            dt.date(2020, 2, 1),
            dt.date(2020, 1, 1),
            {},
        )
    except ValueError as error:
        assert str(error) == "A data inicial deve ser anterior à data final."
    else:
        raise AssertionError("Era esperada uma exceção para período invertido")


def test_load_year_removes_downloaded_file(tmp_path, monkeypatch):
    http_client = FakeHttpClient()
    client = TerraClimateClient(
        data_dir=str(tmp_path),
        http_client=http_client,
    )
    expected = xr.DataArray([1.0], coords={"time": ["2020-01-01"]}, dims="time")
    monkeypatch.setattr(client, "_read_region", lambda path, bbox: expected)

    result = client._load_year(2020, {})

    downloaded_file = tmp_path / "TerraClimate_pdsi_2020.nc"
    assert result.identical(expected)
    assert not downloaded_file.exists()
    assert http_client.head_urls == [client._build_url(2020)]
    assert http_client.get_urls == [client._build_url(2020)]
