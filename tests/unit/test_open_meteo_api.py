import datetime as dt

import pandas as pd
import pytest
import requests

from data_pipeline.modules.climate.open_meteo_api import OpenMeteoClient


class FakeResponse:
    def __init__(self, data=None, error=None):
        self.data = data
        self.error = error

    def raise_for_status(self):
        if self.error:
            raise self.error

    def json(self):
        return self.data


class FakeHttpClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


def test_get_precipitation_builds_request_and_dataframe():
    response = FakeResponse(
        {
            "daily": {
                "time": ["2020-01-01", "2020-01-02"],
                "precipitation_sum": [1.5, 0.0],
            }
        }
    )
    http_client = FakeHttpClient(response)
    client = OpenMeteoClient(http_client=http_client)

    result = client.get_precipitation(
        lat=-15.78,
        lon=-47.92,
        start_date=dt.date(2020, 1, 1),
        end_date=dt.date(2020, 1, 2),
    )

    assert list(result.columns) == ["precipitation_mm"]
    assert result.index.equals(pd.DatetimeIndex(["2020-01-01", "2020-01-02"]))
    assert result["precipitation_mm"].tolist() == [1.5, 0.0]
    assert http_client.calls == [
        (
            client._URL,
            {
                "params": {
                    "latitude": -15.78,
                    "longitude": -47.92,
                    "start_date": "2020-01-01",
                    "end_date": "2020-01-02",
                    "daily": "precipitation_sum",
                    "timezone": "UTC",
                },
                "timeout": 60,
            },
        )
    ]


def test_get_precipitation_propagates_http_errors():
    error = requests.HTTPError("Open-Meteo unavailable")
    http_client = FakeHttpClient(FakeResponse(error=error))
    client = OpenMeteoClient(http_client=http_client)

    with pytest.raises(requests.HTTPError, match="Open-Meteo unavailable"):
        client.get_precipitation(-15.78, -47.92, "2020-01-01", "2020-01-02")
