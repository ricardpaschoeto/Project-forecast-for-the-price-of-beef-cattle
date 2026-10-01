import datetime as dt

import pandas as pd
import pytest
import requests

from data_pipeline.modules.climate.inmet_api import InmetClient


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


def test_get_daily_builds_request_and_dataframe():
    response = FakeResponse(
        {
            "daily": {
                "time": ["2026-09-01", "2026-09-02"],
                "temperature_2m_mean": [24.5, 25.0],
            }
        }
    )
    http_client = FakeHttpClient(response)
    client = InmetClient(http_client=http_client)

    result = client.get_daily(
        latitude=-16.6869,
        longitude=-49.2648,
        start=dt.date(2026, 9, 1),
        end=dt.date(2026, 9, 2),
    )

    assert list(result.columns) == ["tmed"]
    assert result.index.equals(pd.DatetimeIndex(["2026-09-01", "2026-09-02"]))
    assert result["tmed"].tolist() == [24.5, 25.0]
    assert http_client.calls == [
        (
            client._URL,
            {
                "params": {
                    "latitude": -16.6869,
                    "longitude": -49.2648,
                    "start_date": "2026-09-01",
                    "end_date": "2026-09-02",
                    "daily": ["temperature_2m_mean"],
                    "timezone": "America/Sao_Paulo",
                },
                "timeout": 60,
            },
        )
    ]


def test_get_daily_returns_empty_dataframe_without_daily_payload():
    client = InmetClient(http_client=FakeHttpClient(FakeResponse({})))

    result = client.get_daily(
        -16.6869,
        -49.2648,
        dt.date(2026, 9, 1),
        dt.date(2026, 9, 2),
    )

    assert result.empty


def test_get_daily_rejects_reversed_period():
    client = InmetClient()

    with pytest.raises(ValueError, match="A data inicial deve ser anterior"):
        client.get_daily(
            -16.6869,
            -49.2648,
            dt.date(2026, 9, 2),
            dt.date(2026, 9, 1),
        )


def test_get_daily_propagates_http_errors():
    error = requests.HTTPError("Open-Meteo unavailable")
    client = InmetClient(
        http_client=FakeHttpClient(FakeResponse(error=error))
    )

    with pytest.raises(requests.HTTPError, match="Open-Meteo unavailable"):
        client.get_daily(
            -16.6869,
            -49.2648,
            dt.date(2026, 9, 1),
            dt.date(2026, 9, 2),
        )
