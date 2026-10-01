import datetime as dt

import pandas as pd
import pytest
import requests

from data_pipeline.modules.climate.noaa_api import NoaaClient


NOAA_CONTENT = """SEAS YR TOTAL ANOM
DJF  2020  10.0  0.5
JFM  2020  11.0  0.7
"""


class FakeResponse:
    def __init__(self, text="", error=None):
        self.text = text
        self.error = error

    def raise_for_status(self):
        if self.error:
            raise self.error


class FakeHttpClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


def test_get_oni_parses_seasons_and_forward_fills_daily_values():
    client = NoaaClient(
        http_client=FakeHttpClient(FakeResponse(text=NOAA_CONTENT))
    )

    result = client.get_oni(
        dt.date(2020, 1, 15),
        dt.date(2020, 2, 2),
    )

    assert result.index.equals(
        pd.date_range("2020-01-15", "2020-02-02", freq="D")
    )
    assert result.loc["2020-01-15", "el_nino"] == 0.5
    assert result.loc["2020-02-02", "el_nino"] == 0.5


def test_get_oni_uses_noaa_endpoint_and_timeout():
    http_client = FakeHttpClient(FakeResponse(text=NOAA_CONTENT))
    client = NoaaClient(http_client=http_client)

    client.get_oni(dt.date(2020, 1, 15), dt.date(2020, 1, 16))

    assert http_client.calls == [(client._URL, {"timeout": 30})]


def test_get_oni_rejects_reversed_period():
    client = NoaaClient()

    with pytest.raises(ValueError, match="A data inicial deve ser anterior"):
        client.get_oni(dt.date(2020, 2, 1), dt.date(2020, 1, 1))


def test_get_oni_propagates_http_errors():
    error = requests.HTTPError("NOAA unavailable")
    client = NoaaClient(
        http_client=FakeHttpClient(FakeResponse(error=error))
    )

    with pytest.raises(requests.HTTPError, match="NOAA unavailable"):
        client.get_oni(dt.date(2020, 1, 15), dt.date(2020, 1, 16))
