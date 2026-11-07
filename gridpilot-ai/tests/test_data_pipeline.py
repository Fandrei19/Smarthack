from urllib.parse import parse_qs, urlsplit

import pandas as pd
import pytest

from src import data_pipeline as pipeline


@pytest.fixture
def nasa_payload():
    units = {"T2M": "C", "WS10M": "m/s", "WD10M": "Degrees", "ALLSKY_SFC_SW_DWN": "Wh/m^2"}
    return {
        "header": {"time_standard": "UTC", "fill_value": -999.0},
        "parameters": {key: {"units": unit} for key, unit in units.items()},
        "properties": {"parameter": {
            "T2M": {"2025071501": -999.0, "2025071500": 22.47},
            "WS10M": {"2025071500": 4.0, "2025071501": 3.0},
            "WD10M": {"2025071500": 162.0, "2025071501": 72.0},
            "ALLSKY_SFC_SW_DWN": {"2025071500": 12.0, "2025071501": 0.0},
        }},
    }


def test_nasa_weather_export_schema_and_missing_value(nasa_payload):
    weather = pipeline.add_wind_features(pipeline.parse_nasa_weather(nasa_payload), 72)
    assert weather.index[0] == pd.Timestamp("2025-07-15T00:00:00Z")
    rows = pipeline.model_weather_rows(weather, "SEG_1", "NASA_POWER")
    assert list(rows.columns) == [
        "timestamp_utc", "segment_id", "air_temp_c", "wind_speed_ms",
        "wind_direction_deg", "solar_radiation", "source", "data_quality",
    ]
    assert rows["solar_radiation"].tolist() == [12.0, 0.0]
    assert pd.isna(rows.loc[1, "air_temp_c"])
    assert rows["data_quality"].tolist() == ["ok", "missing"]


def test_nasa_rejects_unexpected_units(nasa_payload):
    nasa_payload["parameters"]["WS10M"]["units"] = "km/h"
    with pytest.raises(ValueError, match="Unexpected NASA POWER unit"):
        pipeline.parse_nasa_weather(nasa_payload)


def test_nasa_rejects_local_solar_time(nasa_payload):
    nasa_payload["header"]["time_standard"] = "LST"
    with pytest.raises(ValueError, match="must be in UTC"):
        pipeline.parse_nasa_weather(nasa_payload)


@pytest.mark.parametrize(("direction", "angle", "crosswind"), [
    (72, 0, 0), (252, 0, 0), (342, 90, 4), (12, 60, 3.464101615),
])
def test_wind_angle_and_crosswind(direction, angle, crosswind):
    weather = pd.DataFrame({"wind_from_deg": [direction], "wind_speed_ms": [4]})
    result = pipeline.add_wind_features(weather, 72).iloc[0]
    assert result["wind_angle_deg"] == pytest.approx(angle)
    assert result["crosswind_speed_ms"] == pytest.approx(crosswind)


def test_forecast_requests_humidity_and_aligns_missing_hours(monkeypatch):
    def mock_response(url):
        query = parse_qs(urlsplit(url).query)
        assert "relative_humidity_2m" in query["hourly"][0].split(",")
        assert query["wind_speed_unit"] == ["ms"]
        assert query["timezone"] == ["GMT"]
        return {
            "latitude": 44.38,
            "longitude": 28.14,
            "hourly_units": {
                "temperature_2m": "°C", "wind_speed_10m": "m/s",
                "wind_direction_10m": "°", "shortwave_radiation": "W/m²",
                "relative_humidity_2m": "%",
            },
            "hourly": {
                "time": ["2025-07-15T00:00"], "temperature_2m": [20.0],
                "wind_speed_10m": [3.0], "wind_direction_10m": [90],
                "shortwave_radiation": [0.0], "relative_humidity_2m": [82],
            },
        }

    monkeypatch.setattr(pipeline, "get_json", mock_response)
    grid = pd.date_range("2025-07-15", periods=2, freq="h", tz="UTC")
    weather, _, _ = pipeline.fetch_hourly(pipeline.FORECAST, 44.38, 28.14, grid)
    assert weather["relative_humidity_pct"].iloc[0] == 82
    assert weather.iloc[1].isna().all()
