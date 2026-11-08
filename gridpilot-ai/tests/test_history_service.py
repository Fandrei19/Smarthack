import pandas as pd
import pytest
from src.history_service import read_history, filter_history


def test_history_normalizes_utc_and_removes_duplicate_and_invalid_times(tmp_path):
    path = tmp_path / "weather.csv"
    path.write_text("timestamp_utc,segment_id,air_temp_c,wind_speed_ms,wind_direction_deg,solar_radiation,source,data_quality\n"
                    "2025-07-15T03:00:00+03:00,SEG_1,20,2,90,0,NASA_POWER,ok\n"
                    "2025-07-15T00:00:00Z,SEG_1,20,2,90,0,NASA_POWER,ok\n"
                    "invalid,SEG_1,20,2,90,0,NASA_POWER,ok\n")
    result = read_history(path)
    assert len(result) == 1
    assert result.iloc[0].timestamp_utc == pd.Timestamp("2025-07-15T00:00:00Z")
    assert result.iloc[0].source == "NASA_POWER"


def test_history_rejects_wrong_schema(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("timestamp_utc,value\n2025-07-15,1\n")
    with pytest.raises(ValueError, match="Missing weather columns"):
        read_history(path)


def test_date_range_includes_last_day_and_filters_segments():
    frame = pd.DataFrame({"timestamp_utc": pd.to_datetime([
        "2025-07-15T23:00Z", "2025-07-16T00:00Z", "2025-07-15T12:00Z"], utc=True),
        "segment_id": ["SEG_1", "SEG_1", "SEG_2"]})
    result = filter_history(frame, "2025-07-15", "2025-07-15", ["SEG_1"])
    assert len(result) == 1
    assert result.iloc[0].timestamp_utc.hour == 23
