
import pandas as pd

from src.data_validator import validate_predictions


def test_valid_predictions():
    predictions = pd.DataFrame([
        {
            "timestamp_utc": "2026-11-07T12:00:00Z",
            "segment_id": "S1",
            "q05_ampacity_a": 800,
            "q50_ampacity_a": 850,
            "q95_ampacity_a": 900
        }
    ])

    result = validate_predictions(predictions)

    assert result["is_valid"] is True
    assert result["errors"] == []


def test_missing_required_column():
    predictions = pd.DataFrame([
        {
            "timestamp_utc": "2026-11-07T12:00:00Z",
            "segment_id": "S1",
            "q05_ampacity_a": 800
        }
    ])

    result = validate_predictions(predictions)

    assert result["is_valid"] is False
    assert any("Missing required columns" in error for error in result["errors"])


def test_invalid_quantile_order():
    predictions = pd.DataFrame([
        {
            "timestamp_utc": "2026-11-07T12:00:00Z",
            "segment_id": "S1",
            "q05_ampacity_a": 900,
            "q50_ampacity_a": 850,
            "q95_ampacity_a": 950
        }
    ])

    result = validate_predictions(predictions)

    assert result["is_valid"] is False
    assert any("not correctly ordered" in error for error in result["errors"])
