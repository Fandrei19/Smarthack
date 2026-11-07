import pandas as pd
import pytest

from src.multi_point_monitor import validate_predictions


def create_valid_predictions():
    return pd.DataFrame([
        {
            "timestamp_utc": "2026-11-07T12:00:00Z",
            "segment_id": "SEG_1",
            "q05_ampacity_a": 800.0,
            "q50_ampacity_a": 850.0,
            "q95_ampacity_a": 900.0,
            "data_quality": "ok"
        }
    ])


def test_valid_data_quality():
    predictions = create_valid_predictions()

    result = validate_predictions(predictions)

    assert result["is_valid"] is True


@pytest.mark.parametrize(
    "quality",
    ["warning", "missing", "error", "invalid"]
)
def test_invalid_data_quality(quality):
    predictions = create_valid_predictions()
    predictions.loc[0, "data_quality"] = quality

    result = validate_predictions(predictions)

    assert result["is_valid"] is False
    assert any("Data quality" in error for error in result["errors"])


@pytest.mark.parametrize("quality", [None, ""])
def test_missing_data_quality(quality):
    predictions = create_valid_predictions()
    predictions.loc[0, "data_quality"] = quality

    result = validate_predictions(predictions)

    assert result["is_valid"] is False
    assert any("data quality" in error.lower() for error in result["errors"])


def test_data_quality_is_case_insensitive():
    predictions = create_valid_predictions()
    predictions.loc[0, "data_quality"] = " OK "

    result = validate_predictions(predictions)

    assert result["is_valid"] is True