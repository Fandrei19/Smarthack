
import pandas as pd
import pytest

from src.multi_point_monitor import validate_predictions


def create_valid_predictions():
    return pd.DataFrame([
        {
            "timestamp_utc": "2026-11-07T12:00:00Z",
            "segment_id": "S1",
            "q05_ampacity_a": 800.0,
            "q50_ampacity_a": 850.0,
            "q95_ampacity_a": 900.0
        }
    ])


@pytest.mark.parametrize(
    "invalid_value",
    [None, float("nan"), float("inf"), float("-inf"), 0, -100]
)
def test_invalid_ampacity_values(invalid_value):
    predictions = create_valid_predictions()
    predictions.loc[0, "q05_ampacity_a"] = invalid_value

    result = validate_predictions(predictions)

    assert result["is_valid"] is False

def test_invalid_timestamp():
    predictions = create_valid_predictions()
    predictions.loc[0, "timestamp_utc"] = "not-a-timestamp"

    result = validate_predictions(predictions)

    assert result["is_valid"] is False
    assert any("timestamps" in error for error in result["errors"])


def test_empty_segment_id():
    predictions = create_valid_predictions()
    predictions.loc[0, "segment_id"] = "   "

    result = validate_predictions(predictions)

    assert result["is_valid"] is False
    assert any("Empty segment identifiers" in error for error in result["errors"])


def test_empty_dataframe():
    predictions = pd.DataFrame()

    result = validate_predictions(predictions)

    assert result["is_valid"] is False
    assert any("empty" in error.lower() for error in result["errors"])
