import pandas as pd

from src.multi_point_monitor import validate_predictions


def create_valid_predictions():
    return pd.DataFrame([
        {
            "timestamp_utc": "2026-11-07T12:00:00Z",
            "segment_id": "SEG_1",
            "q05_ampacity_a": 800.0,
            "q50_ampacity_a": 850.0,
            "q95_ampacity_a": 900.0
        },
        {
            "timestamp_utc": "2026-11-07T12:00:00Z",
            "segment_id": "SEG_2",
            "q05_ampacity_a": 810.0,
            "q50_ampacity_a": 860.0,
            "q95_ampacity_a": 910.0
        }
    ])


def test_duplicate_timestamp_and_segment():
    predictions = create_valid_predictions()

    duplicate_row = predictions.iloc[[0]]
    predictions = pd.concat(
        [predictions, duplicate_row],
        ignore_index=True
    )

    result = validate_predictions(predictions)

    assert result["is_valid"] is False
    assert any("Duplicate predictions" in error for error in result["errors"])


def test_same_timestamp_different_segments_is_valid():
    predictions = create_valid_predictions()

    result = validate_predictions(predictions)

    assert result["is_valid"] is True