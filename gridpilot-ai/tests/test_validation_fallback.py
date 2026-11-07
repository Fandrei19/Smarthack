import pandas as pd

from src.multi_point_monitor import validate_predictions
from src.decision_engine import evaluate_line_decision


def create_invalid_predictions():
    """Create prediction data with an invalid quantile order."""
    return pd.DataFrame([
        {
            "timestamp_utc": "2026-11-07T12:00:00Z",
            "segment_id": "SEG_1",
            "q05_ampacity_a": 900.0,
            "q50_ampacity_a": 850.0,
            "q95_ampacity_a": 950.0
        }
    ])


def test_invalid_predictions_trigger_fallback():
    predictions = create_invalid_predictions()

    # Validate the predictions.
    validation = validate_predictions(predictions)

    assert validation["is_valid"] is False

    # Do not calculate a dynamic rating from invalid data.
    decision = evaluate_line_decision(
        line_rating_q05_a=None,
        static_rating_a=800.0,
        monitoring_status="INCOMPLETE"
    )

    assert decision["risk_level"] == "UNCERTAIN"
    assert decision["utilization"] is None
    assert decision["action"] == "FALLBACK_STATIC"
    assert decision["requires_human_review"] is True