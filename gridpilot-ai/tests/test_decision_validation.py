import pytest

from src.decision_engine import evaluate_line_decision


@pytest.mark.parametrize(("dynamic_rating", "static_rating"), [
    (800, None),
    (800, 0),
    (-100, 700),
    (None, 700),
])
def test_invalid_ratings_trigger_static_fallback(dynamic_rating, static_rating):
    result = evaluate_line_decision(
        line_rating_q05_a=dynamic_rating,
        static_rating_a=static_rating,
        monitoring_status="VALID",
    )

    assert result["risk_level"] == "UNCERTAIN"
    assert result["action"] == "FALLBACK_STATIC"
    assert result["reason_code"] == "INVALID_RATING_INPUTS"
    assert result["utilization"] is None
    assert result["requires_human_review"] is True
