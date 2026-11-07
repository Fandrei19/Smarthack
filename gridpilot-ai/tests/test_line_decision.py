
import pytest

from src.decision_engine import evaluate_line_decision


def test_low_risk_decision():
    result = evaluate_line_decision(
        line_rating_q05_a=1000,
        static_rating_a=700,
        monitoring_status="VALID"
    )

    assert result["risk_level"] == "LOW"
    assert result["action"] == "NORMAL_MONITORING"
    assert result["requires_human_review"] is False


def test_medium_risk_decision():
    result = evaluate_line_decision(
        line_rating_q05_a=800,
        static_rating_a=700,
        monitoring_status="VALID"
    )

    assert result["risk_level"] == "MEDIUM"
    assert result["action"] == "INCREASE_MONITORING"


def test_high_risk_decision():
    result = evaluate_line_decision(
        line_rating_q05_a=700,
        static_rating_a=700,
        monitoring_status="VALID"
    )

    assert result["risk_level"] == "HIGH"
    assert result["action"] == "REQUEST_OPERATOR_REVIEW"
    assert result["requires_human_review"] is True


def test_incomplete_monitoring():
    result = evaluate_line_decision(
        line_rating_q05_a=None,
        static_rating_a=700,
        monitoring_status="INCOMPLETE"
    )

    assert result["risk_level"] == "UNKNOWN"
    assert result["utilization"] is None
    assert result["action"] == "REQUEST_DATA_VALIDATION"
    assert result["requires_human_review"] is True


def test_invalid_monitoring_status():
    with pytest.raises(ValueError):
        evaluate_line_decision(
            line_rating_q05_a=800,
            static_rating_a=700,
            monitoring_status="INVALID"
        )
