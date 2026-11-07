
import pytest

from src.decision_engine import classify_risk


def test_low_risk():
    result = classify_risk(0.75)

    assert result["risk_level"] == "LOW"
    assert result["action"] == "NORMAL_MONITORING"
    assert result["requires_human_review"] is False


def test_medium_risk():
    result = classify_risk(0.85)

    assert result["risk_level"] == "MEDIUM"
    assert result["action"] == "INCREASE_MONITORING"
    assert result["requires_human_review"] is False


def test_high_risk():
    result = classify_risk(0.98)

    assert result["risk_level"] == "HIGH"
    assert result["action"] == "REQUEST_OPERATOR_REVIEW"
    assert result["requires_human_review"] is True


def test_negative_utilization():
    with pytest.raises(ValueError):
        classify_risk(-0.1)
