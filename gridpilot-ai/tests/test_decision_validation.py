
import pytest

from src.decision_engine import evaluate_line_decision


def test_missing_static_rating():
    with pytest.raises(ValueError):
        evaluate_line_decision(
            line_rating_q05_a=800,
            static_rating_a=None,
            monitoring_status="VALID"
        )


def test_zero_static_rating():
    with pytest.raises(ValueError):
        evaluate_line_decision(
            line_rating_q05_a=800,
            static_rating_a=0,
            monitoring_status="VALID"
        )


def test_negative_line_rating():
    with pytest.raises(ValueError):
        evaluate_line_decision(
            line_rating_q05_a=-100,
            static_rating_a=700,
            monitoring_status="VALID"
        )


def test_missing_line_rating_with_valid_status():
    with pytest.raises(ValueError):
        evaluate_line_decision(
            line_rating_q05_a=None,
            static_rating_a=700,
            monitoring_status="VALID"
        )
