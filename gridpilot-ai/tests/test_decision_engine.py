
from src.decision_engine import calculate_utilization


def test_calculate_utilization():
    utilization = calculate_utilization(
        line_rating_q05_a=800,
        static_rating_a=700
    )

    assert round(utilization, 3) == 0.875
