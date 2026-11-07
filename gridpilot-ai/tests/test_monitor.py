
import pandas as pd
from src.multi_point_monitor import monitor_line

expected_segments = ["S1", "S2", "S3"]

test_cases = {
    "missing_value": [850, None, 910],
    "zero_value": [850, 0, 910],
    "negative_value": [850, -20, 910],
}

for test_name, values in test_cases.items():
    predictions = pd.DataFrame([
        {
            "timestamp_utc": "2026-11-07T12:00:00Z",
            "segment_id": segment,
            "q05_ampacity_a": value
        }
        for segment, value in zip(expected_segments, values)
    ])

    result = monitor_line(predictions, expected_segments)

    assert result.iloc[0]["monitoring_status"] == "INCOMPLETE", (
        f"{test_name}: expected INCOMPLETE"
    )
    assert pd.isna(result.iloc[0]["line_rating_q05_a"]), (
        f"{test_name}: expected no line rating"
    )

    print(f"{test_name}: PASSED")

print("\nAll invalid-ampacity tests passed!")
