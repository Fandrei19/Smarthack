from pathlib import Path

import pandas as pd

from src.data_validator import validate_predictions


def test_real_predictions_csv_is_valid():
    csv_path = Path("data/processed/predictions.csv")

    predictions = pd.read_csv(csv_path)

    result = validate_predictions(predictions)

    assert result["is_valid"] is True, result["errors"]