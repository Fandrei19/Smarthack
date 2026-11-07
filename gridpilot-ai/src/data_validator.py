
import pandas as pd
import numpy as np

REQUIRED_COLUMNS = [
    "timestamp_utc",
    "segment_id",
    "q05_ampacity_a",
    "q50_ampacity_a",
    "q95_ampacity_a"
]


def validate_predictions(predictions: pd.DataFrame) -> dict:
    """
    Validate the structure and basic values of ML predictions.

    Checks whether:
        - The input is a non-empty DataFrame.
        - All required columns are present.
        - Segment identifiers are not missing.
        - Timestamps are valid.
        - Ampacity predictions are numeric, finite and positive.
        - Quantile predictions are ordered (Q05 <= Q50 <= Q95).

    Args:
        predictions: DataFrame containing ML prediction results.

    Returns:
        Dictionary containing:
            is_valid: Whether all validation checks passed.
            errors: List of validation errors.

    Note:
        This function validates prediction data only.
        It does not calculate ampacity or modify predictions.
    """

    errors = []

    # Check whether the input is a DataFrame.
    if not isinstance(predictions, pd.DataFrame):
        return {
            "is_valid": False,
            "errors": ["Input must be a pandas DataFrame."]
        }

    # Reject empty datasets.
    if predictions.empty:
        return {
            "is_valid": False,
            "errors": ["Prediction DataFrame is empty."]
        }

    # Check required columns.
    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in predictions.columns
    ]

    if missing_columns:
        return {
            "is_valid": False,
            "errors": [f"Missing required columns: {missing_columns}"]
        }

    # Check segment identifiers.
    if predictions["segment_id"].isna().any():
        errors.append("Missing segment identifiers.")

    if predictions["segment_id"].astype("string").str.strip().eq("").any():
        errors.append("Empty segment identifiers.")

    # Validate timestamps.
    timestamps = pd.to_datetime(
        predictions["timestamp_utc"],
        utc=True,
        errors="coerce"
    )

    if timestamps.isna().any():
        errors.append("Invalid or missing timestamps.")

    # Check for duplicate predictions for the same segment and timestamp.
    if not timestamps.isna().any():
        duplicate_keys = pd.DataFrame({
            "timestamp_utc": timestamps,
            "segment_id": predictions["segment_id"].astype("string").str.strip()
        })

        if duplicate_keys.duplicated().any():
            errors.append(
                "Duplicate predictions found for the same timestamp and segment."
            )

    # Validate data quality when the column is provided.
    if "data_quality" in predictions.columns:
        quality_values = (
            predictions["data_quality"]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        if quality_values.isna().any() or quality_values.eq("").any():
            errors.append("Missing or empty data quality values.")

        elif not quality_values.eq("ok").all():
            errors.append("Data quality contains values other than 'ok'.")

    # Check ampacity values.
    quantile_columns = [
        "q05_ampacity_a",
        "q50_ampacity_a",
        "q95_ampacity_a"
    ]

    numeric_values = predictions[quantile_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    if numeric_values.isna().any().any():
        errors.append("Ampacity predictions must be numeric and non-missing.")

    elif not np.isfinite(numeric_values.to_numpy()).all():
        errors.append("Ampacity predictions must be finite.")

    elif (numeric_values <= 0).any().any():
        errors.append("Ampacity predictions must be positive.")

    else:
        # Quantile predictions must follow Q05 <= Q50 <= Q95.
        invalid_order = (
            (numeric_values["q05_ampacity_a"] > numeric_values["q50_ampacity_a"]) |
            (numeric_values["q50_ampacity_a"] > numeric_values["q95_ampacity_a"])
        )

        if invalid_order.any():
            errors.append("Quantile predictions are not correctly ordered.")

    return {
        "is_valid": len(errors) == 0,
        "errors": errors
    }
