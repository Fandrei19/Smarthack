import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "timestamp_utc",
    "segment_id",
    "q05_ampacity_a",
]

FULL_VALIDATION_COLUMNS = [
    "timestamp_utc",
    "segment_id",
    "q05_ampacity_a",
    "q50_ampacity_a",
    "q95_ampacity_a",
]


def validate_predictions(predictions: pd.DataFrame) -> dict:
    """
    Validate the structure and row-level integrity of ML predictions.

    Args:
        predictions: DataFrame containing ML prediction results.

    Returns:
        Dictionary with 'is_valid' (bool) and 'errors' (list[str]).
    """
    errors = []

    if not isinstance(predictions, pd.DataFrame):
        return {
            "is_valid": False,
            "errors": ["Input must be a pandas DataFrame."]
        }

    if predictions.empty:
        return {
            "is_valid": False,
            "errors": ["Prediction DataFrame is empty."]
        }

    missing_columns = [
        column for column in FULL_VALIDATION_COLUMNS
        if column not in predictions.columns
    ]

    if missing_columns:
        return {
            "is_valid": False,
            "errors": [f"Missing required columns: {missing_columns}"]
        }

    if predictions["segment_id"].isna().any():
        errors.append("Missing segment identifiers.")
    elif predictions["segment_id"].astype("string").str.strip().eq("").any():
        errors.append("Empty segment identifiers.")

    timestamps = pd.to_datetime(
        predictions["timestamp_utc"],
        utc=True,
        errors="coerce"
    )

    if timestamps.isna().any():
        errors.append("Invalid or missing timestamps.")
    else:
        duplicate_keys = pd.DataFrame({
            "timestamp_utc": timestamps,
            "segment_id": predictions["segment_id"].astype("string").str.strip()
        })
        if duplicate_keys.duplicated().any():
            errors.append(
                "Duplicate predictions found for the same timestamp and segment."
            )

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


def calculate_line_rating(
    predictions: pd.DataFrame,
    expected_segments: list[str],
) -> pd.DataFrame:
    """
    Monitor multiple segments of a simulated power line per timestamp.

    For every timestamp:
    - Checks that all expected segments are present without duplicates.
    - Validates data quality, positive finite ampacity, and quantile ordering.
    - Calculates conservative line rating as min(q05_ampacity_a) if valid.
    - Identifies bottleneck_segment and tracks when the bottleneck changes.
    - Calculates segment-level margins relative to static_rating_a.
    - Aggregates conservative static_rating_a and worst-case uncertainty_ratio.

    Args:
        predictions: DataFrame from predictions.csv.
        expected_segments: List of required segment IDs (e.g., ["SEG_1", "SEG_2", "SEG_3"]).

    Returns:
        DataFrame with one monitoring record per timestamp.
    """
    if not isinstance(predictions, pd.DataFrame):
        raise ValueError("Predictions must be a pandas DataFrame.")

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in predictions.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    if not expected_segments:
        raise ValueError("No expected segments provided.")

    if predictions.empty:
        raise ValueError("Predictions DataFrame is empty.")

    df = predictions.copy()

    if df["segment_id"].isna().any():
        raise ValueError("Missing segment_id values.")

    df["segment_id"] = df["segment_id"].astype("string").str.strip()
    if df["segment_id"].eq("").any():
        raise ValueError("Empty segment_id values.")

    df["q05_ampacity_a"] = pd.to_numeric(df["q05_ampacity_a"], errors="coerce")

    has_quantiles = {"q50_ampacity_a", "q95_ampacity_a"}.issubset(df.columns)
    if has_quantiles:
        df["q50_ampacity_a"] = pd.to_numeric(df["q50_ampacity_a"], errors="coerce")
        df["q95_ampacity_a"] = pd.to_numeric(df["q95_ampacity_a"], errors="coerce")

    has_static = "static_rating_a" in df.columns
    if has_static:
        df["static_rating_a"] = pd.to_numeric(df["static_rating_a"], errors="coerce")

    has_uncertainty = "uncertainty_ratio" in df.columns
    if has_uncertainty:
        df["uncertainty_ratio"] = pd.to_numeric(df["uncertainty_ratio"], errors="coerce")

    has_quality = "data_quality" in df.columns

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"],
        utc=True,
        errors="coerce"
    )

    results = []
    previous_bottleneck: str | None = None

    for timestamp, group in df.groupby("timestamp_utc", dropna=False, sort=True):
        if pd.isna(timestamp):
            results.append({
                "timestamp_utc": None,
                "line_rating_q05_a": None,
                "static_rating_a": None,
                "uncertainty_ratio": None,
                "bottleneck_segment": None,
                "bottleneck_changed": False,
                "segment_margins_a": {},
                "min_segment_margin_a": None,
                "monitoring_status": "INCOMPLETE",
                "missing_segments": list(expected_segments),
                "reason": "Invalid timestamp."
            })
            continue

        group = group[group["segment_id"].isin(expected_segments)]

        present_segments = set(group["segment_id"])
        missing = sorted(set(expected_segments) - present_segments)
        duplicates = bool(group["segment_id"].duplicated().any())

        q05_vals = group["q05_ampacity_a"]
        invalid_ampacity = bool(
            q05_vals.isna().any()
            or not np.isfinite(q05_vals.to_numpy()).all()
            or (q05_vals <= 0).any()
        )

        invalid_quantiles = False
        if has_quantiles and not invalid_ampacity:
            q50_vals = group["q50_ampacity_a"]
            q95_vals = group["q95_ampacity_a"]
            if (
                q50_vals.isna().any()
                or q95_vals.isna().any()
                or not np.isfinite(q50_vals.to_numpy()).all()
                or not np.isfinite(q95_vals.to_numpy()).all()
                or (q50_vals <= 0).any()
                or (q95_vals <= 0).any()
                or (q05_vals > q50_vals).any()
                or (q50_vals > q95_vals).any()
            ):
                invalid_quantiles = True

        bad_quality = False
        if has_quality:
            q_flags = group["data_quality"].astype("string").str.strip().str.lower()
            if q_flags.isna().any() or not q_flags.eq("ok").all():
                bad_quality = True

        invalid_static = False
        static_fallback = None
        if has_static:
            s_vals = group["static_rating_a"]
            valid_s = s_vals.dropna()
            valid_s = valid_s[np.isfinite(valid_s) & (valid_s > 0)]
            if not valid_s.empty:
                static_fallback = float(valid_s.min())
            if s_vals.isna().any() or not np.isfinite(s_vals.to_numpy()).all() or (s_vals <= 0).any():
                invalid_static = True

        if (
            missing
            or duplicates
            or invalid_ampacity
            or invalid_quantiles
            or bad_quality
            or invalid_static
        ):
            reasons = []
            if missing:
                reasons.append(f"Missing segments: {missing}.")
            if duplicates:
                reasons.append("Duplicate segment predictions.")
            if bad_quality:
                reasons.append("Invalid weather or data quality flag.")
            if invalid_ampacity:
                reasons.append("Invalid ampacity values.")
            if invalid_quantiles:
                reasons.append("Prediction intervals are inconsistent.")
            if invalid_static:
                reasons.append("Invalid static rating values.")

            results.append({
                "timestamp_utc": timestamp,
                "line_rating_q05_a": None,
                "static_rating_a": static_fallback,
                "uncertainty_ratio": None,
                "bottleneck_segment": None,
                "bottleneck_changed": False,
                "segment_margins_a": {},
                "min_segment_margin_a": None,
                "monitoring_status": "INCOMPLETE",
                "missing_segments": missing,
                "reason": " ".join(reasons)
            })
            continue

        bottleneck_idx = group["q05_ampacity_a"].idxmin()
        bottleneck = group.loc[bottleneck_idx]
        bottleneck_seg = str(bottleneck["segment_id"])
        line_rating = float(bottleneck["q05_ampacity_a"])

        bottleneck_changed = (
            previous_bottleneck is not None
            and bottleneck_seg != previous_bottleneck
        )
        previous_bottleneck = bottleneck_seg

        line_static = float(group["static_rating_a"].min()) if has_static else None

        segment_margins = {}
        min_margin = None
        if has_static:
            margins_series = group["q05_ampacity_a"] - group["static_rating_a"]
            segment_margins = {
                str(row["segment_id"]): float(row["q05_ampacity_a"] - row["static_rating_a"])
                for _, row in group.iterrows()
            }
            min_margin = float(margins_series.min())

        worst_uncertainty = None
        if has_uncertainty:
            u_vals = group["uncertainty_ratio"]
            if u_vals.isna().any() or not np.isfinite(u_vals.to_numpy()).all() or (u_vals < 0).any():
                worst_uncertainty = float("nan")
            else:
                worst_uncertainty = float(u_vals.max())

        results.append({
            "timestamp_utc": timestamp,
            "line_rating_q05_a": line_rating,
            "static_rating_a": line_static,
            "uncertainty_ratio": worst_uncertainty,
            "bottleneck_segment": bottleneck_seg,
            "bottleneck_changed": bottleneck_changed,
            "segment_margins_a": segment_margins,
            "min_segment_margin_a": min_margin,
            "monitoring_status": "VALID",
            "missing_segments": [],
            "reason": f"{bottleneck_seg} has the lowest conservative ampacity."
        })

    return pd.DataFrame(results)


def monitor_line(
    predictions: pd.DataFrame,
    expected_segments: list[str],
) -> pd.DataFrame:
    """Alias for calculate_line_rating to preserve backward compatibility."""
    return calculate_line_rating(
        predictions=predictions,
        expected_segments=expected_segments
    )