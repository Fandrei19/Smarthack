
import pandas as pd


REQUIRED_COLUMNS = [
    "timestamp_utc",
    "segment_id",
    "q05_ampacity_a",
]


def monitor_line(
    predictions: pd.DataFrame,
    expected_segments: list[str],
) -> pd.DataFrame:
    """
    Monitor multiple segments of a simulated power line.

    Identify the segment with the lowest Q05 ampacity and calculate
    the conservative rating for the entire line.

    If segments are missing or data is invalid, the result is marked
    INCOMPLETE.
    """

    # 1. Validate input
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in predictions.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    if not expected_segments:
        raise ValueError("No expected segments provided.")

    if predictions.empty:
        raise ValueError("Predictions DataFrame is empty.")

    df = predictions.copy()

    # 2. Validate segment identifiers
    df["segment_id"] = df["segment_id"].astype("string")

    if df["segment_id"].isna().any():
        raise ValueError("Missing segment_id values.")

    df["q05_ampacity_a"] = pd.to_numeric(
        df["q05_ampacity_a"],
        errors="coerce"
    )

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"],
        utc=True,
        errors="coerce"
    )

    results = []

    # 3. Process every timestamp independently
    for timestamp, group in df.groupby(
        "timestamp_utc",
        dropna=False
    ):

        if pd.isna(timestamp):
            results.append({
                "timestamp_utc": None,
                "line_rating_q05_a": None,
                "bottleneck_segment": None,
                "monitoring_status": "INCOMPLETE",
                "missing_segments": expected_segments.copy(),
                "reason": "Invalid timestamp."
            })
            continue

        # Keep only expected segments
        group = group[
            group["segment_id"].isin(expected_segments)
        ]

        present_segments = set(group["segment_id"])
        missing = sorted(
            set(expected_segments) - present_segments
        )

        # 4. Detect duplicates
        duplicates = group["segment_id"].duplicated().any()

        # 5. Detect invalid ampacity values
        invalid_values = (
            group["q05_ampacity_a"].isna().any()
            or (group["q05_ampacity_a"] <= 0).any()
        )

        # 6. Conservative fallback status
        if missing or duplicates or invalid_values:
            reasons = []

            if missing:
                reasons.append(f"Missing segments: {missing}")

            if duplicates:
                reasons.append("Duplicate segment predictions.")

            if invalid_values:
                reasons.append("Invalid ampacity values.")

            results.append({
                "timestamp_utc": timestamp,
                "line_rating_q05_a": None,
                "bottleneck_segment": None,
                "monitoring_status": "INCOMPLETE",
                "missing_segments": missing,
                "reason": " ".join(reasons)
            })

            continue

        # 7. Find bottleneck
        bottleneck_idx = group[
            "q05_ampacity_a"
        ].idxmin()

        bottleneck = group.loc[bottleneck_idx]

        results.append({
            "timestamp_utc": timestamp,
            "line_rating_q05_a": float(
                bottleneck["q05_ampacity_a"]
            ),
            "bottleneck_segment": bottleneck["segment_id"],
            "monitoring_status": "VALID",
            "missing_segments": [],
            "reason": (
                f"{bottleneck['segment_id']} "
                "has the lowest conservative ampacity."
            )
        })

    return pd.DataFrame(results)
