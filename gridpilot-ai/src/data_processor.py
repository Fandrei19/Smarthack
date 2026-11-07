import pandas as pd

# merges two datasets
def combine(weather, predictions):
    # left join: keep weather rows even if the prediction is missing (stays NaN, not 0)
    return weather.merge(predictions, on=["timestamp", "segment_id"], how="left")

# calculates new metrics based on existing rating values
def add_derived(df):
    df = df.copy()
    df["extra_A"] = df["predicted_ampacity"] - df["static_rating"]
    df["extra_pct"] = df["extra_A"] / df["static_rating"] * 100
    return df

# extracts the most recent data point for each segment
def latest_snapshot(df):
    """Latest available hour, one row per segment."""
    return df[df["timestamp"] == df["timestamp"].max()].set_index("segment_id")

# filters the dataset to a specific time window
def filter_window(df, hours=24):
    start = df["timestamp"].max() - pd.Timedelta(hours=hours)
    return df[df["timestamp"] >= start]


def line_rating_by_time(df):
    """per timestamp: line rating = min ampacity across segments, plus the limiting segment.
    If any segment is missing a prediction at that timestamp, the rating is NaN."""
    rows = []
    for ts, g in df.groupby("timestamp"):
        amp = g.set_index("segment_id")["predicted_ampacity"]
        known = not amp.isna().any()
        rows.append({
            "timestamp": ts,
            "line_rating": amp.min() if known else float("nan"),
            "limiting_segment": amp.idxmin() if known else None,
            "static_rating": g["static_rating"].min(),
        })
    return pd.DataFrame(rows)