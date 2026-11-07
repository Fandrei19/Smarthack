import json
import pandas as pd

# lists defining the mandatory columns required in both datasets to ensure data integrity before further processing
WEATHER_COLS = ["timestamp_utc", "segment_id", "air_temp_c", "wind_speed_ms",
                "wind_direction_deg", "solar_radiation", "source", "data_quality"]
PRED_COLS = ["timestamp_utc", "segment_id", "q05_ampacity_a", "q50_ampacity_a",
             "q95_ampacity_a", "static_rating_a", "uncertainty_ratio", "data_quality"]


# validates that all required columns exist in a given data frame
def check(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")

# standardizes and cleans the raw data
def clean(df):
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce", utc=True)
    
    df["segment_id"] = (
        df["segment_id"].astype(str).str.strip().str.upper()
    )

    # removes any rows where the timestamp value is missing + duplicate entries where both the timestamp and segment_id are identical
    return df.dropna(subset=["timestamp"]).drop_duplicates(["timestamp", "segment_id"])


# The live snapshots are small and can change between dashboard runs.
def load_weather(path="data/weather.csv"):
    df = pd.read_csv(path)
    check(df, WEATHER_COLS, "weather.csv")

    df = df.drop(columns=["source"]).rename(columns={
        "timestamp_utc": "timestamp",
        "air_temp_c": "temperature",
        "wind_speed_ms": "wind_speed",
        "wind_direction_deg": "wind_direction",
        "data_quality": "weather_quality",
    })
    return clean(df)


def load_predictions(path="data/predictions.csv"):
    df = pd.read_csv(path)
    check(df, PRED_COLS, "predictions.csv")

    df = df.rename(columns={
        "timestamp_utc": "timestamp",
        "q05_ampacity_a": "q05",
        "q50_ampacity_a": "q50",
        "q95_ampacity_a": "q95",
        "static_rating_a": "static_rating",
        "uncertainty_ratio": "uncertainty",
        "data_quality": "prediction_quality",
    })
    df["predicted_ampacity"] = df["q05"]
    return clean(df)

# opens and parses agent results, returning them as a dict with utf-8 encoding
def load_agent(path="data/agent_results.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
