"""Read local weather history without mixing forecasts with observations."""
from pathlib import Path
import pandas as pd
from src.data_loader import WEATHER_COLS
from src.live_pipeline import ROOT


def history_files(root=ROOT):
    candidates = list((root / "data" / "processed").glob("weather*.csv"))
    candidates += list((root / "data" / "archive").glob("*/weather_live.csv"))
    return sorted(candidates)


def read_history(path):
    frame = pd.read_csv(path)
    missing = set(WEATHER_COLS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing weather columns: {sorted(missing)}")
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp_utc"])
    return frame.drop_duplicates(["timestamp_utc", "segment_id"]).sort_values(["timestamp_utc", "segment_id"])


def filter_history(frame, start, end, segments):
    start = pd.Timestamp(start, tz="UTC")
    stop = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    return frame[(frame.timestamp_utc >= start) & (frame.timestamp_utc < stop)
                 & frame.segment_id.isin(segments)]
