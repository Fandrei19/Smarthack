"""Build an hourly dataset for one-hour-ahead DLR weather prediction.

Source: Open-Meteo Historical Weather API, ERA5 reanalysis (CC BY 4.0).
https://open-meteo.com/en/docs/historical-weather-api
Demo locations represent synthetic spans, not monitored power lines.
Requires pandas. Run: python data_pipeline.py
"""

import argparse
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd


ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
DEMO_POINTS = {
    "S1": (44.38, 28.14),
    "S2": (44.46, 28.48),
    "S3": (44.55, 28.81),
}
FIELDS = {
    "temperature_2m": ("air_temp_c", "°C"),
    "wind_speed_10m": ("wind_speed_ms", "m/s"),
    "wind_direction_10m": ("wind_from_deg", "°"),
    "shortwave_radiation": ("solar_ghi_wm2", "W/m²"),
}
LAGS_H = (1, 3, 6)


def get_json(url):
    for attempt in range(3):
        try:
            with urlopen(url, timeout=35) as response:
                return json.load(response)
        except HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise
        except (URLError, TimeoutError):
            if attempt == 2:
                raise
        time.sleep(2 ** attempt)


def download_weather(latitude, longitude, start, end):
    query = urlencode({
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(FIELDS),
        "models": "era5",
        "wind_speed_unit": "ms",
        "timezone": "GMT",
    })
    payload = get_json(f"{ARCHIVE}?{query}")
    for field, (_, unit) in FIELDS.items():
        actual = payload["hourly_units"].get(field)
        if actual != unit:
            raise ValueError(f"Unexpected unit for {field}: {actual!r}; expected {unit!r}")

    hourly = payload["hourly"]
    index = pd.to_datetime(hourly["time"], utc=True)
    if index.has_duplicates:
        raise ValueError("Open-Meteo response contains duplicate timestamps")
    if not (index == index.floor("h")).all():
        raise ValueError("Open-Meteo response contains non-hourly timestamps")
    weather = pd.DataFrame({
        name: pd.to_numeric(hourly[field], errors="coerce")
        for field, (name, _) in FIELDS.items()
    }, index=index)
    grid = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(hours=23),
                         freq="h", tz="UTC", name="timestamp_utc")
    weather = weather.reindex(grid).replace([np.inf, -np.inf], np.nan)
    weather["wind_speed_ms"] = weather["wind_speed_ms"].where(weather["wind_speed_ms"] >= 0)
    weather["wind_from_deg"] = weather["wind_from_deg"].where(
        weather["wind_from_deg"].between(0, 360)
    ) % 360
    weather["solar_ghi_wm2"] = weather["solar_ghi_wm2"].where(weather["solar_ghi_wm2"] >= 0)
    return weather, payload["latitude"], payload["longitude"]


def training_rows(weather, span_id, location, source_location, train_end, validation_end):
    # Original clock stays intact during shift, even if observations are missing.
    features = weather.copy()
    radians = np.deg2rad(weather["wind_from_deg"])
    features["wind_dir_sin"] = np.sin(radians)
    features["wind_dir_cos"] = np.cos(radians)
    lag_columns = ["air_temp_c", "wind_speed_ms", "wind_dir_sin",
                   "wind_dir_cos", "solar_ghi_wm2"]
    for lag in LAGS_H:
        lagged = features[lag_columns].shift(lag).add_suffix(f"_lag_{lag}h")
        features = features.join(lagged)
    features["hour_utc"] = features.index.hour
    features["month_utc"] = features.index.month

    targets = weather.shift(-1).add_prefix("target_").add_suffix("_1h")
    rows = features.join(targets)
    rows.insert(0, "target_timestamp_utc", rows.index + pd.Timedelta(hours=1))
    rows.insert(0, "span_id", span_id)
    rows["latitude"], rows["longitude"] = location
    rows["era5_latitude"], rows["era5_longitude"] = source_location
    # Drop the row directly before each split boundary so its target cannot cross it.
    for cutoff in (train_end, validation_end):
        rows = rows.loc[~((rows.index < cutoff) & (rows["target_timestamp_utc"] >= cutoff))]
    rows["split"] = np.select(
        [rows.index < train_end, rows.index < validation_end],
        ["train", "validation"], default="test",
    )
    return rows.dropna().reset_index()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2022-01-01", help="First included day (UTC)")
    parser.add_argument("--end", default="2025-12-31", help="Last included day (UTC)")
    parser.add_argument("--train-end", default="2024-01-01", help="First validation day (UTC)")
    parser.add_argument("--validation-end", default="2025-01-01", help="First test day (UTC)")
    parser.add_argument("--point", nargs=3, action="append", metavar=("ID", "LAT", "LON"),
                        help="Repeat to replace the three demo locations")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("dataset.csv"))
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        start, train_end, validation_end, end = (
            pd.Timestamp(value, tz="UTC")
            for value in (args.start, args.train_end, args.validation_end, args.end)
        )
        if any(value != value.normalize() for value in (start, train_end, validation_end, end)):
            raise ValueError("Use dates in YYYY-MM-DD format")
        if not start < train_end < validation_end <= end:
            raise ValueError("Expected start < train-end < validation-end <= end")
        points = DEMO_POINTS if not args.point else {
            span_id: (float(latitude), float(longitude))
            for span_id, latitude, longitude in args.point
        }
        if args.point and len(points) != len(args.point):
            raise ValueError("Point IDs must be unique")
        if any(not (-90 <= lat <= 90 and -180 <= lon <= 180)
               for lat, lon in points.values()):
            raise ValueError("Coordinates outside valid latitude/longitude ranges")

        batches = []
        for span_id, location in points.items():
            weather, era5_lat, era5_lon = download_weather(*location, args.start, args.end)
            rows = training_rows(weather, span_id, location, (era5_lat, era5_lon),
                                 train_end, validation_end)
            if set(rows["split"]) != {"train", "validation", "test"}:
                raise ValueError(f"{span_id}: at least one split has no usable samples")
            print(f"{span_id}: {len(rows):,} rows kept of {len(weather):,} source hours "
                  f"(ERA5 grid: {era5_lat}, {era5_lon})", flush=True)
            batches.append(rows)

        dataset = pd.concat(batches, ignore_index=True).sort_values(["timestamp_utc", "span_id"])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        dataset.to_csv(temporary, index=False, date_format="%Y-%m-%dT%H:%M:%SZ",
                       float_format="%.6g")
        temporary.replace(args.output)
        print(dataset.groupby(["span_id", "split"]).size().to_string())
        print(f"Saved {len(dataset):,} rows to {args.output}")
    except (ValueError, KeyError, URLError, TimeoutError) as exc:
        raise SystemExit(f"Dataset build failed: {exc}") from exc


if __name__ == "__main__":
    main()
