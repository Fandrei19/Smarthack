"""Export hourly weather for GridPilot AI's ampacity model.

Source: Open-Meteo Historical Weather API, ERA5 reanalysis (CC BY 4.0).
https://open-meteo.com/en/docs/historical-weather-api
Demo locations represent synthetic spans, not monitored power lines.
Requires pandas. From gridpilot-ai, run: python -m src.data_pipeline
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
FORECAST = "https://api.open-meteo.com/v1/forecast"
ROOT = Path(__file__).resolve().parents[1]
ML_COLUMN_NAMES = {
    "wind_from_deg": "wind_direction_deg",
    "solar_ghi_wm2": "solar_radiation",
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


def read_points(path):
    with path.open(encoding="utf-8") as handle:
        segments = json.load(handle)["segments"]
    if not isinstance(segments, list) or not segments:
        raise ValueError("Config must contain a nonempty segments list")
    points = {}
    for segment in segments:
        segment_id = segment["segment_id"]
        lat, lon, azimuth = (float(segment[k]) for k in ("latitude", "longitude", "azimuth_deg"))
        if not isinstance(segment_id, str) or not segment_id.strip() or segment_id in points:
            raise ValueError("Segment IDs must be nonempty and unique")
        if not (-90 <= lat <= 90 and -180 <= lon <= 180 and 0 <= azimuth < 360):
            raise ValueError(f"{segment_id}: invalid geometry")
        points[segment_id] = (lat, lon)
    return points


def fetch_hourly(endpoint, latitude, longitude, grid, **options):
    query = urlencode({
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(FIELDS),
        "wind_speed_unit": "ms",
        "timezone": "GMT",
        **options,
    })
    payload = get_json(f"{endpoint}?{query}")
    for field, (_, unit) in FIELDS.items():
        actual = payload["hourly_units"].get(field)
        if actual != unit:
            raise ValueError(f"Unexpected unit for {field}: {actual!r}; expected {unit!r}")

    hourly = payload["hourly"]
    index = pd.to_datetime(hourly["time"], utc=True)
    if index.intersection(grid).empty:
        raise ValueError("Open-Meteo returned no timestamps in the requested window")
    if index.has_duplicates:
        raise ValueError("Open-Meteo response contains duplicate timestamps")
    if not (index == index.floor("h")).all():
        raise ValueError("Open-Meteo response contains non-hourly timestamps")
    weather = pd.DataFrame({
        name: pd.to_numeric(hourly[field], errors="coerce")
        for field, (name, _) in FIELDS.items()
    }, index=index)
    weather = weather.reindex(grid).replace([np.inf, -np.inf], np.nan)
    weather.index.name = "timestamp_utc"
    weather["air_temp_c"] = weather["air_temp_c"].where(weather["air_temp_c"].between(-60, 60))
    weather["wind_speed_ms"] = weather["wind_speed_ms"].where(weather["wind_speed_ms"].between(0, 80))
    weather["wind_from_deg"] = weather["wind_from_deg"].where(
        weather["wind_from_deg"].between(0, 360)
    ) % 360
    weather["solar_ghi_wm2"] = weather["solar_ghi_wm2"].where(weather["solar_ghi_wm2"].between(0, 1500))
    return weather, payload["latitude"], payload["longitude"]


def model_weather_rows(weather, segment_id, source):
    rows = weather.rename(columns=ML_COLUMN_NAMES).reset_index()
    rows.insert(1, "segment_id", segment_id)
    rows["source"] = source
    rows["data_quality"] = np.where(weather.notna().all(axis=1), "ok", "missing")
    return rows


def save_rows(batches, path, id_column):
    rows = pd.concat(batches, ignore_index=True).sort_values(["timestamp_utc", id_column])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    rows.to_csv(temporary, index=False, date_format="%Y-%m-%dT%H:%M:%SZ", float_format="%.6g")
    temporary.replace(path)
    print(f"Saved {len(rows):,} rows to {path}")


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
    parser.add_argument("--segments", type=Path, default=ROOT / "config" / "line_segments.json")
    parser.add_argument("--start", default="2022-01-01", help="First included day (UTC)")
    parser.add_argument("--end", default="2025-12-31", help="Last included day (UTC)")
    parser.add_argument("--train-end", default="2024-01-01", help="First validation day (UTC)")
    parser.add_argument("--validation-end", default="2025-01-01", help="First test day (UTC)")
    parser.add_argument("--forecast-hours", type=int, help="Fetch live forecast for the next N full hours")
    parser.add_argument("--weather-output", type=Path, help="Override the ML weather CSV output path")
    parser.add_argument("--dataset-output", "--output", type=Path,
                        help="Optionally export the separate one-hour-ahead weather feature dataset")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        points = read_points(args.segments)
        if args.forecast_hours is not None:
            if not 1 <= args.forecast_hours <= 168 or args.dataset_output:
                raise ValueError("Use 1–168 forecast hours without --dataset-output")
            start = pd.Timestamp.now(tz="UTC").floor("h") + pd.Timedelta(hours=1)
            grid = pd.date_range(start, periods=args.forecast_hours, freq="h")
            options = dict(start_hour=grid[0].strftime("%Y-%m-%dT%H:%M"),
                           end_hour=grid[-1].strftime("%Y-%m-%dT%H:%M"))
            endpoint, source, filename = FORECAST, "OPEN_METEO_FORECAST", "weather_forecast.csv"
        else:
            start, end = (pd.Timestamp(d, tz="UTC") for d in (args.start, args.end))
            if start != start.normalize() or end != end.normalize() or start > end:
                raise ValueError("Use UTC dates in YYYY-MM-DD format with start <= end")
            grid = pd.date_range(start, end + pd.Timedelta(hours=23), freq="h")
            options = dict(start_date=args.start, end_date=args.end, models="era5")
            endpoint, source, filename = ARCHIVE, "ERA5", "weather.csv"
            if args.dataset_output:
                train_end, validation_end = (
                    pd.Timestamp(d, tz="UTC") for d in (args.train_end, args.validation_end)
                )
                if not start < train_end < validation_end <= end:
                    raise ValueError("Expected start < train-end < validation-end <= end")

        weather_output = args.weather_output or ROOT / "data" / "processed" / filename
        if args.dataset_output and args.dataset_output.resolve() == weather_output.resolve():
            raise ValueError("Weather and dataset output paths must differ")

        weather_batches, feature_batches = [], []
        for segment_id, location in points.items():
            weather, grid_lat, grid_lon = fetch_hourly(endpoint, *location, grid, **options)
            rows = model_weather_rows(weather, segment_id, source)
            weather_batches.append(rows)
            print(f"{segment_id}: {(rows['data_quality'] == 'ok').sum():,}/{len(rows):,} valid hours "
                  f"(source grid: {grid_lat}, {grid_lon})", flush=True)
            if args.dataset_output:
                features = training_rows(weather, segment_id, location, (grid_lat, grid_lon),
                                         train_end, validation_end)
                if set(features["split"]) != {"train", "validation", "test"}:
                    raise ValueError(f"{segment_id}: at least one split has no usable samples")
                feature_batches.append(features)

        save_rows(weather_batches, weather_output, "segment_id")
        if args.dataset_output:
            save_rows(feature_batches, args.dataset_output, "span_id")
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise SystemExit(f"Weather export failed: {exc}") from exc


if __name__ == "__main__":
    main()
