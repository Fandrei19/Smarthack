"""Refresh the live weather, predictions and decisions on a schedule."""

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "processed"


def refresh_live_data():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Generate a complete run before replacing any of the published files.
    with TemporaryDirectory(prefix="gridpilot-live-", dir=OUTPUT_DIR) as staging_dir:
        staging = Path(staging_dir)
        weather = staging / "weather_live.csv"
        predictions = staging / "predictions_live.csv"
        decisions = staging / "agent_results_live.json"
        commands = (
            ["src.data_pipeline", "--forecast-hours", "24", "--weather-output", str(weather)],
            ["src.ml_model", "predict", "--weather", str(weather), "--out", str(predictions)],
            ["src.agent", "--predictions", str(predictions), "--output", str(decisions)],
        )
        for args in commands:
            subprocess.run([sys.executable, "-B", "-m", *args], cwd=ROOT, check=True, timeout=180)

        # Preserve each successful forecast run, including its retrieval time.
        fetched = datetime.now(timezone.utc)
        archive = ROOT / "data" / "archive" / fetched.strftime("%Y%m%dT%H%M%S%fZ")
        archive.mkdir(parents=True)
        for file in (weather, predictions, decisions):
            shutil.copy2(file, archive / file.name)
        (archive / "metadata.json").write_text(json.dumps({
            "retrieved_at_utc": fetched.isoformat(),
            "kind": "archived_forecast", "source": "OPEN_METEO_FORECAST"
        }), encoding="utf-8")
        for file in (weather, predictions, decisions):
            file.replace(OUTPUT_DIR / file.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval-minutes", type=int, default=15)
    parser.add_argument("--once", action="store_true", help="Refresh once and exit")
    args = parser.parse_args()
    if args.interval_minutes < 1:
        parser.error("--interval-minutes must be at least 1")

    try:
        while True:
            try:
                refresh_live_data()
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
                print(f"Live refresh failed: {exc}", file=sys.stderr, flush=True)
                if args.once:
                    return 1
            else:
                refreshed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
                print(f"Live data refreshed at {refreshed_at}", flush=True)
                if args.once:
                    return 0

            time.sleep(args.interval_minutes * 60)
    except KeyboardInterrupt:
        print("Live refresh stopped.", flush=True)
        return 130


if __name__ == "__main__":
    sys.exit(main())
