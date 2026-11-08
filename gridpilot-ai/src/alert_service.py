"""Derive operator notifications without changing the decision engine."""
import hashlib
import json
import pandas as pd
from src.live_pipeline import OUTPUT_DIR


def build_alerts(decisions, weather, predictions, now, updated_at):
    now = pd.Timestamp(now)
    cutoff = now.floor("h")
    alerts = []

    def add(kind, hour, segment, reason, severity="Review", action="Verify input data"):
        key = json.dumps([kind, str(hour), segment, reason], ensure_ascii=True)
        alerts.append(dict(id=hashlib.sha256(key.encode()).hexdigest()[:16],
            severity=severity, category=kind, timestamp_utc=str(hour),
            segment=segment, reason=reason, action=action))

    if (now - pd.Timestamp(updated_at)).total_seconds() > 1800:
        add("Stale data", updated_at, "Whole line", "The last complete update is over 30 minutes old.",
            action="Refresh the forecast and verify data freshness before using recommendations")
    times = []
    for row in decisions:
        hour = pd.to_datetime(row.get("timestamp_utc"), utc=True, errors="coerce")
        if pd.isna(hour):
            add("Invalid decision", "Unknown", "Whole line", "An agent decision has an invalid timestamp.")
            continue
        if hour < cutoff:
            continue
        times.append(hour)
        if row.get("requires_human_review") or row.get("action") == "OPERATOR_REVIEW" or row.get("risk_level") == "HIGH":
            add("Agent review", hour, row.get("bottleneck_segment") or "Whole line",
                row.get("reason") or "The agent requests human review.",
                "Critical" if row.get("risk_level") == "HIGH" else "Review",
                row.get("action", "OPERATOR_REVIEW"))
    if not times:
        add("Expired forecast", "Now", "Whole line", "No current or future agent decisions are available.")
    expected = {"SEG_1", "SEG_2", "SEG_3"}
    for frame, name, values in [(weather, "Weather quality", ["air_temp_c", "wind_speed_ms", "wind_direction_deg", "solar_radiation"]),
                                (predictions, "Prediction quality", ["q05_ampacity_a", "q50_ampacity_a", "q95_ampacity_a"])]:
        required = {"timestamp_utc", "segment_id", "data_quality", *values}
        if not required.issubset(frame.columns):
            add(name, "Now", "Whole line", "Required data columns are missing.")
            continue
        frame = frame.copy()
        frame["timestamp_utc"] = pd.to_datetime(frame.timestamp_utc, utc=True, errors="coerce")
        if frame.timestamp_utc.isna().any():
            add(name, "Unknown", "Whole line", "Some records have invalid timestamps.")
        frame = frame[frame.timestamp_utc >= cutoff]
        for hour in sorted(set(times) | set(frame.timestamp_utc)):
            group = frame[frame.timestamp_utc == hour]
            missing = expected - set(group.segment_id)
            if missing:
                add(name, hour, ", ".join(sorted(missing)), "Expected segment data is missing.")
            for _, row in group.iterrows():
                numeric = pd.to_numeric(row[values], errors="coerce")
                if str(row.data_quality).lower() != "ok" or numeric.isna().any() or numeric.isin([float("inf"), -float("inf")]).any():
                    add(name, hour, str(row.segment_id), f"Quality flag: {row.data_quality}; missing or invalid values must be checked.")
    return sorted(alerts, key=lambda a: (a["severity"] != "Critical", a["timestamp_utc"], a["segment"]))


def load_alerts(output_dir=OUTPUT_DIR):
    try:
        paths = [output_dir / name for name in ("agent_results_live.json", "weather_live.csv", "predictions_live.csv")]
        decisions = json.loads(paths[0].read_text(encoding="utf-8"))
        if not isinstance(decisions, list) or any(not isinstance(row, dict) for row in decisions):
            raise ValueError("Invalid agent decision structure")
        return build_alerts(decisions, pd.read_csv(paths[1]), pd.read_csv(paths[2]),
            pd.Timestamp.now(tz="UTC"), pd.Timestamp(min(p.stat().st_mtime for p in paths), unit="s", tz="UTC"))
    except (OSError, ValueError, KeyError, TypeError):
        return [dict(id="live-input-unavailable", severity="Review", category="Data unavailable",
            timestamp_utc="Now", segment="Whole line", reason="Live files are missing or unreadable.",
            action="Refresh live data; verify the pipeline before using recommendations")]
