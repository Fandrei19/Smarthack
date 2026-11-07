import json
from pathlib import Path
from typing import Any
import pandas as pd

from src.decision_engine import evaluate_line_decision
from src.multi_point_monitor import (
    calculate_line_rating,
    validate_predictions,
)


EXPECTED_SEGMENTS = ["SEG_1", "SEG_2", "SEG_3"]
DECISION_LOG: list[dict[str, Any]] = []


def validate_agent_input(predictions: pd.DataFrame) -> dict:
    """
    Validate prediction data structure and values.
    """
    return validate_predictions(predictions)


def monitor_agent_predictions(
    predictions: pd.DataFrame,
    expected_segments: list[str] | None = None
) -> dict:
    """
    Validate predictions and perform multi-point line monitoring.
    """
    segments = expected_segments or EXPECTED_SEGMENTS
    validation_result = validate_agent_input(predictions)

    if not isinstance(predictions, pd.DataFrame) or predictions.empty:
        return {
            "validation": validation_result,
            "monitoring": None
        }

    required_cols = {"timestamp_utc", "segment_id", "q05_ampacity_a"}
    if not required_cols.issubset(predictions.columns):
        return {
            "validation": validation_result,
            "monitoring": None
        }

    try:
        monitoring_result = calculate_line_rating(
            predictions=predictions,
            expected_segments=segments
        )
    except ValueError:
        return {
            "validation": validation_result,
            "monitoring": None
        }

    return {
        "validation": validation_result,
        "monitoring": monitoring_result
    }


def calculate_line_static_ratings(
    predictions: pd.DataFrame,
    monitoring_results: pd.DataFrame,
    expected_segments: list[str] | None = None
) -> pd.DataFrame:
    """
    Calculate the conservative line-level static rating for each valid timestamp.
    """
    segments = expected_segments or EXPECTED_SEGMENTS
    required_columns = [
        "timestamp_utc",
        "segment_id",
        "static_rating_a"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in predictions.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing static rating columns: {missing_columns}")

    if not {"timestamp_utc", "monitoring_status"}.issubset(monitoring_results.columns):
        raise ValueError("Monitoring results have an invalid structure.")

    valid_results = monitoring_results[
        monitoring_results["monitoring_status"] == "VALID"
    ].copy()

    output = []
    timestamp_predictions = predictions.copy()
    timestamp_predictions["timestamp_utc"] = pd.to_datetime(
        timestamp_predictions["timestamp_utc"],
        utc=True,
        errors="coerce"
    )

    for timestamp in valid_results["timestamp_utc"]:
        timestamp = pd.to_datetime(timestamp, utc=True)
        group = timestamp_predictions[
            timestamp_predictions["timestamp_utc"] == timestamp
        ]
        group = group[group["segment_id"].astype(str).isin(segments)]

        segment_ids = set(group["segment_id"].astype(str))
        if segment_ids != set(segments):
            raise ValueError(f"Expected exactly {segments} for {timestamp}.")

        ratings = pd.to_numeric(group["static_rating_a"], errors="coerce")
        if ratings.isna().any() or not (ratings > 0).all():
            raise ValueError(f"Invalid static rating for timestamp {timestamp}.")

        output.append({
            "timestamp_utc": timestamp,
            "static_rating_a": float(ratings.min())
        })

    return pd.DataFrame(output, columns=["timestamp_utc", "static_rating_a"])


def _format_utc_timestamp(ts: Any) -> str | None:
    if ts is None or pd.isna(ts):
        return None
    parsed = pd.to_datetime(ts, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")


def generate_agent_decisions(
    monitoring_results: pd.DataFrame,
    static_ratings: pd.DataFrame | None = None,
    planned_load_a: float | dict[str, float] | None = None,
    is_stale: bool = False
) -> pd.DataFrame:
    """
    Generate a risk-based decision for every monitored timestamp.
    """
    decisions = []

    for _, monitoring_row in monitoring_results.iterrows():
        timestamp = monitoring_row["timestamp_utc"]
        ts_iso = _format_utc_timestamp(timestamp)
        monitoring_status = monitoring_row["monitoring_status"]
        monitoring_reason = monitoring_row.get("reason")

        row_load = None
        if isinstance(planned_load_a, dict):
            row_load = planned_load_a.get(ts_iso) or planned_load_a.get(str(timestamp))
        elif planned_load_a is not None:
            row_load = planned_load_a

        if monitoring_status == "INCOMPLETE":
            decision = evaluate_line_decision(
                line_rating_q05_a=None,
                static_rating_a=None,
                monitoring_status="INCOMPLETE",
                monitoring_reason=monitoring_reason,
                is_stale=is_stale
            )
        elif monitoring_status == "VALID":
            static_val = monitoring_row.get("static_rating_a")
            if (static_val is None or pd.isna(static_val)) and static_ratings is not None:
                matching_rating = static_ratings[
                    static_ratings["timestamp_utc"] == timestamp
                ]
                if len(matching_rating) != 1:
                    raise ValueError(f"Expected exactly one static rating for {timestamp}.")
                static_val = float(matching_rating.iloc[0]["static_rating_a"])

            uncertainty_val = monitoring_row.get("uncertainty_ratio")
            if pd.isna(uncertainty_val) and "uncertainty_ratio" not in monitoring_row:
                uncertainty_val = None

            decision = evaluate_line_decision(
                line_rating_q05_a=float(monitoring_row["line_rating_q05_a"]),
                static_rating_a=float(static_val) if static_val is not None else None,
                monitoring_status="VALID",
                uncertainty_ratio=uncertainty_val,
                planned_load_a=row_load,
                monitoring_reason=monitoring_reason,
                is_stale=is_stale
            )
        else:
            raise ValueError(f"Unexpected monitoring status: {monitoring_status}")

        record = {
            "timestamp_utc": timestamp,
            "line_rating_q05_a": (
                float(monitoring_row["line_rating_q05_a"])
                if pd.notna(monitoring_row["line_rating_q05_a"])
                else None
            ),
            "bottleneck_segment": (
                str(monitoring_row["bottleneck_segment"])
                if pd.notna(monitoring_row["bottleneck_segment"])
                else None
            ),
            "risk_level": decision["risk_level"],
            "action": decision["action"],
            "requires_human_review": bool(decision["requires_human_review"]),
            "reason": decision["reason"],
            "reason_code": decision["reason_code"],
            "bottleneck_changed": bool(monitoring_row.get("bottleneck_changed", False))
        }
        decisions.append(record)

    return pd.DataFrame(decisions)


def run_agent(
    predictions: pd.DataFrame | str | Path = "data/processed/predictions.csv",
    config: dict[str, Any] | str | Path | None = "config/line_segments.json"
) -> list[dict[str, Any]]:
    cfg: dict[str, Any] = {}
    if isinstance(config, (str, Path)):
        cfg_path = Path(config)
        if cfg_path.exists():
            loaded = json.loads(cfg_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                cfg = loaded
    elif isinstance(config, dict):
        cfg = config

    expected_segments = load_expected_segments_from_config(config)
    planned_load_a = cfg.get("planned_load_a", None)
    is_stale = bool(cfg.get("is_stale", False))
    output_path = cfg.get("output_path", "data/processed/agent_results.json")
    log_path = cfg.get("log_path", None)

    if isinstance(predictions, (str, Path)):
        pred_path = Path(predictions)
        if not pred_path.exists():
            raise FileNotFoundError(f"Predictions file not found: {pred_path}")
        df = pd.read_csv(pred_path)
    elif isinstance(predictions, pd.DataFrame):
        df = predictions.copy()
    else:
        raise ValueError("predictions must be a DataFrame or file path.")

    monitoring_df = calculate_line_rating(
        predictions=df,
        expected_segments=expected_segments
    )

    decisions_df = generate_agent_decisions(
        monitoring_results=monitoring_df,
        static_ratings=None,
        planned_load_a=planned_load_a,
        is_stale=is_stale
    )

    agent_results: list[dict[str, Any]] = []
    DECISION_LOG.clear()

    for _, row in decisions_df.iterrows():
        ts_str = _format_utc_timestamp(row["timestamp_utc"])
        contract_item = {
            "timestamp_utc": ts_str,
            "line_rating_q05_a": (
                float(row["line_rating_q05_a"])
                if pd.notna(row["line_rating_q05_a"])
                else None
            ),
            "bottleneck_segment": (
                str(row["bottleneck_segment"])
                if pd.notna(row["bottleneck_segment"])
                else None
            ),
            "risk_level": str(row["risk_level"]),
            "action": str(row["action"]),
            "requires_human_review": bool(row["requires_human_review"]),
            "reason": str(row["reason"])
        }
        agent_results.append(contract_item)

        DECISION_LOG.append({
            **contract_item,
            "reason_code": str(row["reason_code"]),
            "bottleneck_changed": bool(row["bottleneck_changed"])
        })

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(agent_results, f, indent=2)

    if log_path:
        log_file = Path(log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as f:
            json.dump(DECISION_LOG, f, indent=2)

    return agent_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run GridPilot AI Decision Agent")
    parser.add_argument(
        "--predictions",
        default="data/processed/predictions.csv",
        help="Path to predictions.csv"
    )
    parser.add_argument(
        "--config",
        default="config/line_segments.json",
        help="Path to line_segments.json"
    )
    parser.add_argument(
        "--output",
        default="data/processed/agent_results.json",
        help="Output path for agent_results.json"
    )
    args = parser.parse_args()

    cfg_dict = {"output_path": args.output}
    cfg_dict["expected_segments"] = load_expected_segments_from_config(args.config)

    results = run_agent(predictions=args.predictions, config=cfg_dict)
    summary = summarize_agent_results(results)
    print(f"Processed {summary['total_hours']} hours -> {args.output}")
    print(f"Actions: {summary['action_counts']}")

def load_expected_segments_from_config(config: dict[str, Any] | str | Path | None) -> list[str]:
    if config is None:
        return EXPECTED_SEGMENTS

    if isinstance(config, (str, Path)):
        cfg_path = Path(config)
        if not cfg_path.exists():
            return EXPECTED_SEGMENTS
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    elif isinstance(config, dict):
        raw = config
    else:
        return EXPECTED_SEGMENTS

    if isinstance(raw, list):
        return [
            str(item["segment_id"] if isinstance(item, dict) else item)
            for item in raw
        ]

    segments = raw.get("expected_segments") or raw.get("segments") or raw.get("line_segments")
    if not segments:
        return EXPECTED_SEGMENTS

    return [
        str(item["segment_id"] if isinstance(item, dict) else item)
        for item in segments
    ]


def summarize_agent_results(agent_results: list[dict[str, Any]]) -> dict[str, Any]:
    total_hours = len(agent_results)
    if total_hours == 0:
        return {
            "total_hours": 0,
            "action_counts": {},
            "risk_counts": {},
            "human_review_hours": 0,
            "bottleneck_counts": {},
            "min_line_rating_q05_a": None,
            "max_line_rating_q05_a": None,
        }

    df = pd.DataFrame(agent_results)
    valid_ratings = df["line_rating_q05_a"].dropna()

    return {
        "total_hours": total_hours,
        "action_counts": df["action"].value_counts().to_dict(),
        "risk_counts": df["risk_level"].value_counts().to_dict(),
        "human_review_hours": int(df["requires_human_review"].sum()),
        "bottleneck_counts": df["bottleneck_segment"].dropna().value_counts().to_dict(),
        "min_line_rating_q05_a": float(valid_ratings.min()) if not valid_ratings.empty else None,
        "max_line_rating_q05_a": float(valid_ratings.max()) if not valid_ratings.empty else None,
    }