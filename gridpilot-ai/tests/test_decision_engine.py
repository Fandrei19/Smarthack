import json
import pandas as pd
import pytest

from src.agent import (
    DECISION_LOG,
    load_expected_segments_from_config,
    run_agent,
    summarize_agent_results,
)
from src.decision_engine import evaluate_line_decision
from src.multi_point_monitor import calculate_line_rating, validate_predictions


EXPECTED_SEGMENTS = ["SEG_1", "SEG_2", "SEG_3"]


def _make_base_predictions() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "timestamp_utc": "2022-01-01T00:00:00Z",
            "segment_id": "SEG_1",
            "q05_ampacity_a": 1413.5,
            "q50_ampacity_a": 1417.2,
            "q95_ampacity_a": 1422.6,
            "static_rating_a": 762.0,
            "uncertainty_ratio": 0.006,
            "data_quality": "ok",
        },
        {
            "timestamp_utc": "2022-01-01T00:00:00Z",
            "segment_id": "SEG_2",
            "q05_ampacity_a": 1479.4,
            "q50_ampacity_a": 1484.5,
            "q95_ampacity_a": 1485.0,
            "static_rating_a": 763.0,
            "uncertainty_ratio": 0.004,
            "data_quality": "ok",
        },
        {
            "timestamp_utc": "2022-01-01T00:00:00Z",
            "segment_id": "SEG_3",
            "q05_ampacity_a": 1443.8,
            "q50_ampacity_a": 1448.7,
            "q95_ampacity_a": 1462.5,
            "static_rating_a": 764.4,
            "uncertainty_ratio": 0.013,
            "data_quality": "ok",
        },
    ])


def test_all_segments_valid(tmp_path):
    df = _make_base_predictions()
    out_file = tmp_path / "agent_results.json"

    results = run_agent(df, config={"output_path": out_file})

    assert len(results) == 1
    res = results[0]
    assert res["timestamp_utc"] == "2022-01-01T00:00:00Z"
    assert res["line_rating_q05_a"] == pytest.approx(1413.5)
    assert res["bottleneck_segment"] == "SEG_1"
    assert res["risk_level"] == "LOW"
    assert res["action"] == "RECOMMEND_DYNAMIC_RATING"
    assert res["requires_human_review"] is False
    assert out_file.exists()

    saved = json.loads(out_file.read_text(encoding="utf-8"))
    assert list(saved[0].keys()) == [
        "timestamp_utc",
        "line_rating_q05_a",
        "bottleneck_segment",
        "risk_level",
        "action",
        "requires_human_review",
        "reason",
    ]


def test_one_segment_very_low_rating(tmp_path):
    df = _make_base_predictions()
    df.loc[df["segment_id"] == "SEG_2", "q05_ampacity_a"] = 780.0

    results = run_agent(df, config={"output_path": tmp_path / "out.json"})
    res = results[0]

    assert res["bottleneck_segment"] == "SEG_2"
    assert res["line_rating_q05_a"] == pytest.approx(780.0)
    assert res["risk_level"] == "HIGH"
    assert res["action"] == "OPERATOR_REVIEW"
    assert res["requires_human_review"] is True


def test_one_segment_missing(tmp_path):
    df = _make_base_predictions()
    df = df[df["segment_id"] != "SEG_3"].copy()

    results = run_agent(df, config={"output_path": tmp_path / "out.json"})
    res = results[0]

    assert res["line_rating_q05_a"] is None
    assert res["bottleneck_segment"] is None
    assert res["risk_level"] == "UNCERTAIN"
    assert res["action"] == "FALLBACK_STATIC"
    assert res["requires_human_review"] is True
    assert "Missing segments" in res["reason"]


def test_high_model_uncertainty(tmp_path):
    df = _make_base_predictions()
    df.loc[df["segment_id"] == "SEG_3", "uncertainty_ratio"] = 0.15

    results = run_agent(df, config={"output_path": tmp_path / "out.json"})
    res = results[0]

    assert res["risk_level"] == "UNCERTAIN"
    assert res["action"] == "FALLBACK_STATIC"
    assert res["requires_human_review"] is True
    assert DECISION_LOG[0]["reason_code"] == "UNCERTAINTY_HIGH"


def test_planned_load_exceeds_conservative_rating(tmp_path):
    df = _make_base_predictions()

    results = run_agent(
        df,
        config={
            "planned_load_a": 1450.0,
            "output_path": tmp_path / "out.json",
        },
    )
    res = results[0]

    assert res["risk_level"] == "HIGH"
    assert res["action"] == "OPERATOR_REVIEW"
    assert res["requires_human_review"] is True
    assert DECISION_LOG[0]["reason_code"] == "PLANNED_LOAD_EXCEEDS_DYNAMIC_RATING"


def test_static_rating_exceeds_dynamic_rating(tmp_path):
    df = _make_base_predictions()
    df.loc[df["segment_id"] == "SEG_1", "q05_ampacity_a"] = 700.0
    df.loc[df["segment_id"] == "SEG_1", "q50_ampacity_a"] = 710.0
    df.loc[df["segment_id"] == "SEG_1", "q95_ampacity_a"] = 720.0

    results = run_agent(df, config={"output_path": tmp_path / "out.json"})
    res = results[0]

    assert res["line_rating_q05_a"] == pytest.approx(700.0)
    assert res["risk_level"] == "HIGH"
    assert res["action"] == "OPERATOR_REVIEW"
    assert res["requires_human_review"] is True
    assert DECISION_LOG[0]["reason_code"] == "STATIC_EXCEEDS_DYNAMIC_RATING"


def test_invalid_weather_data_quality(tmp_path):
    df = _make_base_predictions()
    df.loc[df["segment_id"] == "SEG_1", "data_quality"] = "bad_sensor"

    validation = validate_predictions(df)
    assert validation["is_valid"] is False

    results = run_agent(df, config={"output_path": tmp_path / "out.json"})
    res = results[0]

    assert res["line_rating_q05_a"] is None
    assert res["risk_level"] == "UNCERTAIN"
    assert res["action"] == "FALLBACK_STATIC"
    assert res["requires_human_review"] is True
    assert DECISION_LOG[0]["reason_code"] == "INVALID_WEATHER_DATA_QUALITY"


def test_inconsistent_prediction_intervals(tmp_path):
    df = _make_base_predictions()
    df.loc[df["segment_id"] == "SEG_1", "q05_ampacity_a"] = 1500.0
    df.loc[df["segment_id"] == "SEG_1", "q50_ampacity_a"] = 1400.0

    validation = validate_predictions(df)
    assert validation["is_valid"] is False

    results = run_agent(df, config={"output_path": tmp_path / "out.json"})
    res = results[0]

    assert res["line_rating_q05_a"] is None
    assert res["risk_level"] == "UNCERTAIN"
    assert res["action"] == "FALLBACK_STATIC"
    assert res["requires_human_review"] is True
    assert DECISION_LOG[0]["reason_code"] == "INCONSISTENT_PREDICTION_INTERVALS"


def test_bottleneck_change_and_margins():
    df_hour_1 = _make_base_predictions()
    df_hour_2 = _make_base_predictions()
    df_hour_2["timestamp_utc"] = "2022-01-01T01:00:00Z"
    df_hour_2.loc[df_hour_2["segment_id"] == "SEG_3", "q05_ampacity_a"] = 1300.0

    combined = pd.concat([df_hour_1, df_hour_2], ignore_index=True)
    monitored = calculate_line_rating(combined, EXPECTED_SEGMENTS)

    assert len(monitored) == 2
    assert monitored.iloc[0]["bottleneck_segment"] == "SEG_1"
    assert bool(monitored.iloc[0]["bottleneck_changed"]) is False
    assert monitored.iloc[1]["bottleneck_segment"] == "SEG_3"
    assert bool(monitored.iloc[1]["bottleneck_changed"]) is True
    assert monitored.iloc[0]["min_segment_margin_a"] == pytest.approx(1413.5 - 762.0)


def test_stale_weather_data():
    decision = evaluate_line_decision(
        line_rating_q05_a=1400.0,
        static_rating_a=762.0,
        monitoring_status="VALID",
        is_stale=True,
    )
    assert decision["risk_level"] == "UNCERTAIN"
    assert decision["action"] == "FALLBACK_STATIC"
    assert decision["requires_human_review"] is True
    assert decision["reason_code"] == "STALE_WEATHER_DATA"

    from src.agent import load_expected_segments_from_config, summarize_agent_results


def test_agent_with_csv_and_json_config_files(tmp_path):
    df = _make_base_predictions()
    csv_path = tmp_path / "predictions.csv"
    df.to_csv(csv_path, index=False)

    config_path = tmp_path / "line_segments.json"
    out_path = tmp_path / "agent_results.json"
    log_path = tmp_path / "decision_log.json"

    config_data = {
        "segments": [
            {"segment_id": "SEG_1"},
            {"segment_id": "SEG_2"},
            {"segment_id": "SEG_3"},
        ],
        "output_path": str(out_path),
        "log_path": str(log_path),
    }
    config_path.write_text(json.dumps(config_data), encoding="utf-8")

    results = run_agent(predictions=csv_path, config=config_path)

    assert len(results) == 1
    assert out_path.exists()
    assert log_path.exists()

    summary = summarize_agent_results(results)
    assert summary["total_hours"] == 1
    assert summary["action_counts"]["RECOMMEND_DYNAMIC_RATING"] == 1
    assert summary["human_review_hours"] == 0
    assert summary["bottleneck_counts"]["SEG_1"] == 1


def test_load_expected_segments_formats(tmp_path):
    list_cfg = tmp_path / "list_cfg.json"
    list_cfg.write_text(json.dumps(["SEG_A", "SEG_B"]), encoding="utf-8")
    assert load_expected_segments_from_config(list_cfg) == ["SEG_A", "SEG_B"]

    missing_cfg = tmp_path / "does_not_exist.json"
    assert load_expected_segments_from_config(missing_cfg) == EXPECTED_SEGMENTS