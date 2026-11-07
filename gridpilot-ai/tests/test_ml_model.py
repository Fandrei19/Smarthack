"""Tests for physics_engine / ml_model / evaluation (Participant 2).

IMPORTANT: ML-pipeline tests use ``mock_ampacity`` - a smooth, NON-PHYSICAL stand-in label
function - purely to test pipeline mechanics (splits, shapes, ordering, error handling)
without LineRate. They say nothing about real ampacity. Tests that need the real LineRate
are marked with ``pytest.importorskip('linerate')`` and are skipped when it is missing.
Weather here is SYNTHETIC mock data, not NASA POWER.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import evaluation as ev
from src import ml_model as mm
from src import physics_engine as pe

ROOT = Path(__file__).resolve().parents[1]
SEGMENTS = pe.load_segments(ROOT / "data" / "demo" / "line_segments_mock.json")
COND = pe.load_conductor_config()
FAST_CFG = mm.TrainConfig(max_iter=40, backtest_folds=2)


# ------------------------------------------------------------------------ mock data
def make_mock_weather(days: int = 240, seed: int = 0) -> pd.DataFrame:
    """Synthetic hourly weather for the 3 mock segments (NOT real data)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-03-01", periods=days * 24, freq="h", tz="UTC")
    rows = []
    for seg in SEGMENTS:
        doy, hour = idx.dayofyear.to_numpy(), idx.hour.to_numpy()
        temp = 12 + 12 * np.sin(2 * np.pi * (doy - 110) / 365) + 5 * np.sin(2 * np.pi * (hour - 9) / 24) \
            + rng.normal(0, 2, len(idx))
        wind = rng.weibull(2.0, len(idx)) * 5.0
        wdir = rng.uniform(0, 360, len(idx))
        sin_alt = pe.solar_sin_altitude(idx, seg["latitude"], seg["longitude"])
        sol = np.clip(sin_alt, 0, None) * rng.uniform(300, 950, len(idx))
        rows.append(pd.DataFrame({"timestamp_utc": idx, "segment_id": seg["segment_id"], "air_temp_c": temp,
                                  "wind_speed_ms": wind, "wind_direction_deg": wdir, "solar_radiation": sol,
                                  "source": "MOCK", "data_quality": "ok"}))
    return pd.concat(rows, ignore_index=True)


def mock_ampacity(feats, cfg) -> float:
    """NON-PHYSICAL smooth label for pipeline tests only. Mimics qualitative trends."""
    if feats["wind_speed_ms"] > 15.0:  # exercise failure handling
        raise pe.PhysicsError("mock failure: wind > 15 m/s")
    az = float(feats["azimuth_deg"])
    d = abs((float(feats["wind_direction_deg"]) - az + 180) % 360 - 180)
    d = 180 - d if d > 90 else d
    cross = float(feats["wind_speed_ms"]) * np.sin(np.radians(d))
    return float(max(100.0, 1100 - 14 * (float(feats["air_temp_c"]) - 20) + 120 * np.sqrt(cross)
                     - 0.25 * float(feats["solar_radiation"])))


@pytest.fixture(scope="module")
def trained():
    weather = make_mock_weather()
    bundle, ds, report = mm.train_pipeline(weather, SEGMENTS, COND, FAST_CFG, ampacity_fn=mock_ampacity)
    return bundle, ds, report, weather


# ----------------------------------------------------------------------- evaluation
def test_chronological_split_has_no_overlap_and_gap():
    ds = make_mock_weather(days=60)
    s = ev.chronological_split(ds, gap_hours=24)
    assert s["train"]["timestamp_utc"].max() < s["val"]["timestamp_utc"].min()
    assert s["val"]["timestamp_utc"].max() < s["test"]["timestamp_utc"].min()
    assert s["val"]["timestamp_utc"].min() - s["train"]["timestamp_utc"].max() > pd.Timedelta(hours=24)


def test_chronological_split_rejects_tiny_data():
    with pytest.raises(ValueError):
        ev.chronological_split(make_mock_weather(days=1).head(5))


def test_lower_bound_metrics_counts_overestimation():
    y = np.array([100.0, 100.0, 100.0, 100.0])
    q = np.array([90.0, 110.0, 100.0, 80.0])
    m = ev.lower_bound_metrics(y, q)
    assert m["overestimation_count"] == 1 and m["overestimation_frequency"] == 0.25
    assert m["max_overestimation_a"] == 10.0 and m["lower_bound_coverage"] == 0.75


def test_conformal_shift_controls_overestimation_on_calibration_set():
    rng = np.random.default_rng(1)
    y = rng.normal(1000, 50, 2000)
    q = y + rng.normal(0, 30, 2000)
    shift = mm.conformal_lower_shift(y, q, 0.05)
    assert shift > 0
    assert np.mean((q - shift) > y) <= 0.05 + 1e-9


# ------------------------------------------------------------------------- dataset
def test_dataset_schema_and_failed_calculations_are_excluded(tmp_path):
    w = make_mock_weather(days=30)
    w.loc[[10, 500, 900], "wind_speed_ms"] = 20.0  # mock_ampacity raises PhysicsError for these
    ds = mm.build_training_dataset(w, SEGMENTS, COND, mock_ampacity, tmp_path / "ds.csv", tmp_path / "fail.csv")
    assert list(ds.columns) == ["timestamp_utc", "segment_id", *mm.FEATURE_COLUMNS, "data_quality", "target_ampacity_a"]
    assert ds["target_ampacity_a"].notna().all() and (ds["wind_speed_ms"] < 20).all()
    assert (tmp_path / "ds.csv").exists()
    assert len(ds) == len(w) - 3
    assert len(pd.read_csv(tmp_path / "fail.csv")) == 3
    # reproducible
    ds2 = mm.build_training_dataset(w, SEGMENTS, COND, mock_ampacity)
    pd.testing.assert_frame_equal(ds, ds2)


def test_generate_labels_flags_unknown_segment():
    w = make_mock_weather(days=2).head(6).copy()
    w.loc[0, "segment_id"] = "NOPE"
    out = pe.generate_labels(w, SEGMENTS, COND, mock_ampacity)
    assert out.loc[0, "label_status"] == "unknown_segment" and np.isnan(out.loc[0, "target_ampacity_a"])


def test_generate_labels_aborts_if_most_calculations_fail():
    w = make_mock_weather(days=2).head(30).copy()

    def always_fail(f, c):
        raise pe.PhysicsError("boom")
    with pytest.raises(RuntimeError):
        pe.generate_labels(w, SEGMENTS, COND, always_fail)


# --------------------------------------------------------------------- model export
def test_save_load_roundtrip_and_predict_schema(trained, tmp_path):
    bundle, _, _, weather = trained
    path = mm.save_model(bundle, tmp_path / "m.joblib")
    loaded = mm.load_model(path)
    pred = mm.predict_ampacity(weather.head(50), loaded)
    assert list(pred.columns) == mm.PREDICTION_COLUMNS
    assert len(pred) == 50
    pd.testing.assert_frame_equal(pred, mm.predict_ampacity(weather.head(50), bundle))


def test_load_model_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        mm.load_model(tmp_path / "missing.joblib")
    import joblib
    joblib.dump({"not": "a bundle"}, tmp_path / "bad.joblib")
    with pytest.raises(ValueError):
        mm.load_model(tmp_path / "bad.joblib")


def test_prediction_ordering_and_nonnegative(trained):
    bundle, _, _, weather = trained
    p = mm.predict_ampacity(weather, bundle)
    assert (p["q05_ampacity_a"] <= p["q50_ampacity_a"]).all()
    assert (p["q50_ampacity_a"] <= p["q95_ampacity_a"]).all()
    assert (p["q05_ampacity_a"] >= 0).all()
    assert (p["uncertainty_ratio"] >= 0).all()
    assert p["static_rating_a"].notna().all()


def test_missing_feature_column_raises(trained):
    bundle, _, _, weather = trained
    with pytest.raises(ValueError, match="wind_speed_ms"):
        mm.predict_ampacity(weather.head(5).drop(columns=["wind_speed_ms"]), bundle)


def test_invalid_values_give_nan_and_invalid_input_without_imputation(trained):
    bundle, _, _, weather = trained
    w = weather.head(6).copy().reset_index(drop=True)
    w["solar_radiation"] = w["solar_radiation"].astype(object)
    w.loc[0, "air_temp_c"] = np.nan          # missing
    w.loc[1, "wind_speed_ms"] = -3.0         # physically invalid
    w.loc[2, "wind_direction_deg"] = 400.0   # out of range
    w.loc[3, "solar_radiation"] = "abc"      # non-numeric
    w.loc[4, "segment_id"] = "UNKNOWN_SEG"   # unknown segment
    p = mm.predict_ampacity(w, bundle)
    assert len(p) == 6
    assert (p.loc[:4, "data_quality"] == mm.INVALID_DQ).all()
    assert p.loc[:4, ["q05_ampacity_a", "q50_ampacity_a", "q95_ampacity_a"]].isna().all().all()
    assert p.loc[5, "data_quality"] == "ok" and p.loc[5, "q50_ampacity_a"] > 0


def test_predict_preserves_row_order_and_utc_format(trained):
    bundle, _, _, weather = trained
    w = weather.sample(20, random_state=3)
    p = mm.predict_ampacity(w, bundle)
    assert list(p["segment_id"]) == list(w["segment_id"])
    assert p["timestamp_utc"].str.endswith("Z").all()


# ---------------------------------------------------------------- pipeline report
def test_report_contains_required_metrics_and_honesty_flags(trained):
    _, _, report, _ = trained
    assert report["labels_from_linerate_cigre601"] is False and report["WARNING"]
    dep = report["test"]["models"]["hgb_quantile_calibrated(DEPLOYED)"]
    for key in ("mae_a", "rmse_a", "pinball_q05", "lower_bound_coverage", "overestimation_frequency",
                "overestimation_count"):
        assert key in dep
    assert "baseline_linear(+conformal q05)" in report["test"]["models"]
    assert report["test_slices"]["low_wind (<1.0 m/s)"]["n"] >= 0
    assert "forecast" in report["uncertainty_statement"]


def test_ml_beats_mean_baseline_on_mock_labels(trained):
    _, _, report, _ = trained
    m = report["test"]["models"]
    assert m["hgb_point(squared_error)"]["mae_a"] < m["baseline_dummy_median(+const q05/q95)"]["mae_a"]


# --------------------------------------------------------------------- physics (no LineRate)
def test_solar_altitude_sanity():
    # summer-solstice solar noon at lon 28.4E is ~10:06 UTC; max altitude ~ 90 - 44.6 + 23.44 = 68.8 deg
    noon = pd.DatetimeIndex(["2025-06-21T10:06:00Z"])
    assert abs(float(pe.solar_sin_altitude(noon, 44.6, 28.4)[0]) - np.sin(np.radians(68.8))) < 0.02
    assert float(pe.solar_sin_altitude(pd.DatetimeIndex(["2025-06-21T22:00:00Z"]), 44.6, 28.4)[0]) < 0


def test_conductor_config_loads_and_matches_documented_example():
    c = COND["conductor"]
    assert c["conductor_diameter"] == 0.0281 and c["core_diameter"] == 0.0104
    assert c["resistance_at_temperature1"] == 7.283e-5 and c["resistance_at_temperature2"] == 8.688e-5
    assert pe._conductor_kwargs(c)["aluminium_cross_section_area"] != pe._conductor_kwargs(c)["aluminium_cross_section_area"]  # NaN


def test_calculate_ampacity_reports_missing_linerate_or_rejects_bad_input():
    feats = {"timestamp_utc": "2025-06-21T12:00:00Z", "air_temp_c": 20, "wind_speed_ms": 2, "wind_direction_deg": 90,
             "solar_radiation": 500, "latitude": 44.6, "longitude": 28.4, "azimuth_deg": 40}
    with pytest.raises(pe.PhysicsError):
        pe.calculate_ampacity({**feats, "wind_speed_ms": -1}, COND)
    with pytest.raises(pe.PhysicsError):
        pe.calculate_ampacity({k: v for k, v in feats.items() if k != "air_temp_c"}, COND)
    try:
        import linerate  # noqa: F401
    except ImportError:
        with pytest.raises(pe.LineRateNotAvailableError):
            pe.calculate_ampacity(feats, COND)


# --------------------------------------------------------------------- physics (needs LineRate)
@pytest.fixture(scope="module")
def lr():
    return pytest.importorskip("linerate")


def _f(**kw):
    base = {"timestamp_utc": "2025-06-21T10:00:00Z", "air_temp_c": 25.0, "wind_speed_ms": 3.0,
            "wind_direction_deg": 130.0, "solar_radiation": 700.0, "latitude": 44.6, "longitude": 28.4,
            "azimuth_deg": 40.0, "altitude_m": 60.0}
    base.update(kw)
    return base


def test_linerate_ampacity_finite_positive(lr):
    a = pe.calculate_ampacity(_f(), COND)
    assert np.isfinite(a) and 0 < a < 8000


def test_linerate_night_is_finite(lr):
    assert np.isfinite(pe.calculate_ampacity(_f(timestamp_utc="2025-06-21T23:00:00Z", solar_radiation=0.0), COND))


def test_linerate_physical_monotonicity(lr):
    assert pe.calculate_ampacity(_f(air_temp_c=35), COND) < pe.calculate_ampacity(_f(air_temp_c=10), COND)
    assert pe.calculate_ampacity(_f(wind_speed_ms=1), COND) < pe.calculate_ampacity(_f(wind_speed_ms=6), COND)
    # perpendicular wind cools better than parallel wind at equal speed
    perp = pe.calculate_ampacity(_f(wind_direction_deg=130.0), COND)   # 90 deg to azimuth 40
    para = pe.calculate_ampacity(_f(wind_direction_deg=40.0), COND)
    assert para < perp


def test_linerate_wind_direction_flip_symmetry(lr):
    a = pe.calculate_ampacity(_f(wind_direction_deg=100.0), COND)
    b = pe.calculate_ampacity(_f(wind_direction_deg=280.0), COND)
    assert a == pytest.approx(b, rel=1e-3)


def test_linerate_span_azimuth_matches_segment(lr):
    (la0, lo0, a0), (la1, lo1, a1) = pe._build_towers(44.6, 28.4, 40.0, 400.0, 60.0)
    span = lr.Span(conductor=lr.Conductor(**pe._conductor_kwargs(COND["conductor"])),
                   start_tower=lr.Tower(latitude=la0, longitude=lo0, altitude=a0),
                   end_tower=lr.Tower(latitude=la1, longitude=lo1, altitude=a1), num_conductors=1)
    assert np.degrees(span.conductor_azimuth) == pytest.approx(40.0, abs=0.5)


def test_linerate_static_rating_is_positive_for_all_mock_segments(lr):
    for s in SEGMENTS:
        assert pe.compute_static_rating(s, COND) > 0


# ------------------------------------------------------------- weights / regimes / CI (new)
def test_severity_weights_are_feature_based_and_severe_overrides_moderate():
    cfg = mm.TrainConfig()
    df = pd.DataFrame({"air_temp_c": [20, 29, 31, 31, 29], "wind_speed_ms": [0.5, 1.5, 0.5, 1.5, 3.0]})
    w = mm.severity_weights(df, cfg)
    assert list(w) == [1.0, 3.0, 5.0, 3.0, 1.0]
    assert set(mm.severity_weights(df, mm.TrainConfig(weight_moderate=1.0, weight_severe=1.0))) == {1.0}


def test_weighted_training_records_weights_and_regime_shifts(trained):
    bundle, _, report, _ = trained
    assert report["train_weight_counts"]
    for k in ("q05_shift_a", "q05_shift_day_a", "q05_shift_night_a"):
        assert bundle["models"][k] >= 0
    assert set(report["bootstrap_ci_test"]) == set(ev.CI_SLICES)


def test_regime_calibration_can_be_disabled():
    w = make_mock_weather(days=120)
    cfg = mm.TrainConfig(max_iter=30, backtest_folds=2, calibrate_by_regime=False)
    b, _, _ = mm.train_pipeline(w, SEGMENTS, COND, cfg, ampacity_fn=mock_ampacity)
    m = b["models"]
    assert m["q05_shift_day_a"] == m["q05_shift_night_a"] == m["q05_shift_a"]


def test_block_bootstrap_ci_is_ordered_and_handles_small_inputs():
    rng = np.random.default_rng(0)
    n = 2000
    y = rng.normal(1000, 50, n)
    q50, q05 = y + rng.normal(0, 5, n), y + rng.normal(-20, 15, n)
    days = np.repeat(np.arange(n // 20), 20)
    ci = ev.block_bootstrap_ci(y, q50, q05, days, seed=1)
    for k in ("mae_a", "overestimation_frequency"):
        assert ci[k]["ci_low"] <= ci[k]["estimate"] <= ci[k]["ci_high"]
    assert "note" in ev.block_bootstrap_ci(y[:20], q50[:20], q05[:20], days[:20])
    assert ev.block_bootstrap_ci(y, q50, q05, days, mask=np.zeros(n, bool)) == {"n": 0}


def test_compare_configs_uses_validation_only_and_requires_same_split():
    w = make_mock_weather(days=200)
    ds = mm.build_training_dataset(w, SEGMENTS, COND, mock_ampacity)
    base = mm.TrainConfig(max_iter=30)
    table = mm.compare_configs(ds, {"unweighted": mm.TrainConfig(max_iter=30, weight_moderate=1.0, weight_severe=1.0),
                                    "weighted": base})
    assert list(table.index) == ["unweighted", "weighted"] and (table["val_mae_a"] > 0).all()
    with pytest.raises(ValueError):
        mm.compare_configs(ds, {"a": base, "b": mm.TrainConfig(train_frac=0.5)})


def test_train_pipeline_accepts_prebuilt_dataset_and_skips_label_generation():
    w = make_mock_weather(days=150)
    ds = mm.build_training_dataset(w, SEGMENTS, COND, mock_ampacity)

    def must_not_label_rows(f, c):  # static rating calls are allowed (sun-free static weather), rows are not
        if f["solar_radiation"] != 0.0 or f.get("clearness_ratio") is None:
            raise AssertionError("label generation should have been skipped")
        return mock_ampacity(f, c)
    b, _, rep = mm.train_pipeline(None, SEGMENTS, COND, mm.TrainConfig(max_iter=30, backtest_folds=2),
                                  ampacity_fn=must_not_label_rows, dataset=ds)
    assert rep["test"]["models"]
    with pytest.raises(ValueError):
        mm.train_pipeline(None, SEGMENTS, COND, dataset=ds.drop(columns=["crosswind_ms"]))


def test_monotonic_constraint_is_respected():
    w = make_mock_weather(days=150)
    ds = mm.build_training_dataset(w, SEGMENTS, COND, mock_ampacity)
    cfg = mm.TrainConfig(max_iter=40, monotonic=True)
    splits = ev.chronological_split(ds)
    m = mm.fit_models(splits["train"], splits["val"], cfg)
    probe = splits["test"][mm.FEATURE_COLUMNS].iloc[[5]].copy()
    grid = pd.concat([probe] * 40, ignore_index=True)
    grid["air_temp_c"] = np.linspace(-10, 40, 40)
    assert np.all(np.diff(m["hgb_q50"].predict(grid)) <= 1e-9)  # ampacity never increases with temperature
    grid = pd.concat([probe] * 40, ignore_index=True)
    grid["wind_speed_ms"] = np.linspace(0, 12, 40)
    assert np.all(np.diff(m["hgb_q50"].predict(grid)) >= -1e-9)  # never decreases with wind speed


def test_old_bundle_without_regime_shifts_still_predicts(trained):
    bundle, _, _, weather = trained
    old = dict(bundle)
    old["models"] = {k: v for k, v in bundle["models"].items() if k not in ("q05_shift_day_a", "q05_shift_night_a")}
    p = mm.predict_ampacity(weather.head(30), old)
    assert (p["q05_ampacity_a"] <= p["q50_ampacity_a"]).all()