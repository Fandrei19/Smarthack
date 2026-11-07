"""Ampacity ML model: features, dataset, training, calibration, export, prediction.

SIMULATION ONLY. Labels are produced by the CIGRE TB 601 implementation in LineRate for
a SIMULATED conductor/line (see physics_engine + config/conductor_config.json). The model
never saw real PPC / grid measurements.

Units: temperatures degC, wind m/s, wind direction deg (meteorological, clockwise from
north), solar_radiation W/m^2 (GHI), ampacity A, timestamps UTC.

Uncertainty - what the q05/q50/q95 outputs ARE and ARE NOT (see also UNCERTAINTY_STATEMENT):
    ARE:     approximation error of a gradient-boosting model that imitates the physics
             function weather -> ampacity (model/regression uncertainty), calibrated on a
             later time block.
    ARE NOT: (a) weather *forecast* error - the model is trained on observed/reanalysis
             weather and fed a forecast at inference, so real-world spread is larger;
             (b) conductor parameter, sag, ageing or CIGRE model-form error - labels are a
             deterministic function of weather, so the data contain no such noise;
             (c) any real grid measurement.

CLI:
    python -m src.ml_model train   --weather data/processed/weather.csv --segments config/line_segments.json
    python -m src.ml_model predict --weather data/processed/weather.csv --out data/processed/predictions.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from src import evaluation as ev
from src import physics_engine as pe

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "ampacity_model.joblib"
DEFAULT_DATASET_PATH = ROOT / "data" / "processed" / "training_dataset.csv"

REQUIRED_WEATHER_COLUMNS = ("timestamp_utc", "segment_id", "air_temp_c", "wind_speed_ms",
                            "wind_direction_deg", "solar_radiation")
PREDICTION_COLUMNS = ["timestamp_utc", "segment_id", "q05_ampacity_a", "q50_ampacity_a",
                      "q95_ampacity_a", "static_rating_a", "uncertainty_ratio", "data_quality"]
FEATURE_COLUMNS = ["air_temp_c", "wind_speed_ms", "wind_angle_of_attack_deg", "crosswind_ms",
                   "parallel_wind_ms", "solar_radiation", "sin_solar_altitude",
                   "hour_sin", "hour_cos", "doy_sin", "doy_cos", "line_azimuth_deg"]
INVALID_DQ = "invalid_input"

UNCERTAINTY_STATEMENT = (
    "q05/q50/q95 quantify only the approximation error of an ML model fitted to a deterministic, "
    "synthetic physics function (CIGRE TB 601 via LineRate) of observed weather. They do NOT include "
    "weather-forecast error (the model is trained on historical weather, used with a forecast), "
    "conductor-parameter/sag/ageing uncertainty, CIGRE model-form error, or any real grid measurement. "
    "Real-world uncertainty is therefore larger than the reported intervals."
)


# ------------------------------------------------------------------------- config
@dataclass(frozen=True)
class TrainConfig:
    """Training hyper-parameters (fixed -> reproducible; no randomness in HGB when early_stopping=False)."""
    random_state: int = 42
    train_frac: float = 0.70
    val_frac: float = 0.15
    gap_hours: float = 24.0
    low_alpha: float = 0.05
    high_alpha: float = 0.95
    max_iter: int = 300
    learning_rate: float = 0.08
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 40
    l2_regularization: float = 1.0
    backtest_folds: int = 4


def _hgb(cfg: TrainConfig, **kw: Any) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=cfg.max_iter, learning_rate=cfg.learning_rate, max_leaf_nodes=cfg.max_leaf_nodes,
        min_samples_leaf=cfg.min_samples_leaf, l2_regularization=cfg.l2_regularization,
        early_stopping=False, random_state=cfg.random_state, **kw)


# ------------------------------------------------------------------ validation/features
def validate_weather(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Coerce types and flag invalid rows (NaN, non-numeric, out-of-range).

    Returns (clean copy with parsed UTC timestamps, boolean ``valid`` mask).
    Raises ValueError if a required column is missing entirely.
    """
    missing = [c for c in REQUIRED_WEATHER_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"weather data is missing required columns: {missing}")
    out = df.copy()
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce")
    for c in ("air_temp_c", "wind_speed_ms", "wind_direction_deg", "solar_radiation"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if "data_quality" not in out.columns:
        logger.warning("weather data has no 'data_quality' column; filling with 'unknown'")
        out["data_quality"] = "unknown"
    out["data_quality"] = out["data_quality"].fillna("unknown").astype(str)
    valid = (
        out["timestamp_utc"].notna() & out["segment_id"].notna()
        & out["air_temp_c"].between(-60, 60) & out["wind_speed_ms"].between(0, 80)
        & out["wind_direction_deg"].between(0, 360) & out["solar_radiation"].between(0, 1500)
    )  # between() is False for NaN
    return out, valid


def add_features(df: pd.DataFrame, segments: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    """Add geometry + engineered features. Rows must already be valid & have known segment_id.

    crosswind_ms = v*sin(a), parallel_wind_ms = v*cos(a), where a in [0, 90] deg is the acute
    angle between wind axis and line axis (direction-flip symmetric, like CIGRE).
    """
    out = df.copy()
    seg_ids = out["segment_id"].astype(str)
    for col, key in (("latitude", "latitude"), ("longitude", "longitude"), ("line_azimuth_deg", "azimuth_deg")):
        out[col] = seg_ids.map(lambda s: float(segments[s][key]))
    delta = np.abs((out["wind_direction_deg"] - out["line_azimuth_deg"] + 180.0) % 360.0 - 180.0)
    aoa = np.where(delta > 90.0, 180.0 - delta, delta)
    out["wind_angle_of_attack_deg"] = aoa
    out["crosswind_ms"] = out["wind_speed_ms"] * np.sin(np.radians(aoa))
    out["parallel_wind_ms"] = out["wind_speed_ms"] * np.cos(np.radians(aoa))
    sin_alt = np.empty(len(out))
    for sid in seg_ids.unique():
        m = (seg_ids == sid).to_numpy()
        g = segments[sid]
        sin_alt[m] = pe.solar_sin_altitude(out.loc[m, "timestamp_utc"], float(g["latitude"]), float(g["longitude"]))
    out["sin_solar_altitude"] = sin_alt
    ts = out["timestamp_utc"].dt
    hour = ts.hour + ts.minute / 60.0
    out["hour_sin"], out["hour_cos"] = np.sin(2 * np.pi * hour / 24.0), np.cos(2 * np.pi * hour / 24.0)
    doy = ts.dayofyear
    out["doy_sin"], out["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    return out


def _geo_map(segments: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(s["segment_id"]): dict(s) for s in segments}


# ---------------------------------------------------------------------------- dataset
def build_training_dataset(weather: pd.DataFrame, segments: Sequence[Mapping[str, Any]],
                           conductor_config: Mapping[str, Any],
                           ampacity_fn: pe.AmpacityFn = pe.calculate_ampacity,
                           out_path: str | Path | None = None,
                           failures_path: str | Path | None = None) -> pd.DataFrame:
    """weather.csv rows -> features + ``target_ampacity_a`` (physics reference label).

    Invalid weather rows and failed physical calculations are logged and excluded
    (failures optionally written to ``failures_path``). Output is sorted by (time, segment),
    so identical inputs give an identical file.
    """
    geo = _geo_map(segments)
    clean, valid = validate_weather(weather)
    known = clean["segment_id"].astype(str).isin(geo)
    n_inv, n_unk = int((~valid).sum()), int((valid & ~known).sum())
    if n_inv or n_unk:
        logger.warning("dropping %d invalid-weather rows and %d rows with unknown segment_id", n_inv, n_unk)
    clean = clean[valid & known].copy()
    clean["segment_id"] = clean["segment_id"].astype(str)
    clean = clean.sort_values(["timestamp_utc", "segment_id"]).reset_index(drop=True)
    if clean.empty:
        raise ValueError("no valid weather rows to build a training dataset from")

    labelled = pe.generate_labels(clean, segments, conductor_config, ampacity_fn=ampacity_fn)
    failed = labelled[labelled["label_status"] != "ok"]
    if len(failed) and failures_path:
        Path(failures_path).parent.mkdir(parents=True, exist_ok=True)
        failed[["timestamp_utc", "segment_id", "label_status"]].to_csv(failures_path, index=False)
    ok = labelled[labelled["label_status"] == "ok"].copy()
    ds = add_features(ok, geo)
    keep = ["timestamp_utc", "segment_id", *FEATURE_COLUMNS, "data_quality", "target_ampacity_a"]
    ds = ds[keep].reset_index(drop=True)
    logger.info("training dataset: %d rows (%d label failures excluded)", len(ds), len(failed))
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        ds.to_csv(out_path, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    return ds


# --------------------------------------------------------------------------- training
def conformal_lower_shift(y: np.ndarray, q_low: np.ndarray, miss: float = 0.05) -> float:
    """Shift [A] to subtract from a lower-bound prediction so that on the calibration set
    at most ~``miss`` of rows have q_low > y (split-conformal, one-sided).
    Guarantee holds only under exchangeability; under a seasonal drift it is approximate."""
    s = np.sort(np.asarray(q_low, float) - np.asarray(y, float))
    k = min(len(s), int(np.ceil((len(s) + 1) * (1.0 - miss))))
    return float(max(0.0, s[k - 1]))


def fit_models(train: pd.DataFrame, val: pd.DataFrame, cfg: TrainConfig) -> dict[str, Any]:
    """Fit baselines + HGB models on ``train``; calibrate lower bounds on ``val`` (later in time)."""
    X, y = train[FEATURE_COLUMNS], train["target_ampacity_a"].to_numpy()
    Xv, yv = val[FEATURE_COLUMNS], val["target_ampacity_a"].to_numpy()
    m: dict[str, Any] = {
        "dummy_median": DummyRegressor(strategy="median").fit(X, y),
        "dummy_q05": DummyRegressor(strategy="quantile", quantile=cfg.low_alpha).fit(X, y),
        "dummy_q95": DummyRegressor(strategy="quantile", quantile=cfg.high_alpha).fit(X, y),
        "linear": LinearRegression().fit(X, y),
        "hgb_point": _hgb(cfg, loss="squared_error").fit(X, y),
        "hgb_q05": _hgb(cfg, loss="quantile", quantile=cfg.low_alpha).fit(X, y),
        "hgb_q50": _hgb(cfg, loss="quantile", quantile=0.5).fit(X, y),
        "hgb_q95": _hgb(cfg, loss="quantile", quantile=cfg.high_alpha).fit(X, y),
    }
    m["q05_shift_a"] = conformal_lower_shift(yv, m["hgb_q05"].predict(Xv), cfg.low_alpha)
    m["linear_shift_a"] = conformal_lower_shift(yv, m["linear"].predict(Xv), cfg.low_alpha)
    m["dummy_q05_shift_a"] = 0.0
    return m


def predict_quantiles(models: Mapping[str, Any], X: pd.DataFrame) -> dict[str, np.ndarray]:
    """Deployed quantile estimates: calibrated lower bound, median, upper; ordered q05<=q50<=q95.

    Quantile crossing (independent models) is repaired by sorting the three values per row,
    which can only lower q05 / raise q95 (conservative). ``crossing`` flags repaired rows.
    """
    q05 = models["hgb_q05"].predict(X) - models["q05_shift_a"]
    q50, q95 = models["hgb_q50"].predict(X), models["hgb_q95"].predict(X)
    stacked = np.vstack([q05, q50, q95])
    crossing = np.any(np.diff(stacked, axis=0) < 0, axis=0)
    s = np.sort(stacked, axis=0)
    return {"q05": np.maximum(s[0], 0.0), "q50": np.maximum(s[1], 0.0), "q95": np.maximum(s[2], 0.0),
            "crossing": crossing}


def _evaluate_split(df: pd.DataFrame, models: Mapping[str, Any], static: Mapping[str, float]) -> dict[str, Any]:
    X, y = df[FEATURE_COLUMNS], df["target_ampacity_a"].to_numpy()
    dq = predict_quantiles(models, X)
    lin = models["linear"].predict(X)
    per_model = {
        "baseline_dummy_median(+const q05/q95)": ev.evaluate_predictions(y, {
            "q50": models["dummy_median"].predict(X), "q05": models["dummy_q05"].predict(X),
            "q95": models["dummy_q95"].predict(X)}),
        "baseline_linear(+conformal q05)": ev.evaluate_predictions(y, {
            "q50": lin, "q05": np.maximum(lin - models["linear_shift_a"], 0.0)}),
        "hgb_point(squared_error)": ev.evaluate_predictions(y, {"q50": models["hgb_point"].predict(X)}),
        "hgb_quantile_raw_q05": ev.evaluate_predictions(y, {
            "q50": models["hgb_q50"].predict(X), "q05": models["hgb_q05"].predict(X),
            "q95": models["hgb_q95"].predict(X)}),
        "hgb_quantile_calibrated(DEPLOYED)": ev.evaluate_predictions(
            y, {"q50": dq["q50"], "q05": dq["q05"], "q95": dq["q95"]}),
    }
    per_model["hgb_quantile_calibrated(DEPLOYED)"]["quantile_crossing_fraction"] = float(dq["crossing"].mean())
    st = df["segment_id"].map(static).to_numpy(dtype=float)
    return {"models": per_model,
            "static_vs_dlr": ev.static_rating_comparison(y, dq["q05"], st),
            "_deployed_preds": dq}


def _time_cut(df: pd.DataFrame, frac: float, gap_hours: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    uniq = np.sort(df["timestamp_utc"].unique())
    t = pd.Timestamp(uniq[int(len(uniq) * frac) - 1])
    t = t.tz_localize("UTC") if t.tzinfo is None else t
    gap = pd.Timedelta(hours=gap_hours)
    return df[df["timestamp_utc"] <= t], df[df["timestamp_utc"] > t + gap]


def rolling_origin_backtest(dataset: pd.DataFrame, cfg: TrainConfig,
                            static: Mapping[str, float]) -> list[dict[str, Any]]:
    """Expanding-window backtest: each fold fits on the past (last 20% of it used for
    calibration) and is scored on the following block. Shows how stable the safety metrics are."""
    rows = []
    for k, (tr, te) in enumerate(ev.rolling_origin_splits(dataset, cfg.backtest_folds, gap_hours=cfg.gap_hours)):
        fit, cal = _time_cut(tr, 0.8, cfg.gap_hours)
        if len(fit) < 100 or len(cal) < 50:
            continue
        mods = fit_models(fit, cal, cfg)
        res = _evaluate_split(te, mods, static)["models"]
        dep, raw = res["hgb_quantile_calibrated(DEPLOYED)"], res["hgb_quantile_raw_q05"]
        rows.append({"fold": k, "n_train": len(fit), "n_test": len(te),
                     "test_start": str(te["timestamp_utc"].min()), "test_end": str(te["timestamp_utc"].max()),
                     "mae_a": dep["mae_a"], "overestimation_freq_calibrated": dep["overestimation_frequency"],
                     "overestimation_freq_raw_q05": raw["overestimation_frequency"],
                     "max_overestimation_a": dep["max_overestimation_a"]})
    return rows


def train_pipeline(weather: pd.DataFrame, segments: Sequence[Mapping[str, Any]],
                   conductor_config: Mapping[str, Any], cfg: TrainConfig = TrainConfig(),
                   ampacity_fn: pe.AmpacityFn = pe.calculate_ampacity,
                   dataset_path: str | Path | None = None, failures_path: str | Path | None = None
                   ) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Dataset -> chronological split -> fit -> evaluate -> deployable bundle.

    Returns (bundle, dataset, report). The bundle holds only the deployed quantile models.
    """
    ds = build_training_dataset(weather, segments, conductor_config, ampacity_fn, dataset_path, failures_path)
    geo = _geo_map(segments)
    static = {sid: pe.compute_static_rating(g, conductor_config, ampacity_fn) for sid, g in geo.items()
              if sid in set(ds["segment_id"])}
    splits = ev.chronological_split(ds, cfg.train_frac, cfg.val_frac, cfg.gap_hours)
    train, val, test = splits["train"], splits["val"], splits["test"]
    logger.info("split rows train/val/test = %d/%d/%d", len(train), len(val), len(test))
    models = fit_models(train, val, cfg)

    res_test = _evaluate_split(test, models, static)
    dq = res_test.pop("_deployed_preds")
    slices = ev.slice_analysis(test.reset_index(drop=True), test["target_ampacity_a"].to_numpy(),
                               {"q50": dq["q50"], "q05": dq["q05"], "q95": dq["q95"]})
    res_val = _evaluate_split(val, models, static)
    res_val.pop("_deployed_preds")
    is_real = ampacity_fn is pe.calculate_ampacity
    report = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "labels_from_linerate_cigre601": is_real,
        "WARNING": None if is_real else "LABELS NOT FROM LINERATE (mock ampacity_fn). Results are NOT physical.",
        "uncertainty_statement": UNCERTAINTY_STATEMENT,
        "split": {k: {"rows": len(v), "start": str(v["timestamp_utc"].min()), "end": str(v["timestamp_utc"].max())}
                  for k, v in splits.items()} | {"gap_hours": cfg.gap_hours},
        "q05_conformal_shift_a": models["q05_shift_a"],
        "test": res_test, "validation": res_val,
        "test_slices": slices, "slice_note": ev.SLICE_NOTE,
        "rolling_origin_backtest": rolling_origin_backtest(ds, cfg, static),
        "static_ratings_a": static, "config": asdict(cfg),
    }
    deployed = {k: models[k] for k in ("hgb_q05", "hgb_q50", "hgb_q95", "q05_shift_a")}
    bundle = {
        "format_version": 1,
        "feature_columns": list(FEATURE_COLUMNS),
        "models": deployed,
        "segments": {sid: {k: g[k] for k in ("latitude", "longitude", "azimuth_deg")} for sid, g in geo.items()},
        "static_rating_a": static,
        "training_ranges": {c: [float(ds[c].min()), float(ds[c].max())] for c in FEATURE_COLUMNS},
        "metadata": {
            "created_utc": report["created_utc"], "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__, "pandas_version": pd.__version__,
            "train_config": asdict(cfg), "n_training_rows": int(len(train)),
            "dataset_hash": hashlib.sha256(pd.util.hash_pandas_object(ds, index=False).values.tobytes()).hexdigest(),
            "labels_from_linerate_cigre601": is_real,
            "label_source": "linerate.Cigre601 (CIGRE TB 601), SIMULATED conductor" if is_real
            else "MOCK ampacity_fn - NOT PHYSICAL",
            "uncertainty_statement": UNCERTAINTY_STATEMENT,
        },
    }
    return bundle, ds, report


# ------------------------------------------------------------------------ export/load
def save_model(bundle: Mapping[str, Any], path: str | Path = DEFAULT_MODEL_PATH) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(bundle), p)
    return p


def load_model(path: str | Path | None = None) -> dict[str, Any]:
    """Load a bundle saved by :func:`save_model`. Only load files you trust (joblib = pickle)."""
    p = Path(path) if path else DEFAULT_MODEL_PATH
    if not p.exists():
        raise FileNotFoundError(f"model file not found: {p}")
    bundle = joblib.load(p)
    needed = {"format_version", "feature_columns", "models", "segments", "static_rating_a", "metadata"}
    if not isinstance(bundle, dict) or not needed <= bundle.keys():
        raise ValueError(f"{p} is not a valid GridPilot model bundle")
    if bundle["metadata"].get("sklearn_version") != sklearn.__version__:
        logger.warning("model trained with scikit-learn %s, running %s",
                       bundle["metadata"].get("sklearn_version"), sklearn.__version__)
    return bundle


def predict_ampacity(weather: pd.DataFrame | str | Path, bundle: Mapping[str, Any] | None = None,
                     model_path: str | Path | None = None) -> pd.DataFrame:
    """Predict per-segment ampacity [A] from weather rows (observed or forecast).

    Output has exactly PREDICTION_COLUMNS, one row per input row, same order.
    Rows that are invalid (NaN/out-of-range values, unparsable timestamp, unknown segment)
    get NaN predictions and ``data_quality='invalid_input'`` - they are never imputed or
    guessed, so downstream logic can fall back conservatively.
    Raises ValueError if a required column is missing entirely.
    ``uncertainty_ratio`` = (q95 - q05) / q50 [-], i.e. MODEL uncertainty only (see module docstring).
    """
    if isinstance(weather, (str, Path)):
        weather = pd.read_csv(weather)
    bundle = bundle or load_model(model_path)
    clean, valid = validate_weather(weather)
    geo = bundle["segments"]
    seg = clean["segment_id"].astype(str)
    ok = (valid & seg.isin(geo)).to_numpy()
    n = len(clean)
    q05, q50, q95, stat = (np.full(n, np.nan) for _ in range(4))
    if ok.any():
        feats = add_features(clean.loc[ok].assign(segment_id=seg[ok]), geo)
        X = feats[bundle["feature_columns"]]
        rng = bundle.get("training_ranges", {})
        oor = [c for c in X.columns if c in rng and ((X[c] < rng[c][0]).any() or (X[c] > rng[c][1]).any())]
        if oor:
            logger.warning("inputs outside training range for %s - tree models extrapolate flat; "
                           "treat predictions with caution", oor)
        mods = bundle["models"]
        dq = predict_quantiles(mods, X)
        q05[ok], q50[ok], q95[ok] = dq["q05"], dq["q50"], dq["q95"]
        stat[ok] = seg[ok].map(bundle["static_rating_a"]).to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(q50 > 0, (q95 - q05) / q50, np.nan)
    ts = clean["timestamp_utc"]
    ts_out = ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ").where(ts.notna(), weather["timestamp_utc"].astype(str))
    return pd.DataFrame({
        "timestamp_utc": ts_out.to_numpy(), "segment_id": seg.to_numpy(),
        "q05_ampacity_a": q05, "q50_ampacity_a": q50, "q95_ampacity_a": q95,
        "static_rating_a": stat, "uncertainty_ratio": ratio,
        "data_quality": np.where(ok, clean["data_quality"].to_numpy(), INVALID_DQ),
    })[PREDICTION_COLUMNS]


# ------------------------------------------------------------------------------ CLI
def _write_report(report: Mapping[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evaluation_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    cols = ["n", "mae_a", "rmse_a", "pinball_q05", "pinball_q50", "pinball_q95", "lower_bound_coverage",
            "overestimation_frequency", "overestimation_count", "max_overestimation_a", "interval_90_coverage"]
    md = [f"# Evaluation report ({report['created_utc']})", "",
          f"**Labels from LineRate CIGRE601:** {report['labels_from_linerate_cigre601']}",
          report.get("WARNING") or "", "", f"> {report['uncertainty_statement']}", "",
          f"Split: `{json.dumps(report['split'])}`", "",
          ev.results_to_markdown("Test set - models", report["test"]["models"], cols), "",
          ev.results_to_markdown("Test set - deployed model by weather slice", report["test_slices"], cols), "",
          f"_{report['slice_note']}_", "", "### Static vs DLR lower bound (test)", "",
          f"`{json.dumps(report['test']['static_vs_dlr'])}`", "", "### Rolling-origin backtest", "",
          ev.results_to_markdown("folds", {str(r['fold']): r for r in report["rolling_origin_backtest"]},
                                 ["n_test", "mae_a", "overestimation_freq_calibrated",
                                  "overestimation_freq_raw_q05", "max_overestimation_a"])]
    (out_dir / "evaluation_report.md").write_text("\n".join(md), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GridPilot ampacity ML (Participant 2)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--weather", default=str(ROOT / "data/processed/weather.csv"))
    t.add_argument("--segments", default=str(ROOT / "config/line_segments.json"))
    t.add_argument("--conductor-config", default=None)
    t.add_argument("--dataset-out", default=str(DEFAULT_DATASET_PATH))
    t.add_argument("--model-out", default=str(DEFAULT_MODEL_PATH))
    t.add_argument("--report-dir", default=str(ROOT / "models"))
    p = sub.add_parser("predict")
    p.add_argument("--weather", required=True)
    p.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--out", default=str(ROOT / "data/processed/predictions.csv"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.cmd == "train":
        cond = pe.load_conductor_config(args.conductor_config)
        segs = pe.load_segments(args.segments)
        weather = pd.read_csv(args.weather)
        bundle, _, report = train_pipeline(
            weather, segs, cond, dataset_path=args.dataset_out,
            failures_path=Path(args.dataset_out).with_name("training_label_failures.csv"))
        save_model(bundle, args.model_out)
        _write_report(report, Path(args.report_dir))
        logger.info("saved model -> %s, report -> %s", args.model_out, args.report_dir)
    else:
        out = predict_ampacity(args.weather, model_path=args.model)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out, index=False)
        logger.info("wrote %d predictions -> %s", len(out), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())