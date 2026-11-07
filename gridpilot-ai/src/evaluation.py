"""Chronological splitting and metrics for the ampacity models.

Units: ampacity in A; timestamps UTC. No dependency on ml_model / dashboard.

Conventions for the conservative lower bound q05:
    * coverage                = P(y_ref >= q05)      (target >= 0.95)
    * overestimation frequency = P(q05 > y_ref)       (target <= 0.05)  <- the safety metric
    * overestimation magnitude = mean/max of (q05 - y_ref) over violating rows [A]
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_pinball_loss


# ------------------------------------------------------------------------ splitting
def chronological_split(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15,
                        gap_hours: float = 24.0, time_col: str = "timestamp_utc") -> dict[str, pd.DataFrame]:
    """Split by *time* (all segments share the same cut-offs), never by random rows.

    A gap (embargo) of ``gap_hours`` is dropped between train|val and val|test to limit
    leakage through the strong hour-to-hour autocorrelation of weather.
    """
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("need 0 < train_frac, val_frac and train_frac + val_frac < 1")
    ts = pd.to_datetime(df[time_col], utc=True)
    uniq = np.sort(ts.unique())
    if len(uniq) < 10:
        raise ValueError(f"Too few distinct timestamps ({len(uniq)}) for a chronological split")
    t_train_end = uniq[int(len(uniq) * train_frac) - 1]
    t_val_end = uniq[int(len(uniq) * (train_frac + val_frac)) - 1]
    gap = pd.Timedelta(hours=gap_hours)
    t_train_end, t_val_end = pd.Timestamp(t_train_end), pd.Timestamp(t_val_end)
    if t_train_end.tzinfo is None:
        t_train_end, t_val_end = t_train_end.tz_localize("UTC"), t_val_end.tz_localize("UTC")
    train = df[ts <= t_train_end]
    val = df[(ts > t_train_end + gap) & (ts <= t_val_end)]
    test = df[ts > t_val_end + gap]
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("Chronological split produced an empty partition; need more data or a smaller gap")
    return {"train": train, "val": val, "test": test}


def rolling_origin_splits(df: pd.DataFrame, n_folds: int = 4, initial_frac: float = 0.5,
                          gap_hours: float = 24.0, time_col: str = "timestamp_utc"
                          ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Expanding-window backtest folds: train on [start, t_k], test on the next block."""
    ts = pd.to_datetime(df[time_col], utc=True)
    uniq = pd.DatetimeIndex(np.sort(ts.unique()))
    if uniq.tz is None:
        uniq = uniq.tz_localize("UTC")
    n = len(uniq)
    start = int(n * initial_frac)
    block = (n - start) // n_folds
    if block < 2:
        raise ValueError("Not enough timestamps for the requested number of folds")
    gap = pd.Timedelta(hours=gap_hours)
    folds = []
    for k in range(n_folds):
        t_tr_end = uniq[start + k * block - 1]
        t_te_end = uniq[min(n - 1, start + (k + 1) * block - 1)]
        tr = df[ts <= t_tr_end]
        te = df[(ts > t_tr_end + gap) & (ts <= t_te_end)]
        if len(te):
            folds.append((tr, te))
    return folds


# -------------------------------------------------------------------------- metrics
def point_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """MAE and RMSE [A]."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    return {"mae_a": float(mean_absolute_error(y, pred)),
            "rmse_a": float(np.sqrt(np.mean((y - pred) ** 2)))}


def lower_bound_metrics(y: np.ndarray, q_low: np.ndarray, nominal_miss: float = 0.05) -> dict[str, float]:
    """Safety metrics for a conservative lower bound (see module docstring)."""
    y, q = np.asarray(y, float), np.asarray(q_low, float)
    over = q - y
    viol = over > 0
    n_viol = int(viol.sum())
    return {
        "lower_bound_coverage": float(1.0 - viol.mean()),
        "overestimation_frequency": float(viol.mean()),
        "overestimation_count": n_viol,
        "nominal_overestimation_frequency": nominal_miss,
        "mean_overestimation_a": float(over[viol].mean()) if n_viol else 0.0,
        "max_overestimation_a": float(over[viol].max()) if n_viol else 0.0,
        "mean_conservatism_a": float(-over[~viol].mean()) if (~viol).any() else 0.0,
    }


def evaluate_predictions(y: np.ndarray, preds: Mapping[str, np.ndarray],
                         low_alpha: float = 0.05, high_alpha: float = 0.95) -> dict[str, Any]:
    """Metrics for one model. ``preds`` may hold any of 'q50' (point/median), 'q05', 'q95'."""
    y = np.asarray(y, float)
    out: dict[str, Any] = {"n": int(len(y))}
    if "q50" in preds:
        out.update(point_metrics(y, preds["q50"]))
        out["pinball_q50"] = float(mean_pinball_loss(y, preds["q50"], alpha=0.5))
    if "q05" in preds:
        out["pinball_q05"] = float(mean_pinball_loss(y, preds["q05"], alpha=low_alpha))
        out.update(lower_bound_metrics(y, preds["q05"], nominal_miss=low_alpha))
    if "q95" in preds:
        out["pinball_q95"] = float(mean_pinball_loss(y, preds["q95"], alpha=high_alpha))
        out["upper_bound_coverage"] = float(np.mean(y <= preds["q95"]))
    if "q05" in preds and "q95" in preds:
        out["interval_90_coverage"] = float(np.mean((y >= preds["q05"]) & (y <= preds["q95"])))
        out["mean_interval_width_a"] = float(np.mean(preds["q95"] - preds["q05"]))
    return out


def static_rating_comparison(y: np.ndarray, q05: np.ndarray, static: np.ndarray) -> dict[str, float]:
    """How the DLR lower bound compares with the static rating (all A)."""
    y, q05, st = (np.asarray(a, float) for a in (y, q05, static))
    return {
        "frac_q05_above_static": float(np.mean(q05 > st)),
        "mean_q05_minus_static_a": float(np.mean(q05 - st)),
        "frac_static_above_reference": float(np.mean(st > y)),  # static would be unsafe
    }


# -------------------------------------------------------------------- slice analysis
DEFAULT_SLICES: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "all": lambda d: pd.Series(True, index=d.index),
    "low_wind (<1.0 m/s)": lambda d: d["wind_speed_ms"] < 1.0,
    "low_crosswind (<0.5 m/s)": lambda d: d["crosswind_ms"] < 0.5,
    "high_temp (>=30 C)": lambda d: d["air_temp_c"] >= 30.0,
    "low_wind & high_temp": lambda d: (d["wind_speed_ms"] < 1.0) & (d["air_temp_c"] >= 30.0),
    # thresholds below must match TrainConfig.moderate_/severe_ defaults (28 C & 2 m/s; 30 C & 1 m/s)
    "moderate_severity (T>=28C & wind<2)": lambda d: (d["air_temp_c"] >= 28.0) & (d["wind_speed_ms"] < 2.0),
    "severe (T>=30C & wind<1)": lambda d: (d["air_temp_c"] >= 30.0) & (d["wind_speed_ms"] < 1.0),
    "daytime (sun up)": lambda d: d["sin_solar_altitude"] > 0.0,
    "night": lambda d: d["sin_solar_altitude"] <= 0.0,
    "strong_sun (GHI>=600 W/m2)": lambda d: d["solar_radiation"] >= 600.0,
    "high_wind (>=8 m/s)": lambda d: d["wind_speed_ms"] >= 8.0,
}
MODERATE_SLICE = "moderate_severity (T>=28C & wind<2)"
CI_SLICES = ("all", "daytime (sun up)", "night", MODERATE_SLICE, "severe (T>=30C & wind<1)")


def block_bootstrap_ci(y: np.ndarray, q50: np.ndarray, q05: np.ndarray, groups: np.ndarray,
                       mask: np.ndarray | None = None, n_boot: int = 300, seed: int = 0,
                       level: float = 0.95, min_groups: int = 10) -> dict[str, Any]:
    """Block-bootstrap CI (resampling whole days, since hourly errors are autocorrelated and
    segments share weather) for MAE of q50 [A] and overestimation frequency of q05."""
    y, q50, q05 = (np.asarray(a, float) for a in (y, q50, q05))
    m = np.ones(len(y), bool) if mask is None else np.asarray(mask, bool)
    if m.sum() == 0:
        return {"n": 0}
    codes, uniq = pd.factorize(np.asarray(groups)[m])
    g = len(uniq)
    if g < min_groups:
        return {"n": int(m.sum()), "n_blocks": g, "note": "too few day-blocks for a bootstrap CI"}
    n_g = np.bincount(codes, minlength=g).astype(float)
    s_abs = np.bincount(codes, weights=np.abs(y[m] - q50[m]), minlength=g)
    s_vio = np.bincount(codes, weights=(q05[m] > y[m]).astype(float), minlength=g)
    idx = np.random.default_rng(seed).integers(0, g, size=(n_boot, g))
    tot = n_g[idx].sum(axis=1)
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    out: dict[str, Any] = {"n": int(m.sum()), "n_blocks": g}
    for name, sums in (("mae_a", s_abs), ("overestimation_frequency", s_vio)):
        boot = sums[idx].sum(axis=1) / tot
        out[name] = {"estimate": float(sums.sum() / n_g.sum()),
                     "ci_low": float(np.percentile(boot, lo)), "ci_high": float(np.percentile(boot, hi))}
    return out


SLICE_NOTE = ("Slice thresholds are fixed illustrative values (not derived from standards); "
              "slices with n < 30 should not be interpreted.")


def slice_analysis(frame: pd.DataFrame, y: np.ndarray, preds: Mapping[str, np.ndarray],
                   slices: Mapping[str, Callable[[pd.DataFrame], pd.Series]] | None = None
                   ) -> dict[str, dict[str, Any]]:
    """Evaluate the same predictions on weather-condition slices of ``frame``
    (needs wind_speed_ms, crosswind_ms, air_temp_c, sin_solar_altitude, solar_radiation)."""
    y = np.asarray(y, float)
    res: dict[str, dict[str, Any]] = {}
    for name, fn in (slices or DEFAULT_SLICES).items():
        mask = np.asarray(fn(frame.reset_index(drop=True)), dtype=bool)
        if mask.sum() == 0:
            res[name] = {"n": 0}
            continue
        res[name] = evaluate_predictions(y[mask], {k: np.asarray(v)[mask] for k, v in preds.items()})
    return res


# ------------------------------------------------------------------ report helpers
def results_to_markdown(title: str, table: Mapping[str, Mapping[str, Any]],
                        columns: Sequence[str]) -> str:
    """Small markdown table: rows = model/slice names, columns = metric keys."""
    lines = [f"### {title}", "", "| name | " + " | ".join(columns) + " |",
             "|---|" + "---|" * len(columns)]
    for name, m in table.items():
        cells = []
        for c in columns:
            v = m.get(c, "")
            cells.append(f"{v:.4g}" if isinstance(v, float) else str(v))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"