"""Physics engine: CIGRE TB 601 steady-state ampacity via Statnett LineRate.

SIMULATION ONLY. All conductor / span / static-rating parameters come from
``config/conductor_config.json`` and are demonstrative. No real PPC grid data.

Units (everything in this module):
    temperatures  degC          wind speed      m/s
    wind direction  degrees clockwise from north (meteorological, as in weather.csv)
    solar_radiation W/m^2 (global horizontal irradiance)
    ampacity      A             lengths / altitude  m
    timestamps    UTC (tz-naive input is *assumed* UTC)

LineRate interface used (verified against the official docs, NOT executed in the
authoring sandbox because LineRate could not be installed there):
    linerate.Conductor(core_diameter=..., conductor_diameter=..., ...)   # types.html
    linerate.Tower(latitude=, longitude=, altitude=)                      # types.html
    linerate.Span(conductor=, start_tower=, end_tower=, num_conductors=)  # types.html
    linerate.Weather(air_temperature=, wind_direction=<rad, east of north>,
                     wind_speed=, ground_albedo=, clearness_ratio=)       # types.html
    linerate.Cigre601(span, weather, time)                                # model.html
    model.compute_steady_state_ampacity(max_conductor_temperature,
                                        min_ampacity=, max_ampacity=, tolerance=)
    linerate.equations.cigre601.solar_heating.compute_direct_solar_radiation(
        sin_solar_altitude, clearness_ratio, height_above_sea_level)
    linerate.equations.cigre601.solar_heating.compute_diffuse_sky_radiation(
        direct_solar_radiation, sin_solar_altitude)
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_CONDUCTOR_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "conductor_config.json"
EARTH_RADIUS_M = 6_371_000.0

REQUIRED_WEATHER_KEYS = (
    "timestamp_utc", "air_temp_c", "wind_speed_ms", "wind_direction_deg", "solar_radiation",
    "latitude", "longitude", "azimuth_deg",
)

AmpacityFn = Callable[[Mapping[str, Any], Mapping[str, Any]], float]


class PhysicsError(RuntimeError):
    """A single physical calculation failed (bad input, no solution, saturation)."""


class LineRateNotAvailableError(ImportError):
    """LineRate is not installed / its documented interface is missing.

    Deliberately NOT a PhysicsError: it is an environment problem and must abort
    label generation instead of being logged once per row.
    """


# --------------------------------------------------------------------------- config
def load_conductor_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the conductor/operating configuration JSON."""
    p = Path(path) if path else DEFAULT_CONDUCTOR_CONFIG_PATH
    try:
        with open(p, encoding="utf-8") as fh:
            cfg = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read conductor config {p}: {exc}") from exc
    for section in ("conductor", "span", "operating", "static_rating"):
        if section not in cfg or not isinstance(cfg[section], dict):
            raise ValueError(f"Conductor config {p} is missing section '{section}'")
    return cfg


def load_segments(path: str | Path) -> list[dict[str, Any]]:
    """Load segment geometry: list (or {"segments": [...]}) of dicts with
    segment_id, latitude [deg], longitude [deg], azimuth_deg [deg from north],
    optional altitude_m [m]."""
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read segments file {path}: {exc}") from exc
    segs = raw.get("segments") if isinstance(raw, dict) else raw
    if not isinstance(segs, list) or not segs:
        raise ValueError(f"{path}: expected a non-empty list of segments")
    for s in segs:
        missing = [k for k in ("segment_id", "latitude", "longitude", "azimuth_deg") if k not in s]
        if missing:
            raise ValueError(f"{path}: segment {s.get('segment_id')} missing keys {missing}")
    return segs


# ------------------------------------------------------------------ solar geometry
def solar_sin_altitude(timestamps: pd.Series | pd.DatetimeIndex, latitude_deg: float,
                       longitude_deg: float) -> np.ndarray:
    """Sine of the solar altitude angle (NOAA / Spencer-1971 Fourier-series
    approximation, accuracy ~0.5 deg - adequate for a clearness-ratio estimate).

    This is standard astronomy, NOT a LineRate API. LineRate computes its own
    solar position internally; small differences are expected.
    """
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    n = ts.dayofyear.to_numpy(dtype=float)
    hour = ts.hour.to_numpy(dtype=float) + ts.minute.to_numpy() / 60.0 + ts.second.to_numpy() / 3600.0
    g = 2.0 * np.pi / 365.0 * (n - 1.0 + (hour - 12.0) / 24.0)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                       - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))  # minutes
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))  # rad
    tst = hour * 60.0 + eqtime + 4.0 * longitude_deg  # true solar time, minutes (UTC => tz 0)
    ha = np.radians(tst / 4.0 - 180.0)
    lat = math.radians(latitude_deg)
    return np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)


# --------------------------------------------------------------- LineRate plumbing
def _import_linerate():
    try:
        import linerate  # noqa: WPS433 (lazy import: module stays importable without it)
        from linerate.equations.cigre601 import solar_heating
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise LineRateNotAvailableError(
            "LineRate is not installed or its documented modules are missing "
            "(`pip install linerate`, Python>=3.10). Physics labels cannot be generated."
        ) from exc
    for name in ("Conductor", "Tower", "Span", "Weather", "Cigre601"):
        if not hasattr(linerate, name):
            raise LineRateNotAvailableError(
                f"Installed linerate has no '{name}'. Interface differs from the documented one "
                "(see module docstring) - check the installed version's docs."
            )
    return linerate, solar_heating


def _conductor_kwargs(conductor_cfg: Mapping[str, Any]) -> dict[str, Any]:
    kw = {k: v for k, v in conductor_cfg.items() if not k.startswith("_")}
    if kw.get("aluminium_cross_section_area") is None:
        kw["aluminium_cross_section_area"] = float("nan")  # as in the official example
    return kw


def _build_towers(lat: float, lon: float, azimuth_deg: float, length_m: float,
                  altitude_m: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Two towers +-length/2 along `azimuth_deg` around the midpoint (flat-earth, zero inclination)."""
    half = length_m / 2.0
    az = math.radians(azimuth_deg)
    d_north, d_east = half * math.cos(az), half * math.sin(az)
    dlat = math.degrees(d_north / EARTH_RADIUS_M)
    dlon = math.degrees(d_east / (EARTH_RADIUS_M * math.cos(math.radians(lat))))
    return (lat - dlat, lon - dlon, altitude_m), (lat + dlat, lon + dlon, altitude_m)


def estimate_clearness_ratio(ghi_wm2: float, sin_alt: float, altitude_m: float,
                             solar_heating_module: Any, cap: float = 1.0, n_grid: int = 201) -> float:
    """Map observed global horizontal irradiance to LineRate's ``clearness_ratio``.

    LineRate needs a clearness ratio N_s (it scales the CIGRE601 direct-beam term only,
    per the solar_heating docs), not an irradiance. ASSUMPTION (ours, not from the docs):
    pick N_s in [0, cap] such that the CIGRE601 model's horizontal global irradiance
    ``I_B*sin(H_s) + I_d`` (using LineRate's documented direct/diffuse functions) best
    matches the observed GHI. Known limitation: the CIGRE601 diffuse term has a floor
    (I_d does not vanish when I_B -> 0), so overcast skies map to N_s = 0 and the model
    may still over-estimate heating -> conservative (lower) ampacity.
    Night (sin_alt <= 0) returns 0.0.
    """
    if not (math.isfinite(ghi_wm2) and math.isfinite(sin_alt)) or sin_alt <= 0.0 or ghi_wm2 <= 0.0:
        return 0.0
    ns = np.linspace(0.0, cap, n_grid)
    i_b = solar_heating_module.compute_direct_solar_radiation(sin_alt, ns, altitude_m)
    i_d = solar_heating_module.compute_diffuse_sky_radiation(i_b, sin_alt)
    ghi_model = i_b * sin_alt + i_d
    return float(ns[int(np.argmin(np.abs(ghi_model - ghi_wm2)))])


# ------------------------------------------------------------------- main function
def _check_inputs(w: Mapping[str, Any]) -> None:
    missing = [k for k in REQUIRED_WEATHER_KEYS if k not in w]
    if missing:
        raise PhysicsError(f"missing inputs: {missing}")
    vals = {k: w[k] for k in REQUIRED_WEATHER_KEYS if k != "timestamp_utc"}
    for k, v in vals.items():
        try:
            if not math.isfinite(float(v)):
                raise PhysicsError(f"non-finite input {k}={v}")
        except (TypeError, ValueError) as exc:
            raise PhysicsError(f"non-numeric input {k}={v!r}") from exc
    if not -60.0 <= float(w["air_temp_c"]) <= 60.0:
        raise PhysicsError(f"air_temp_c out of plausible range: {w['air_temp_c']}")
    if not 0.0 <= float(w["wind_speed_ms"]) <= 80.0:
        raise PhysicsError(f"wind_speed_ms out of plausible range: {w['wind_speed_ms']}")
    if not 0.0 <= float(w["wind_direction_deg"]) <= 360.0:
        raise PhysicsError(f"wind_direction_deg must be in [0, 360]: {w['wind_direction_deg']}")
    if not 0.0 <= float(w["solar_radiation"]) <= 1500.0:
        raise PhysicsError(f"solar_radiation out of range [0,1500] W/m2: {w['solar_radiation']}")


def calculate_ampacity(weather_features: Mapping[str, Any], conductor_config: Mapping[str, Any]) -> float:
    """Steady-state ampacity [A] of one simulated span for one weather sample (CIGRE TB 601).

    Args:
        weather_features: mapping with ``timestamp_utc``, ``air_temp_c`` [degC],
            ``wind_speed_ms`` [m/s], ``wind_direction_deg`` [deg, meteorological],
            ``solar_radiation`` [W/m2 GHI], ``latitude``/``longitude`` [deg],
            ``azimuth_deg`` [deg from north]; optional ``altitude_m`` [m] and an explicit
            ``clearness_ratio`` [-] (overrides the GHI-based estimate).
        conductor_config: dict from :func:`load_conductor_config`.

    Returns:
        Max current [A] keeping the conductor <= ``operating.max_conductor_temperature_c``.

    Raises:
        PhysicsError: invalid inputs, non-finite result, no feasible current, or result
            saturated at the search upper bound.
        LineRateNotAvailableError: LineRate missing / interface mismatch.

    Note: wind direction is passed to LineRate as radians; CIGRE angle-of-attack effects
    are symmetric under a 180 deg flip, so the from/to convention does not matter.
    """
    _check_inputs(weather_features)
    linerate, solar_heating = _import_linerate()
    op, span_cfg = conductor_config["operating"], conductor_config["span"]
    w = weather_features

    ts = pd.Timestamp(w["timestamp_utc"])
    ts = ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo is not None else ts
    time = np.datetime64(ts.to_datetime64(), "s")

    lat, lon, az = float(w["latitude"]), float(w["longitude"]), float(w["azimuth_deg"])
    alt = float(w.get("altitude_m", span_cfg.get("default_altitude_m", 0.0)))

    if w.get("clearness_ratio") is not None:
        ns = float(w["clearness_ratio"])
    else:
        sin_alt = float(solar_sin_altitude(pd.DatetimeIndex([ts]), lat, lon)[0])
        ns = estimate_clearness_ratio(float(w["solar_radiation"]), sin_alt, alt, solar_heating,
                                      cap=float(op.get("clearness_ratio_max", 1.0)))
    try:
        conductor = linerate.Conductor(**_conductor_kwargs(conductor_config["conductor"]))
        (la0, lo0, a0), (la1, lo1, a1) = _build_towers(lat, lon, az, float(span_cfg["length_m"]), alt)
        span = linerate.Span(
            conductor=conductor,
            start_tower=linerate.Tower(latitude=la0, longitude=lo0, altitude=a0),
            end_tower=linerate.Tower(latitude=la1, longitude=lo1, altitude=a1),
            num_conductors=int(span_cfg.get("num_conductors", 1)),
        )
        weather = linerate.Weather(
            air_temperature=float(w["air_temp_c"]),
            wind_direction=math.radians(float(w["wind_direction_deg"])),
            wind_speed=float(w["wind_speed_ms"]),
            ground_albedo=float(op["ground_albedo"]),
            clearness_ratio=ns,
        )
        model = linerate.Cigre601(span, weather, time)
        amp = float(model.compute_steady_state_ampacity(
            float(op["max_conductor_temperature_c"]),
            min_ampacity=float(op.get("ampacity_search_min_a", 0.0)),
            max_ampacity=float(op.get("ampacity_search_max_a", 8000.0)),
            tolerance=float(op.get("ampacity_tolerance_a", 1.0)),
        ))
    except PhysicsError:
        raise
    except (TypeError, AttributeError) as exc:  # interface mismatch with installed version
        raise LineRateNotAvailableError(f"LineRate interface mismatch: {exc!r}") from exc
    except Exception as exc:  # numerical failure inside LineRate for this sample
        raise PhysicsError(f"LineRate failed: {exc!r}") from exc

    if not math.isfinite(amp):
        raise PhysicsError(f"non-finite ampacity {amp}")
    tol = float(op.get("ampacity_tolerance_a", 1.0))
    if amp <= tol:
        raise PhysicsError(f"no feasible current (ampacity {amp:.2f} A)")
    if amp >= float(op.get("ampacity_search_max_a", 8000.0)) - tol:
        raise PhysicsError(f"ampacity saturated at search bound ({amp:.1f} A)")
    return amp


def compute_static_rating(segment: Mapping[str, Any], conductor_config: Mapping[str, Any],
                          ampacity_fn: AmpacityFn = calculate_ampacity) -> float:
    """Static rating [A] of a segment under the PROPOSED static weather assumptions
    in ``conductor_config['static_rating']`` (team must confirm; see config notes)."""
    sr = conductor_config["static_rating"]
    az = float(segment["azimuth_deg"])
    feats = {
        "timestamp_utc": sr["timestamp_utc"],  # tz handling happens in calculate_ampacity
        "air_temp_c": sr["air_temp_c"],
        "wind_speed_ms": sr["wind_speed_ms"],
        "wind_direction_deg": (az + float(sr["wind_angle_of_attack_deg"])) % 360.0,
        "solar_radiation": 0.0,  # unused: clearness_ratio given explicitly
        "clearness_ratio": sr["clearness_ratio"],
        "latitude": segment["latitude"], "longitude": segment["longitude"], "azimuth_deg": az,
        "altitude_m": segment.get("altitude_m", conductor_config["span"].get("default_altitude_m", 0.0)),
    }
    return float(ampacity_fn(feats, conductor_config))


# ------------------------------------------------------------------ label generation
def generate_labels(weather: pd.DataFrame, segments: Sequence[Mapping[str, Any]],
                    conductor_config: Mapping[str, Any], ampacity_fn: AmpacityFn = calculate_ampacity,
                    max_failure_fraction: float = 0.2, log_every: int = 20000) -> pd.DataFrame:
    """Add ``target_ampacity_a`` [A] and ``label_status`` ('ok' or failure reason) per row.

    Failed rows keep NaN targets and are logged; they are NOT silently dropped here.
    Raises RuntimeError if more than ``max_failure_fraction`` of rows fail.
    ``ampacity_fn`` is injectable so the pipeline can be tested without LineRate.
    """
    geo = {str(s["segment_id"]): s for s in segments}
    out = weather.copy()
    targets = np.full(len(out), np.nan)
    status = np.full(len(out), "ok", dtype=object)
    ts_col, seg_col = out["timestamp_utc"].to_numpy(), out["segment_id"].astype(str).to_numpy()
    cols = {c: out[c].to_numpy() for c in ("air_temp_c", "wind_speed_ms", "wind_direction_deg", "solar_radiation")}
    for i in range(len(out)):
        seg = geo.get(seg_col[i])
        if seg is None:
            status[i] = "unknown_segment"
            continue
        feats = {"timestamp_utc": ts_col[i], **{k: v[i] for k, v in cols.items()},
                 "latitude": seg["latitude"], "longitude": seg["longitude"], "azimuth_deg": seg["azimuth_deg"],
                 "altitude_m": seg.get("altitude_m", conductor_config["span"].get("default_altitude_m", 0.0))}
        try:
            targets[i] = ampacity_fn(feats, conductor_config)
        except PhysicsError as exc:
            status[i] = f"physics_error: {exc}"[:200]
        if log_every and (i + 1) % log_every == 0:
            logger.info("labels: %d/%d rows processed", i + 1, len(out))
    out["target_ampacity_a"] = targets
    out["label_status"] = status
    n_bad = int((status != "ok").sum())
    if n_bad:
        logger.warning("label generation: %d/%d rows failed (%.2f%%)", n_bad, len(out), 100 * n_bad / len(out))
        for reason, cnt in pd.Series(status[status != "ok"]).str.slice(0, 60).value_counts().head(5).items():
            logger.warning("  %5d x %s", cnt, reason)
    if len(out) and n_bad / len(out) > max_failure_fraction:
        raise RuntimeError(f"{n_bad}/{len(out)} physical calculations failed (> {max_failure_fraction:.0%}); "
                           "check inputs / LineRate integration before training.")
    return out