# GridPilot AI – Ampacity model card (Participant 2)

**SIMULATION / DECISION-SUPPORT PROTOTYPE.** Simulated line in Dobrogea, Romania. The model was
trained on physics-generated labels, **never on real PPC / Transelectrica grid measurements**.

## Physics reference (src/physics_engine.py)
* Library: Statnett **LineRate**, class `linerate.Cigre601` (CIGRE TB 601), steady-state ampacity via
  `compute_steady_state_ampacity`. Interface taken from the official docs
  (<https://statnett.github.io/linerate/api/types.html>, `/api/model.html`, `/auto_examples/plot_cigre.html`).
* **Conductor: CIGRE TB 601 "Example B" values exactly as in the official LineRate example** (docs do not name the type):

| parameter | value | unit |
|---|---|---|
| core_diameter | 10.4e-3 | m |
| conductor_diameter | 28.1e-3 | m |
| outer_layer_strand_diameter | 2.2e-3 | m |
| emissivity / solar_absorptivity | 0.9 / 0.9 | – |
| temperature1 / temperature2 | 25 / 75 | °C |
| resistance_at_temperature1 / 2 | 7.283e-5 / 8.688e-5 | Ω/m |
| aluminium_cross_section_area | NaN (no core magnetisation loss) | m² |
| constant_magnetic_effect | 1 | – |
| current_density_proportional_magnetic_effect | 0 | m²/A |
| max_magnetic_core_relative_resistance_increase | 1 | – |

### Our assumptions (NOT from LineRate/CIGRE – team must confirm; all in `config/conductor_config.json`)
| item | value | why |
|---|---|---|
| max conductor temperature | 75 °C | = upper resistance reference temp, avoids resistance extrapolation |
| ground albedo | 0.2 | "soil, grass, crops" value in LineRate docstring (example uses 0.15) |
| span geometry | 2 towers ±200 m along segment azimuth, simplex, zero inclination | LineRate derives azimuth from tower coordinates |
| solar → clearness ratio | N_s ∈ [0,1] that best reproduces observed GHI with LineRate's CIGRE601 direct+diffuse functions | LineRate wants N_s, weather.csv has GHI; no documented conversion |
| static rating weather | 35 °C, 0.61 m/s at 90° to line, N_s = 1, solstice noon | **PROPOSED placeholder**, not an official convention |
| solar position | NOAA/Spencer series, ~0.5° | LineRate computes its own internally |

## Model (src/ml_model.py)
* Features (weather + segment geometry only, units in README of module): air temp, wind speed, angle of attack,
  crosswind, parallel wind, GHI, sin(solar altitude), hour/day-of-year sin/cos, line azimuth.
* Baselines: `DummyRegressor` (median; constant 5 %/95 % quantiles) and `LinearRegression` (+ conformal lower bound).
* Tree model: `HistGradientBoostingRegressor` squared error (point) and `loss="quantile"` for q05/q50/q95.
* **Deployed q05 = HGB q05 minus a split-conformal shift** computed on the validation block (later in time than train),
  aiming for ≤ 5 % overestimation. q05 ≤ q50 ≤ q95 is enforced by sorting (can only lower q05).
* Split: chronological 70/15/15 by unique timestamp with a 24 h gap; plus expanding-window rolling-origin backtest.
* Metrics: MAE, RMSE, pinball (q05/q50/q95), lower-bound coverage, **overestimation frequency / count / max (A)**,
  slices (low wind, low crosswind, high temp, night/day, strong sun, high wind).
* `uncertainty_ratio = (q95 − q05) / q50`.

## What the uncertainty is and is not
* **Is:** ML approximation error w.r.t. a deterministic physics function, calibrated on a later time block.
* **Is not:** weather-forecast error (trained on historical NASA POWER weather, used with Open-Meteo forecasts – different
  source, resolution, wind height, radiation definition); conductor/sag/ageing/model-form error (labels contain none);
  real measurements. **Real uncertainty is larger than the reported intervals.** The conformal guarantee needs
  exchangeability and is only approximate under seasonal drift – see the backtest folds.

## Status of verification (read before trusting numbers)
* LineRate could **not be installed** in the authoring sandbox (no network): `calculate_ampacity` follows the documented
  API but has **never been executed**. Six LineRate tests (`tests/test_ml_model.py`) skip without it.
* ML code paths were exercised only with a **non-physical mock label function** on synthetic weather. No real metrics exist yet.
* Regenerate `models/ampacity_model.joblib` and `models/evaluation_report.*` after running the training CLI with real inputs.
  Check `labels_from_linerate_cigre601: true` in the report.