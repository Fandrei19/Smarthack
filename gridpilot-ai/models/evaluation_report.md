# Evaluation report (2026-11-07T14:53:36Z)

**Labels from LineRate CIGRE601:** True


> q05/q50/q95 quantify only the approximation error of an ML model fitted to a deterministic, synthetic physics function (CIGRE TB 601 via LineRate) of observed weather. They do NOT include weather-forecast error (the model is trained on historical weather, used with a forecast), conductor-parameter/sag/ageing uncertainty, CIGRE model-form error, or any real grid measurement. Real-world uncertainty is therefore larger than the reported intervals.

Split: `{"train": {"rows": 73632, "start": "2022-01-01 00:00:00+00:00", "end": "2024-10-19 15:00:00+00:00"}, "val": {"rows": 15708, "start": "2024-10-20 16:00:00+00:00", "end": "2025-05-26 19:00:00+00:00"}, "test": {"rows": 15708, "start": "2025-05-27 20:00:00+00:00", "end": "2025-12-31 23:00:00+00:00"}, "gap_hours": 24.0}`

### Test set - models

| name | n | mae_a | rmse_a | pinball_q05 | pinball_q50 | pinball_q95 | lower_bound_coverage | overestimation_frequency | overestimation_count | max_overestimation_a | interval_90_coverage |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_dummy_median(+const q05/q95) | 15708 | 150.3 | 191.8 | 22 | 75.16 | 16.46 | 0.9386 | 0.06137 | 964 | 446.8 | 0.923 |
| baseline_linear(+conformal q05) | 15708 | 42.03 | 59.36 | 9.062 | 21.01 |  | 0.9538 | 0.04622 | 726 | 376.3 |  |
| hgb_point(squared_error) | 15708 | 4.449 | 6.137 |  | 2.224 |  |  |  |  |  |  |
| hgb_quantile_raw_q05 | 15708 | 5.336 | 7.84 | 0.7491 | 2.668 | 1.628 | 0.9286 | 0.07143 | 1122 | 60.95 | 0.8659 |
| hgb_quantile_calibrated(DEPLOYED) | 15708 | 4.45 | 6.793 | 0.7198 | 2.225 | 1.613 | 0.9605 | 0.03947 | 620 | 60.28 | 0.9176 |


### Test set - deployed model by weather slice

| name | n | mae_a | rmse_a | pinball_q05 | pinball_q50 | pinball_q95 | lower_bound_coverage | overestimation_frequency | overestimation_count | max_overestimation_a | interval_90_coverage |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 15708 | 4.45 | 6.793 | 0.7198 | 2.225 | 1.613 | 0.9605 | 0.03947 | 620 | 60.28 | 0.9176 |
| low_wind (<1.0 m/s) | 471 | 9.828 | 16.3 | 1.498 | 4.914 | 7.473 | 0.9342 | 0.06582 | 31 | 39.1 | 0.9236 |
| low_crosswind (<0.5 m/s) | 1722 | 6.412 | 10.86 | 1.153 | 3.206 | 5.546 | 0.9489 | 0.0511 | 88 | 60.28 | 0.9326 |
| high_temp (>=30 C) | 963 | 7.815 | 13.09 | 1.951 | 3.908 | 6.114 | 0.9045 | 0.09553 | 92 | 60.28 | 0.8941 |
| low_wind & high_temp | 35 | 21.32 | 32.18 | 4.589 | 10.66 | 16.72 | 0.7714 | 0.2286 | 8 | 39.1 | 0.7714 |
| daytime (sun up) | 8020 | 5.797 | 8.408 | 0.9482 | 2.898 | 2.497 | 0.9397 | 0.06035 | 484 | 60.28 | 0.9009 |
| night | 7688 | 3.045 | 4.532 | 0.4815 | 1.523 | 0.6907 | 0.9823 | 0.01769 | 136 | 13.9 | 0.935 |
| strong_sun (GHI>=600 W/m2) | 1802 | 6.562 | 10.96 | 1.428 | 3.281 | 5.559 | 0.939 | 0.06104 | 110 | 60.28 | 0.9001 |
| high_wind (>=8 m/s) | 72 | 5.266 | 7.319 | 0.7057 | 2.633 | 0.6846 | 0.9306 | 0.06944 | 5 | 1.946 | 0.8472 |


_Slice thresholds are fixed illustrative values (not derived from standards); slices with n < 30 should not be interpreted._

### Static vs DLR lower bound (test)

`{"frac_q05_above_static": 0.9945887445887446, "mean_q05_minus_static_a": 552.1261399691772, "frac_static_above_reference": 0.0035014005602240898}`

### Rolling-origin backtest

### folds

| name | n_test | mae_a | overestimation_freq_calibrated | overestimation_freq_raw_q05 | max_overestimation_a |
|---|---|---|---|---|---|
| 0 | 13077 | 5.206 | 0.02569 | 0.08052 | 28.59 |
| 1 | 13077 | 4.676 | 0.03885 | 0.08389 | 57.92 |
| 2 | 13077 | 4.718 | 0.03082 | 0.06683 | 32.39 |
| 3 | 13077 | 4.516 | 0.04076 | 0.07219 | 60.3 |
