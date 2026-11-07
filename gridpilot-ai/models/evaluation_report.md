# Evaluation report (2026-11-07T18:57:15Z)

**Labels from LineRate CIGRE601:** True


> q05/q50/q95 quantify only the approximation error of an ML model fitted to a deterministic, synthetic physics function (CIGRE TB 601 via LineRate) of observed weather. They do NOT include weather-forecast error (the model is trained on historical weather, used with a forecast), conductor-parameter/sag/ageing uncertainty, CIGRE model-form error, or any real grid measurement. Real-world uncertainty is therefore larger than the reported intervals.

Split: `{"train": {"rows": 52596, "start": "2022-01-01 00:00:00+00:00", "end": "2024-01-01 11:00:00+00:00"}, "val": {"rows": 26226, "start": "2024-01-02 12:00:00+00:00", "end": "2024-12-31 17:00:00+00:00"}, "test": {"rows": 26226, "start": "2025-01-01 18:00:00+00:00", "end": "2025-12-31 23:00:00+00:00"}, "gap_hours": 24.0}`

### Test set - models

| name | n | mae_a | rmse_a | pinball_q05 | pinball_q50 | pinball_q95 | lower_bound_coverage | overestimation_frequency | overestimation_count | max_overestimation_a | interval_90_coverage |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_dummy_median(+const q05/q95) | 26226 | 156.3 | 193.2 | 22.13 | 78.14 | 15.83 | 0.9499 | 0.0501 | 1314 | 458.5 | 0.9038 |
| baseline_linear(+conformal q05) | 26226 | 42.24 | 59.5 | 9.165 | 21.12 |  | 0.9492 | 0.05083 | 1333 | 384.7 |  |
| hgb_point(squared_error) | 26226 | 4.853 | 6.692 |  | 2.426 |  |  |  |  |  |  |
| hgb_quantile_raw_q05 | 26226 | 5.755 | 8.296 | 0.8592 | 2.878 | 1.376 | 0.9257 | 0.07428 | 1948 | 55.42 | 0.8522 |
| hgb_quantile_calibrated(DEPLOYED) | 26226 | 4.831 | 7.269 | 0.8096 | 2.415 | 1.347 | 0.9706 | 0.02944 | 772 | 52.06 | 0.922 |


### Test set - deployed model by weather slice

| name | n | mae_a | rmse_a | pinball_q05 | pinball_q50 | pinball_q95 | lower_bound_coverage | overestimation_frequency | overestimation_count | max_overestimation_a | interval_90_coverage |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 26226 | 4.831 | 7.269 | 0.8096 | 2.415 | 1.347 | 0.9706 | 0.02944 | 772 | 52.06 | 0.922 |
| low_wind (<1.0 m/s) | 746 | 8.918 | 14.98 | 1.192 | 4.459 | 6.208 | 0.933 | 0.06702 | 50 | 25.65 | 0.9236 |
| low_crosswind (<0.5 m/s) | 2860 | 6.902 | 11.08 | 1.033 | 3.451 | 4.617 | 0.9503 | 0.04965 | 142 | 25.65 | 0.9224 |
| high_temp (>=30 C) | 963 | 8.065 | 12.56 | 1.801 | 4.032 | 5.679 | 0.9242 | 0.0758 | 73 | 52.06 | 0.9169 |
| low_wind & high_temp | 35 | 16.96 | 22.76 | 2.094 | 8.478 | 14.49 | 0.8571 | 0.1429 | 5 | 15.09 | 0.8571 |
| moderate_severity (T>=28C & wind<2) | 250 | 9.201 | 13.76 | 1.514 | 4.601 | 10.16 | 0.908 | 0.092 | 23 | 23.24 | 0.908 |
| severe (T>=30C & wind<1) | 35 | 16.96 | 22.76 | 2.094 | 8.478 | 14.49 | 0.8571 | 0.1429 | 5 | 15.09 | 0.8571 |
| daytime (sun up) | 13178 | 6.533 | 9.055 | 0.983 | 3.266 | 2.118 | 0.9678 | 0.03217 | 424 | 52.06 | 0.9189 |
| night | 13048 | 3.112 | 4.837 | 0.6345 | 1.556 | 0.5682 | 0.9733 | 0.02667 | 348 | 16.07 | 0.9251 |
| strong_sun (GHI>=600 W/m2) | 2594 | 6.776 | 10.02 | 1.288 | 3.388 | 4.483 | 0.9584 | 0.04163 | 108 | 52.06 | 0.9233 |
| high_wind (>=8 m/s) | 146 | 6.231 | 8.076 | 0.8625 | 3.115 | 1.267 | 0.9932 | 0.006849 | 1 | 4.386 | 0.8425 |


_Slice thresholds are fixed illustrative values (not derived from standards); slices with n < 30 should not be interpreted._

### Day-block bootstrap 95% CI (test, deployed model)

| name | n | mae_a [95% CI] | overestimation_freq [95% CI] |
|---|---|---|---|
| all | 26226 | 4.831 [4.667, 5.039] | 0.02944 [0.02606, 0.03309] |
| daytime (sun up) | 13178 | 6.533 [6.29, 6.783] | 0.03217 [0.02644, 0.03906] |
| night | 13048 | 3.112 [2.982, 3.268] | 0.02667 [0.02319, 0.02994] |
| moderate_severity (T>=28C & wind<2) | 250 | 9.201 [7.454, 11.42] | 0.092 [0.05343, 0.1404] |


### Static vs DLR lower bound (test)

`{"frac_q05_above_static": 0.9966064211088234, "mean_q05_minus_static_a": 592.0852745109424, "frac_static_above_reference": 0.0020971554945473955}`

### Rolling-origin backtest

### folds

| name | n_test | mae_a | overestimation_freq_calibrated | overestimation_freq_raw_q05 | max_overestimation_a |
|---|---|---|---|---|---|
| 0 | 13077 | 5.175 | 0.01812 | 0.08029 | 36 |
| 1 | 13077 | 4.767 | 0.0377 | 0.08488 | 41.81 |
| 2 | 13077 | 4.669 | 0.02608 | 0.06485 | 20.68 |
| 3 | 13077 | 4.584 | 0.03755 | 0.07861 | 40.83 |
