# Live forecasts and weather history

Run the dashboard from gridpilot-ai using the Python environment with the trained model dependencies:

```sh
python -m streamlit run app.py
```

The sidebar refreshes Open-Meteo weather, local ML predictions and agent decisions on first opening, then every 15 minutes while a Streamlit session is active. Refresh now triggers an immediate update. A failed refresh retains the previous successful files and displays a warning. Expired forecast periods are excluded from the monitoring pages. Weather values are forecasts, not sensor measurements; all times are UTC.

Weather History is under Analytics. Choose Historical weather (available NASA POWER / ERA5 files) or Saved forecasts, then select a dataset, inclusive date range, segments and weather variable. Download selected data exports the filtered hourly rows. Large chart selections use daily averages, while table/export retain full resolution.

Each successful refresh saves weather, predictions, decisions and retrieval metadata in data/archive/<UTC run ID>/. Archives stay local and are not pushed to GitHub. Saved forecasts remain forecasts even after their valid time has passed. No historical observations are inferred from them. Archives accumulate over time; retain or remove old run folders according to local storage needs.

Historical CSV files must use the agreed eight-column weather schema. No API calls are needed to browse history already on disk. Current CSV locations remain unchanged for compatibility with other components.
