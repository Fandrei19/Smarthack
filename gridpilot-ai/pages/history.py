import json
import altair as alt
import pandas as pd
import streamlit as st
from src.history_service import history_files, read_history, filter_history
from src.live_pipeline import ROOT

st.title("Weather History")
st.caption("Explore historical weather and saved forecast runs. All timestamps are UTC.")
files = history_files()
if not files:
    st.info("No weather files available yet. Run the data pipeline or refresh the live forecast.")
    st.stop()
kind = st.radio("Dataset type", ["Historical weather", "Saved forecasts"], horizontal=True)
options = []
for path in files:
    try:
        header = pd.read_csv(path, nrows=1)
        source = str(header.get("source", pd.Series(dtype=str)).iloc[0]) if len(header) else ""
        forecast = "FORECAST" in source.upper()
        if forecast == (kind == "Saved forecasts"):
            options.append(path)
    except (OSError, ValueError, IndexError):
        continue
if not options:
    st.info("No datasets in this category yet. Successful live refreshes are archived automatically.")
    st.stop()
selected = st.selectbox("Dataset / forecast run", options,
    format_func=lambda p: str(p.relative_to(ROOT / "data")))
try:
    frame = read_history(selected)
except (OSError, ValueError) as exc:
    st.error(f"Cannot read this dataset: {exc}")
    st.stop()
if frame.empty:
    st.warning("This file contains no valid timestamps.")
    st.stop()
sources = ", ".join(sorted(frame.source.dropna().astype(str).unique()))
st.caption(f"Source: {sources} · {len(frame):,} rows · {frame.timestamp_utc.min():%Y-%m-%d} to {frame.timestamp_utc.max():%Y-%m-%d}")
if kind == "Saved forecasts":
    st.info("These are saved predictions of the weather, including future hours. They are not measurements of what actually happened.")
    metadata = selected.parent / "metadata.json"
    if metadata.exists():
        try:
            st.caption("Retrieved at: " + json.loads(metadata.read_text(encoding="utf-8"))["retrieved_at_utc"])
        except (OSError, ValueError, KeyError):
            st.caption("Retrieval time unavailable.")
else:
    st.info("Historical NASA POWER / ERA5 weather is a gridded estimate, not telemetry measured on the conductor.")
first, last = frame.timestamp_utc.min().date(), frame.timestamp_utc.max().date()
a, b = st.columns(2)
dates = a.date_input("Date range (UTC)", value=(first, min(first + pd.Timedelta(days=6), last)),
                     min_value=first, max_value=last)
segments = b.multiselect("Segments", sorted(frame.segment_id.unique()), default=sorted(frame.segment_id.unique()))
if len(dates) != 2:
    st.info("Select both a start and an end date.")
    st.stop()
filtered = filter_history(frame, dates[0], dates[1], segments)
if filtered.empty:
    st.info("No rows match your selection.")
    st.stop()
c1,c2,c3 = st.columns(3)
c1.metric("Records", f"{len(filtered):,}")
c2.metric("Segments", filtered.segment_id.nunique())
c3.metric("Rows flagged for review", int(filtered.data_quality.fillna("missing").ne("ok").sum()))
metrics = {"Air temperature (°C)": "air_temp_c", "Wind speed (m/s)": "wind_speed_ms",
           "Wind direction (°)": "wind_direction_deg", "Solar radiation (W/m²)": "solar_radiation"}
metric = st.selectbox("Weather variable", list(metrics))
column = metrics[metric]
plot = filtered
if len(plot) > 5000:
    plot = (filtered.set_index("timestamp_utc").groupby("segment_id")[[column]]
            .resample("D").mean().reset_index())
    plot["data_quality"] = "daily_average"
    st.caption("Large selection: the chart shows daily averages. The table and CSV retain hourly data.")
chart = alt.Chart(plot).mark_line().encode(
    x=alt.X("timestamp_utc:T", title="Time (UTC)", scale=alt.Scale(type="utc")),
    y=alt.Y(f"{column}:Q", title=metric, scale=alt.Scale(zero=False)),
    color="segment_id:N", tooltip=["timestamp_utc:T", "segment_id:N", f"{column}:Q", "data_quality:N"])
st.altair_chart(chart.properties(height=340).interactive(), use_container_width=True)
st.dataframe(filtered, hide_index=True, use_container_width=True)
st.download_button("Download selected data", filtered.to_csv(index=False), "weather_history_selection.csv", "text/csv")
