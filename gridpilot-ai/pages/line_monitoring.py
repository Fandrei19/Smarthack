import altair as alt
import pandas as pd
import streamlit as st

from src.dashboard_service import get_dashboard_data
from src.data_processor import decision_for_hour, filter_forecast_window, line_rating_by_time

st.title("Line Monitoring")
st.caption("One transmission line, three segments: the line can carry only as much "
           "as its weakest segment. All times are UTC.")
st.caption("Ampacity is the estimated current limit. Q05 is a conservative estimate "
           "relative to the simulated physics reference.")

# load data
try:
    data = get_dashboard_data()
except FileNotFoundError as e:
    st.error(f"Missing file: {e.filename}")
    st.stop()
except ValueError as e:
    st.error(str(e))
    st.stop()

combined = data["combined"]
if combined.empty:
    st.warning("No weather/prediction data to display.")
    st.stop()


def fmt(value, unit="", decimals=0):
    return "Unavailable" if pd.isna(value) else f"{value:.{decimals}f}{unit}"


# filters: interval + the hour to inspect
f1, f2 = st.columns([1, 3])
hours = f1.selectbox("Interval", [6, 12, 24], index=2,
                     format_func=lambda h: f"Next {h}h")

window = filter_forecast_window(combined, hours)
line_df = line_rating_by_time(window)

times = sorted(window["timestamp"].drop_duplicates())

hour = f2.select_slider("Inspect hour (UTC)",
                        options=times,
                        value=times[0],
                        format_func=lambda t: f"{t:%b %d, %H:%M}")

snap = window[window["timestamp"] == hour].set_index("segment_id")
now = line_df[line_df["timestamp"] == hour].iloc[0]
static, rating, limiting = now["static_rating"], now["line_rating"], now["limiting_segment"]

# summary: the answer first
missing = list(snap.index[snap["q05"].isna()])

if missing:
    st.warning(f"At {hour:%H:%M} UTC the line rating cannot be computed: "
               f"no prediction for {', '.join(missing)}. Static rating: {fmt(static, ' A')}.")
else:
    pct = (rating - static) / static * 100
    msg = (f"At {hour:%H:%M} UTC the estimated dynamic limit is **{rating:.0f} A**, "
           f"{pct:+.1f}% vs. the static rating ({static:.0f} A). "
           f"Limiting segment: **{limiting}**.")
    (st.success if rating >= static else st.warning)(msg)

decision = decision_for_hour(data["agent"], hour)
if decision:
    show = st.warning if decision["requires_human_review"] else st.info
    show(f"Agent recommendation: **{decision['action']}** · {decision['reason']}")


now_rule = alt.Chart(pd.DataFrame({"timestamp": [hour]})).mark_rule(color="#333").encode(x="timestamp:T")

# 1. which segment limits the line
st.subheader("1 · Which segment limits the line?")
bars_df = snap.reset_index()[["segment_id", "q05"]].dropna()

bars_df["role"] = (bars_df["segment_id"] == limiting).map(
    {True: "Limiting segment",
     False: "Other segments"})

bars = alt.Chart(bars_df).mark_bar().encode(
    y=alt.Y("segment_id:N", title=None),
    x=alt.X("q05:Q", title="Ampacity (A)"),

    color=alt.Color("role:N", title=None,
                    scale=alt.Scale(domain=["Limiting segment", "Other segments"],
                                    range=["#d62728", "#1f77b4"])),

    tooltip=["segment_id:N", alt.Tooltip("q05:Q", format=".0f")],
)

labels = bars.mark_text(align="left", dx=4).encode(text=alt.Text("q05:Q", format=".0f"), color=alt.value("#555"))

static_rule = alt.Chart(pd.DataFrame({"static": [static]})).mark_rule(strokeDash=[5, 5], color="#7f7f7f").encode(x="static:Q")

st.altair_chart((bars + labels + static_rule).properties(height=160), use_container_width=True)

st.caption("Dashed line = static rating. Segments whose bar passes it have spare capacity.")

# 2. how it changed
st.subheader("2 · How did the line rating change?")
base = alt.Chart(line_df).encode(x=alt.X("timestamp:T", title="Timestamp (UTC)"))
trend = base.mark_line(color="#bbb").encode(y=alt.Y("line_rating:Q", title="Ampacity (A)", scale=alt.Scale(zero=False)))

dots = base.mark_point(filled=True, size=70).encode(
    y="line_rating:Q",
    color=alt.Color("limiting_segment:N", title="Limiting segment"),
    tooltip=["timestamp:T", alt.Tooltip("line_rating:Q", format=".0f"),
             "limiting_segment:N"])

static_line = base.mark_line(strokeDash=[5, 5], color="#7f7f7f").encode(y="static_rating:Q")

st.altair_chart((trend + dots + static_line + now_rule).properties(height=280),use_container_width=True)

st.caption("Each dot is one hour, colored by the segment that limited the line. "
           "Dashed = static rating, vertical line = inspected hour. Gaps = missing predictions.")


# 3. why: weather
st.subheader("3 · What explains it? Weather")
seg_options = list(snap.index)
seg = st.selectbox("Segment",
                   seg_options,
                   index=seg_options.index(limiting) if limiting in seg_options else 0)
w = snap.loc[seg]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Ampacity", fmt(w["q05"], " A"), border=True)
m2.metric("Wind speed", fmt(w["wind_speed"], " m/s", 1), border=True)
m3.metric("Air temperature", fmt(w["temperature"], " °C", 1), border=True)
m4.metric("Solar radiation", fmt(w["solar_radiation"], " W/m²"), border=True)

seg_df = window[window["segment_id"] == seg]


def weather_chart(col, title):
    c = alt.Chart(seg_df).mark_line().encode(
        x=alt.X("timestamp:T", title=None),
        y=alt.Y(f"{col}:Q", title=title, scale=alt.Scale(zero=False)))

    return (c + now_rule).properties(height=110)


st.altair_chart(
    alt.vconcat(weather_chart("wind_speed", "Wind (m/s)"),
                weather_chart("temperature", "Temp (°C)"),
                weather_chart("solar_radiation", "Solar (W/m²)")
                ).resolve_scale(x="shared"),
    use_container_width=True)

st.caption("Less wind and more sun or heat cool the conductor less, so ampacity drops.")

# details, collapsed
with st.expander("Uncertainty range by segment"):
    st.caption("Shaded band = q05–q95. A wider band means a less certain estimate "
               "(uncertainty = (q95 − q05) / q50).")

    b = alt.Chart(window).encode(x=alt.X("timestamp:T", title="Timestamp (UTC)"),
                                 color=alt.Color("segment_id:N", title="Segment"))

    band = b.mark_area(opacity=0.15).encode(
        y=alt.Y("q05:Q", title="Ampacity (A)", scale=alt.Scale(zero=False)), y2="q95:Q")

    line = b.mark_line().encode(
        y="q05:Q",
        tooltip=["timestamp:T", "segment_id:N", "q05:Q", "q50:Q", "q95:Q",
                 alt.Tooltip("uncertainty:Q", format=".2f")])

    st.altair_chart((band + line).properties(height=300).interactive(),use_container_width=True)

with st.expander("Data table"):
    st.caption(f"{int(window['q05'].isna().sum())} missing predictions in the selected interval · "
               f"last timestamp: {combined['timestamp'].max():%Y-%m-%d %H:%M} UTC")

    cols = ["timestamp", "segment_id", "temperature", "wind_speed", "wind_direction",
            "solar_radiation", "static_rating", "q05", "q50", "q95",
            "extra_pct", "uncertainty", "weather_quality", "prediction_quality"]

    table = window[cols].sort_values(["timestamp", "segment_id"], ascending=[False, True])

    st.dataframe(
        table, hide_index=True, use_container_width=True,
        column_config={
            "timestamp": st.column_config.DatetimeColumn("Timestamp (UTC)",
                                                         format="YYYY-MM-DD HH:mm"),
            "extra_pct": st.column_config.NumberColumn("q05 vs. static (%)", format="%+.1f%%"),
            "uncertainty": st.column_config.NumberColumn(format="%.2f"),
        })

    st.download_button("Download CSV", table.to_csv(index=False),
                       file_name="line_monitoring.csv", mime="text/csv")
