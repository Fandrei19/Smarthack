import altair as alt
import pandas as pd
import streamlit as st
from pathlib import Path
import json
from src.live_monitor import live_monitor

from src.dashboard_service import get_dashboard_data
from src.data_processor import filter_window, line_rating_by_time

st.title("⚡ GridPilot AI — Dobrogea Live Overview")
live_monitor()
# Load real data from processed folder via dashboard service
try:
    data = get_dashboard_data()
except FileNotFoundError as e:
    st.error(f"Missing required data file: {e.filename}. Please run the pipeline or place files in data/processed/.")
    st.stop()
except ValueError as e:
    st.error(str(e))
    st.stop()

combined = data["combined"]

if combined.empty:
    st.warning("No processed prediction or weather data available in data/processed/.")
    st.stop()

latest = data["latest"]
agent = data["agent"]
ts = data["hour"]

if isinstance(agent, list):  
    agent = next((d for d in agent if pd.to_datetime(d.get("timestamp_utc"), utc=True) == ts), {})

def fmt(value, unit="", decimals=0):
    return "Unavailable" if pd.isna(value) else f"{value:.{decimals}f}{unit}"

# Decision dictionary mapping agent actions to clear, human-friendly explanations
DECISIONS = {
    "RECOMMEND_DYNAMIC_RATING": ("success", "🟢", "The line can carry more than usual",
                                 "Current weather conditions favor cable cooling, offering safe extra capacity."),
    "KEEP_STATIC": ("warning", "🟡", "Stick to the standard rating",
                    "Extra thermal headroom exists, but remains below the safety switching margin."),
    "OPERATOR_REVIEW": ("error", "🔴", "Operator review required",
                    "Thermal margins are tight; manual engineering oversight is recommended."),
    "FALLBACK_STATIC": ("warning", "🟡", "Fallback to static rating",
                    "Incomplete or stale sensor data requires conservative operation."),
}

def room_status(pct):
    if pd.isna(pct):
        return "⚪", "No data"
    if pct >= 25:
        return "🟢", "Plenty of room"
    if pct >= 5.3:
        return "🟡", "Limited room"
    return "🔴", "Almost no room"

amp = latest.get("predicted_ampacity", pd.Series(dtype=float))
static_vals = latest.get("static_rating", pd.Series([0]))
static = static_vals.min() if not static_vals.empty else 0

if amp.isna().any() or amp.empty:
    line_rating, bottleneck = float("nan"), None
else:
    line_rating, bottleneck = amp.min(), amp.idxmin()

segments = sorted(latest.index) if not latest.empty else []
gain = line_rating - static if not pd.isna(line_rating) else 0

st.caption(f"Forecast for {ts:%b %d, %Y - %H:%M} UTC. Weather is forecast data, while the model was trained "
           "on historical reanalysis; the q05–q95 range covers model error only, not forecast error.")

# 1. Main System Status & Agent Recommendation
if agent:
    action_key = agent.get("action", "KEEP_STATIC")
    kind, icon, headline, why = DECISIONS.get(action_key, ("info", "ℹ️", str(action_key), ""))
    with st.container(border=True):
        getattr(st, kind)(f"### {icon} {headline}\n{why}")
        if not pd.isna(gain) and static > 0:
            a, b = st.columns(2)
            a.metric("Current Dynamic Rating Capacity", fmt(line_rating, " A"),
                     f"{gain:+.0f} A ({gain / static * 100:+.0f}%) vs. Static Limit",
                     help="Governed by the weakest monitored line segment using Q05 conservative bounds.")
            b.metric("Standard Static Limit", fmt(static, " A"),
                     help="Fixed seasonal thermal threshold.")
        st.caption(f"AI Agent Rationale: {agent.get('reason', 'Standard operation active.')}")
        if agent.get("requires_human_review") or agent.get("human_review"):
            st.warning("👤 **Human operator confirmation required for this hour.**")
else:
    st.info("No active agent decision logged for this specific timestamp.")

# 2. Bottleneck Analysis
if bottleneck and bottleneck in latest.index:
    r = latest.loc[bottleneck]
    st.info(f"**Critical Bottleneck Identified:** Segment **{bottleneck}** is limiting the corridor capacity. "
            f"Conditions: Wind {fmt(r.get('wind_speed', 0), ' m/s', 1)}, "
            f"Temperature {fmt(r.get('temperature', 0), ' °C', 1)}, Solar Radiation {fmt(r.get('solar_radiation', 0), ' W/m²')}.")

# 3. Segment Breakdown Cards
st.subheader("Dobrogea Line Segments Health Status")
if segments:
    cols = st.columns(min(len(segments), 3))
    for i, s in enumerate(segments):
        col = cols[i % len(cols)]
        row = latest.loc[s]
        extra_pct = row.get("extra_pct", 0)
        dot, label = room_status(extra_pct)
        with col.container(border=True):
            st.markdown(f"#### {dot} Segment {s}")
            st.markdown(f"**Status:** {label}" + (" *(Bottleneck)*" if s == bottleneck else ""))
            st.metric("Ampacity Limit", fmt(row.get("predicted_ampacity"), " A"),
                      None if pd.isna(extra_pct) else f"{extra_pct:+.0f}% headroom")

# 4. 24-Hour Historical & Forecast Trend
st.subheader("24-Hour Corridor Thermal Envelope")
try:
    line_df = line_rating_by_time(filter_window(combined, 24))
    if not line_df.empty:
        base = alt.Chart(line_df).encode(x=alt.X("timestamp:T", title="UTC Time", scale=alt.Scale(type="utc")))

        rating = base.mark_area(opacity=0.2, color="#2ca02c").encode(
            y=alt.Y("line_rating:Q", title="Ampacity (Amps)", scale=alt.Scale(zero=False)),
            y2="static_rating:Q")

        rating_line = base.mark_line(color="#2ca02c", point=True).encode(
            y="line_rating:Q",
            tooltip=["timestamp:T", alt.Tooltip("line_rating:Q", format=".0f", title="Dynamic Rating (A)"),
                     alt.Tooltip("limiting_segment:N", title="Bottleneck Segment")])

        static_line = base.mark_line(strokeDash=[4, 4], color="#7f7f7f").encode(y="static_rating:Q")

        st.altair_chart((rating + rating_line + static_line).properties(height=280), use_container_width=True)
        st.caption("Green area = DLR extra capacity above the standard static rating (dashed gray line).")
    else:
        st.info("Insufficient time-series data for the past 24 hours.")
except Exception as e:
    st.info(f"Could not render trend chart: {e}")

# Navigation Links
st.divider()
nav1, nav2 = st.columns(2)
nav1.page_link("pages/line_monitoring.py", label="Open Live Telemetry Monitor", icon=":material/monitoring:")
nav2.page_link("pages/alerts.py", label="View Agent Action Proposals", icon=":material/notifications_active:")