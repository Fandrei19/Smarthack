import pandas as pd
import streamlit as st
import altair as alt 

from src.dashboard_service import get_dashboard_data

st.title("Overview")

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
latest = data["latest"]
agent = data["agent"]

if isinstance(agent, list):
    first_hour = combined["timestamp"].min()
    agent = next(
        (row for row in agent if pd.to_datetime(row.get("timestamp_utc"), utc=True) == first_hour),
        {},
    )

# checks if the dataset is empty
if combined.empty:
    st.warning("No weather/prediction data to display.")
    st.stop()

# formats raw numbers for display
def fmt(value, unit = "", decimals = 0, signed = False):
    if pd.isna(value):
        return "Unavailable"
    sign = "+" if signed else ""

    return f"{value:{sign}.{decimals}f}{unit}"


# segment list extraction and timestamp display
segments = sorted(latest.index)
critical = agent.get("bottleneck_segment")
ts = combined["timestamp"].min()
st.caption(f"First forecast hour: {ts:%Y-%m-%d %H:%M UTC}")

# agent status
status = str(agent.get("risk_level", "")).upper()
banner = {
    "HIGH": st.error,
    "UNCERTAIN": st.warning,
    "MEDIUM": st.warning,
    "LOW": st.success,
}.get(status, st.info)

banner(f"**{status or 'N/A'}** · {agent.get('action', 'N/A')}")


# --- key performance indicators (KPI grid)
# --- line-level values
amp = latest["predicted_ampacity"]
static = latest["static_rating"].min()


# if any segment has no prediction, the line rating is unknown (never ignore a missing segment)
if amp.isna().any():
    line_rating, bottleneck = float("nan"), None
else:
    line_rating, bottleneck = amp.min(), amp.idxmin()

line_gain = line_rating - static
line_gain_pct = line_gain / static * 100
gain_label = (None if pd.isna(line_gain)
              else f"{line_gain:+.0f} A ({line_gain_pct:+.1f}%) vs. static")

n_alerts = int(bool(agent.get("requires_human_review")))


# --- line KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Static Rating", fmt(static, " A"), border=True)
c2.metric("Dynamic Line Rating", fmt(line_rating, " A"), gain_label, border=True)
c3.metric("Bottleneck Segment", bottleneck or "Unavailable", border=True)
c4.metric("Active Alerts", n_alerts, border=True)
st.caption("Dynamic line rating = lowest estimated ampacity among the segments.")

if bottleneck and critical and bottleneck != critical:
    st.warning(f"Bottleneck from predictions is {bottleneck}, "
               f"but the agent reports {critical} as critical.")


# --- segment cards
st.subheader("Segments")
cols = st.columns(len(segments))

for col, s in zip(cols, segments):
    row = latest.loc[s]

    with col.container(border=True):
        tags = []

        if s == bottleneck:
            tags.append("bottleneck")

        if s == critical:
            tags.append("agent: critical")

        st.markdown(f"**Segment {s}**" + (f" · _{', '.join(tags)}_" if tags else ""))

        st.metric("Ampacity", fmt(row["predicted_ampacity"], " A"),
                  None if pd.isna(row["extra_pct"])
                  else f"{row['extra_pct']:+.1f}% vs. static")

        st.caption(f"Uncertainty: {fmt(row['uncertainty'], '', 2)} · "
                   f"{fmt(row['temperature'], ' °C', 1)} · "
                   f"{fmt(row['wind_speed'], ' m/s', 1)}")


# chart + weather
left, right = st.columns([2, 1])

with left:
    with st.container(border = True):
        st.subheader("Line & Segment Ampacity Over Time")
        hours = st.selectbox("Interval", [6, 12, 24], index = 2,
                             format_func = lambda h: f"Next {h}h")

        window = combined[combined["timestamp"] < ts + pd.Timedelta(hours=hours)]
        
        chart_data = window.pivot(index = "timestamp", columns = "segment_id",
                                  values = "predicted_ampacity")
        chart_data["Static"] = window.groupby("timestamp")["static_rating"].min()

        segment_columns = [
            col for col in chart_data.columns
            if col != "Static"
        ]

        chart_data["Dynamic Line Rating"] = (
           chart_data[segment_columns].min(axis=1, skipna=False)
        )

        chart_df = chart_data.reset_index().melt(
            id_vars=["timestamp"], 
            var_name="Segment", 
            value_name="Ampacity"
        )
        
        segments_list = [
            col for col in chart_data.columns
            if col not in ["Static", "Dynamic Line Rating"]
        ]

        chart_series = (
            segments_list
            + ["Dynamic Line Rating", "Static"]
        )
        
        color_scale = alt.Scale(
            domain=chart_series,
            range=[
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd"
            ]
        )

        c = alt.Chart(chart_df).mark_line().encode(
            x = alt.X("timestamp:T", title="Timestamp"),
            y = alt.Y("Ampacity:Q", title="Ampacity (A)"),
            color = alt.Color("Segment:N", scale=color_scale, title="Segment"),

            strokeDash = alt.condition(
                alt.datum.Segment == 'Static', 
                alt.value([5, 5]),                      # dotted static line
                alt.value([0])     
            )
        ).properties(
            height=320
        ).interactive()

        st.altair_chart(c, use_container_width=True)


with right:
    with st.container(border = True):
        st.subheader("Weather Conditions")
        default = segments.index(critical) if critical in segments else 0

        seg = st.selectbox("Segment", segments, index=default)
        
        w = latest.loc[seg]
        m1, m2 = st.columns(2)
        
        m1.metric("Temperature", fmt(w["temperature"], " °C", 1))
        m2.metric("Wind Speed", fmt(w["wind_speed"], " m/s", 1))
        
        m3, m4 = st.columns(2)
        m3.metric("Solar Radiation", fmt(w["solar_radiation"], " W/m²"))
        m4.metric("Wind Direction", fmt(w["wind_direction"], "°"))
        


# agent recommendation
with st.container(border=True):
    st.subheader("Agent Recommendation")
    st.write(f"**Reason:** {agent.get('reason', 'N/A')}")
    st.write(f"**Action:** {agent.get('action', 'N/A')}")
    
    for a in agent.get("alerts", []):
        sev = str(a.get("severity", "")).upper()
        show = {"CRITICAL": st.error, "WARNING": st.warning}.get(sev, st.info)
        show(f"Segment {a.get('segment', '?')}: {a.get('message', '')}")
    
    st.page_link("pages/ai_agent.py", label="Open agent",
                 icon=":material/smart_toy:")
