import altair as alt
import pandas as pd
import streamlit as st

from src.dashboard_service import get_dashboard_data
from src.data_processor import decision_for_hour, line_rating_by_time

st.title("⚡ Grid status")

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

latest = data["latest"]
ts = combined["timestamp"].min()
agent = decision_for_hour(data["agent"], ts)


def fmt(value, unit="", decimals=0):
    return "Unavailable" if pd.isna(value) else f"{value:.{decimals}f}{unit}"


# agent action in everyday words: (alert style, icon, headline, explanation)
DECISIONS = {
    "RECOMMEND_DYNAMIC_RATING": ("success", "🟢", "The line can carry more than usual",
                                 "Today's weather cools the cables well, so there is room to spare."),
    "KEEP_STATIC": ("warning", "🟡", "Stick to the usual limit",
                    "There is some extra room, but not enough to change anything."),
    "OPERATOR_REVIEW": ("error", "🔴", "Please take a look",
                        "The extra room is almost gone, so a person should decide."),
    "FALLBACK_STATIC": ("warning", "🟡", "Play it safe for now",
                    "The available estimates do not justify a dynamic limit."),
}


def room_status(pct):
    """Traffic light for a segment, using the same cut-offs as the agent (R = 0.80 / 0.95)."""
    if pd.isna(pct):
        return "⚪", "No data"
    if pct >= 25:
        return "🟢", "Plenty of room"
    if pct >= 5.3:
        return "🟡", "Limited room"
    return "🔴", "Almost no room"


amp = latest["predicted_ampacity"]
static = latest["static_rating"].min()

if amp.isna().any():
    line_rating, bottleneck = float("nan"), None
else:
    line_rating, bottleneck = amp.min(), amp.idxmin()

segments = sorted(latest.index)
gain = line_rating - static

st.caption(f"First forecast hour: {ts:%b %d, %H:%M} UTC")

# 1. the answer
if agent:
    kind, icon, headline, why = DECISIONS.get(agent.get("action"),
                                              ("info", "ℹ️", str(agent.get("action", "N/A")), ""))
    with st.container(border=True):
        getattr(st, kind)(f"## {icon} {headline}\n{why}")
        if not pd.isna(gain):
            a, b = st.columns(2)
            a.metric("Estimated dynamic limit", fmt(line_rating, " A"),
                     f"{gain:+.0f} A ({gain / static * 100:+.0f}%) vs. the usual limit",
                     help="Set by the weakest segment, using a cautious (q05) estimate.")
            b.metric("Usual limit", fmt(static, " A"),
                     help="The fixed limit that applies in any weather.")
        st.caption(f"Agent's note: {agent.get('reason', 'N/A')}")
        if agent.get("requires_human_review"):
            st.write("👤 **A person needs to confirm this decision.**")
else:
    st.info("The agent has no decision for this hour.")


# 2. why: the weakest segment
if bottleneck:
    r = latest.loc[bottleneck]
    st.info(f"**Why this number?** The line is only as strong as its weakest part, "
            f"**{bottleneck}**: wind {fmt(r['wind_speed'], ' m/s', 1)}, "
            f"{fmt(r['temperature'], ' °C', 1)}, sun {fmt(r['solar_radiation'], ' W/m²')}. "
            f"Less wind and more sun or heat mean less cooling.")


limit_agent = agent.get("bottleneck_segment")
if bottleneck and limit_agent and bottleneck != limit_agent:
    st.warning(f"The predictions point to {bottleneck}, but the agent says {limit_agent}.")


# 3. segments with traffic lights
st.subheader("How is each part of the line doing?")
cols = st.columns(len(segments))
for col, s in zip(cols, segments):
    row = latest.loc[s]
    dot, label = room_status(row["extra_pct"])
    with col.container(border=True):
        st.markdown(f"### {dot} {s}")
        st.markdown(f"**{label}**" + (" · weakest part" if s == bottleneck else ""))
        st.metric("Can carry", fmt(row["predicted_ampacity"], " A"),
                  None if pd.isna(row["extra_pct"]) else f"{row['extra_pct']:+.0f}% vs. usual limit")

# 4. next 24h
st.subheader("Capacity forecast for the next 24 hours")
line_df = line_rating_by_time(combined)

base = alt.Chart(line_df).encode(x=alt.X("timestamp:T", title="Time (UTC)", scale=alt.Scale(type="utc")))

rating = base.mark_area(opacity=0.15, color="#2ca02c").encode(
    y=alt.Y("line_rating:Q", title="Amps (A)", scale=alt.Scale(zero=False)),
    y2="static_rating:Q")

rating_line = base.mark_line(color="#2ca02c", point=True).encode(
    y="line_rating:Q",
    tooltip=["timestamp:T", alt.Tooltip("line_rating:Q", format=".0f", title="Can carry"),
             alt.Tooltip("limiting_segment:N", title="Weakest part")])

static_line = base.mark_line(strokeDash=[5, 5], color="#7f7f7f").encode(y="static_rating:Q")

st.altair_chart((rating + rating_line + static_line).properties(height=260),use_container_width=True)

st.caption("Green line = what the line can carry. Dashed gray = the usual limit. "
           "The shaded gap is the extra room. Gaps in the line = missing data.")

# 5. learn more
with st.expander("How does the agent decide?"):
    ratio = static / line_rating if pd.notna(line_rating) and line_rating > 0 else float("nan")
    st.write(f"It compares the static limit to the estimated dynamic limit. "
             f"The ratio for this hour is **{fmt(ratio, '', 2)}**.")
    
    st.markdown(
        "- **0.80 or lower:** the line can carry more than usual\n"
        "- **0.80 to 0.95:** stick to the usual limit\n"
        "- **above 0.95:** a person should look\n"
        "- **missing data:** play it safe")
    st.caption("These thresholds are assumptions made for the prototype.")


l1, l2 = st.columns(2)
l1.page_link("pages/line_monitoring.py", label="See details and weather",
             icon=":material/electric_bolt:")

l2.page_link("pages/ai_agent.py", label="Ask the agent", icon=":material/smart_toy:")
