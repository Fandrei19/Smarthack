import pandas as pd
import pydeck as pdk
import streamlit as st

from src.dashboard_service import get_dashboard_data
from src.data_processor import decision_for_hour

st.set_page_config(layout="wide")
st.title("Dobrogea · simulated generation sites")
st.caption("P1–P3 are illustrative generation sites at the existing weather coordinates. "
           "SEG_1–SEG_3 remain conductor sections. Plant injections and power flows are not simulated yet.")

# data loading and validation
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
    st.warning("No weather or prediction data available to display.")
    st.stop()

# coordinate and geometry configuration (extensible for all regions)
# note: pydeck expects coordinates in GeoJSON format: [long, lat]
SEGMENT_GEOMETRY = {
    # Dobrogea Region (default segments)
    "SEG_1": {"path": [[28.14, 44.38], [28.30, 44.42]], "center": [28.14, 44.38]},
    "SEG_2": {"path": [[28.30, 44.42], [28.48, 44.46]], "center": [28.48, 44.46]},
    "SEG_3": {"path": [[28.48, 44.46], [28.81, 44.55]], "center": [28.81, 44.55]},

    # template for future regional expansion (e.g., Muntenia / Transylvania):
    # "SEG_MUNTENIA_1": {"path": [[26.10, 44.43], [25.96, 44.45]], "center": [26.03, 44.44]},
}

def fmt(value, unit="", decimals=0):
    """Format numeric values safely, returning 'Unavailable' for NaN."""
    return "Unavailable" if pd.isna(value) else f"{value:.{decimals}f}{unit}"

def room_status(pct):
    """
    Determine room status label, emoji, and RGB color
    based on percentage headroom above static rating limit.
    """
    if pd.isna(pct):
        return "⚪", "No data", [150, 150, 150]
    if pct >= 25:
        return "🟢", "Plenty of room", [44, 160, 44]
    if pct >= 5.3:
        return "🟡", "Limited room", [255, 170, 0]
    return "🔴", "Almost no room", [214, 39, 40]


# time selection and data wrangling
times = sorted(combined["timestamp"].drop_duplicates())
hour = st.select_slider(
    "Hour (UTC)",
    options=times,
    value=times[0],
    format_func=lambda t: f"{t:%b %d, %H:%M}"
)
snap = combined[combined["timestamp"] == hour].set_index("segment_id")
decision = decision_for_hour(data["agent"], hour)
if decision:
    show = st.warning if decision["requires_human_review"] else st.info
    show(f"Agent recommendation: **{decision['action']}** · {decision['reason']}")

rows = []
for s_id, geom in SEGMENT_GEOMETRY.items():
    r = snap.loc[s_id] if s_id in snap.index else None
    pct = r["extra_pct"] if r is not None else float("nan")
    emoji, label, color = room_status(pct)

    amp_str = fmt(r["q05"] if r is not None else float("nan"), " A")
    vs_str = "n/a" if pd.isna(pct) else f"{pct:+.0f}% vs. usual limit"

    rows.append({
        "name": s_id,
        "pct": pct,
        "emoji": emoji,
        "status": label,
        "color": color,
        "path": geom["path"],
        "center": geom["center"],
        "amp": amp_str,
        "vs": vs_str,
        "wind": fmt(r["wind_speed"] if r is not None else float("nan"), " m/s", 1),
        "temp": fmt(r["temperature"] if r is not None else float("nan"), " °C", 1),
        "tooltip_text": f"Segment: {s_id}\nStatus: {label}\nCapacity: {amp_str} ({vs_str})"
    })

df = pd.DataFrame(rows)
plants = pd.DataFrame([
    {"center": SEGMENT_GEOMETRY[seg]["center"], "code": f"P{i}",
     "label": f"Plant P{i}", "color": color,
     "tooltip_text": f"P{i}: illustrative generation site\nWeather reference: {seg}\nNo measured generation or SCADA connection"}
    for i, (seg, color) in enumerate(zip(SEGMENT_GEOMETRY,
        [[0, 114, 178], [230, 159, 0], [204, 121, 167]]), start=1)
])

# headline alerts and weakest segment identification
known = df.dropna(subset=["pct"])
weakest = known.loc[known["pct"].idxmin(), "name"] if not known.empty else None
red = list(df.loc[df["status"] == "Almost no room", "name"])
yellow = list(df.loc[df["status"] == "Limited room", "name"])
nodata = list(df.loc[df["status"] == "No data", "name"])

if red:
    st.error(f"**Attention on {', '.join(red)}**: almost no spare capacity at {hour:%H:%M} UTC.")
elif yellow:
    st.warning(f"**Limited room on {', '.join(yellow)}** at {hour:%H:%M} UTC.")
elif known.empty:
    st.info("No prediction available for this hour.")
else:
    st.success(f"**All segments have room** at {hour:%H:%M} UTC.")

if nodata and not known.empty:
    st.warning(f"No prediction for {', '.join(nodata)} at this hour, so line capacity cannot be computed.")

# UI layout: pydeck map and side cards
left = st.container()
right = st.container()

with left:
    # layer 1: power line transmission paths
    path_layer = pdk.Layer(
        "PathLayer",
        df,
        get_path="path",
        get_color="color",
        width_scale=20,
        width_min_pixels=6,
        pickable=True,
        auto_highlight=True,
    )

    # Fixed-size outlined site markers remain readable at any zoom.
    point_layer = pdk.Layer(
        "ScatterplotLayer", plants, get_position="center",
        get_fill_color="color", get_line_color=[255, 255, 255],
        stroked=True, line_width_min_pixels=3,
        get_radius=180, radius_scale=1, radius_min_pixels=12, radius_max_pixels=16,
        pickable=True,
    )
    code_layer = pdk.Layer(
        "TextLayer", plants, get_position="center", get_text="code",
        get_color=[255, 255, 255], get_size=14, size_scale=1, size_min_pixels=14, size_max_pixels=14, get_alignment_baseline="'center'",
        get_text_anchor="'middle'", font_weight=700,
    )
    label_layer = pdk.Layer(
        "TextLayer", plants, get_position="center", get_text="label",
        get_color=[20, 40, 60], get_size=15, size_scale=1, size_min_pixels=15, size_max_pixels=15, get_pixel_offset=[0, -36],
        get_text_anchor="'middle'", get_alignment_baseline="'bottom'",
        background=True, get_background_color=[255, 255, 255, 235],
        background_padding=[7, 5], font_weight=600,
    )
    view_state = pdk.ViewState(
        latitude=44.465, longitude=28.475, zoom=8.5, pitch=0
    )

    r_deck = pdk.Deck(
        layers=[path_layer, point_layer, code_layer, label_layer],
        initial_view_state=view_state,
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        tooltip={"text": "{tooltip_text}"}
    )

    st.pydeck_chart(r_deck, height=570)
    st.caption("P1 / P2 / P3 = illustrative plants; colored paths = conductor segments. "
               "🟢 plenty of room · 🟡 limited · 🔴 almost none · ⚪ no data. Hover over segments for details.")

with right:
    st.subheader("Simulated sites")
    for plant in plants.to_dict("records"):
        st.markdown(f"**{plant['code']} · Simulated plant**")
        st.caption(f"{plant['center'][1]:.3f}° N, {plant['center'][0]:.3f}° E · illustrative location")
    st.divider()
    st.subheader("Conductor segments")
    for r in rows:
        with st.container(border=True):
            st.markdown(f"**{r['emoji']} {r['name']}**" + (" · weakest" if r["name"] == weakest else ""))
            st.caption(f"{r['status']}  \nCapacity {r['amp']} ({r['vs']})  \n"
                       f"Wind {r['wind']} · {r['temp']}")

    st.page_link("pages/line_monitoring.py", label="See details and weather",
                 icon=":material/electric_bolt:")
