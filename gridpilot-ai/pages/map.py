import pandas as pd
import pydeck as pdk
import streamlit as st

from src.dashboard_service import get_dashboard_data
from src.data_processor import filter_window

st.set_page_config(layout="wide")
st.title("Power Grid Map - Romania")

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
times = sorted(filter_window(combined, 24)["timestamp"].drop_duplicates())
hour = st.select_slider(
    "Hour (UTC)", 
    options=times, 
    value=times[-1],
    format_func=lambda t: f"{t:%b %d, %H:%M}"
)
snap = combined[combined["timestamp"] == hour].set_index("segment_id")

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
left, right = st.columns([3, 1])

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

    # layer 2: station / measurement center points
    point_layer = pdk.Layer(
        "ScatterplotLayer",
        df,
        get_position="center",
        get_color="color",
        get_radius=3000,
        radius_min_pixels=8,
        pickable=True,
    )

    # set initial camera view centered over Eastern Romania (Dobrogea focus)
    # adjust zoom level to ~6.5 if displaying all of Romania at once
    view_state = pdk.ViewState(
        latitude=44.60,
        longitude=28.40,
        zoom=8.5,
        pitch=0
    )

    r_deck = pdk.Deck(
        layers=[path_layer, point_layer],
        initial_view_state=view_state,
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        tooltip={"text": "{tooltip_text}"}
    )

    st.pydeck_chart(r_deck)
    st.caption("🟢 plenty of room · 🟡 limited · 🔴 almost none · ⚪ no data. Hover over segments for details.")

with right:
    st.subheader("Segments")
    for r in rows:
        with st.container(border=True):
            st.markdown(f"**{r['emoji']} {r['name']}**" + (" · weakest" if r["name"] == weakest else ""))
            st.caption(f"{r['status']}  \nCapacity {r['amp']} ({r['vs']})  \n"
                       f"Wind {r['wind']} · {r['temp']}")
            
    st.page_link("pages/line_monitoring.py", label="See details and weather",
                 icon=":material/electric_bolt:")