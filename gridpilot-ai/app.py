from datetime import datetime, timezone
from subprocess import CalledProcessError
from threading import Lock
from time import monotonic

import streamlit as st

from src.live_pipeline import refresh_live_data


st.set_page_config(
    page_title = "GridPilot AI",
    page_icon = "⚡",
    layout = "wide",
    initial_sidebar_state = "expanded"
)


@st.cache_resource
def live_refresh_state():
    return {"lock": Lock(), "attempted": None, "updated": None, "error": False}


@st.fragment(run_every="1min")
def live_status():
    state = live_refresh_state()
    with state["lock"]:
        now = monotonic()
        if state["attempted"] is None or now - state["attempted"] >= 15 * 60:
            state["attempted"] = now
            try:
                with st.spinner("Refreshing forecast and agent decisions…"):
                    refresh_live_data()
            except (CalledProcessError, OSError):
                state["error"] = True
            else:
                st.cache_data.clear()
                state["updated"] = datetime.now(timezone.utc)
                state["error"] = False
        updated, error = state["updated"], state["error"]

    if error:
        st.warning("Live refresh failed. The dashboard shows the last available forecast.")
    previous = st.session_state.get("displayed_live_update")
    was_rendered = "displayed_live_update" in st.session_state
    st.session_state["displayed_live_update"] = updated
    
    if updated is not None:
        st.caption(f"Forecast refreshed {updated:%d %b %Y, %H:%M UTC}")
        if was_rendered and previous != updated:
            st.rerun()

# pages
overview = st.Page(
    "pages/overview.py",
    title = "Overview",
    icon = ":material/dashboard:",
    default = True
)

maps = st.Page(
    "pages/map.py",
    title = "Map",
)

monitoring = st.Page(
    "pages/line_monitoring.py",
    title = "Line Monitoring",
    icon = ":material/electric_bolt:"
)

alerts = st.Page(
    "pages/alerts.py",
    title = "Alerts",
    icon = ":material/notifications:"
)

reports = st.Page(
    "pages/reports.py",
    title = "Reports",
    icon = ":material/description:"
)

settings = st.Page(
    "pages/settings.py",
    title = "Settings",
    icon = ":material/settings:"
)

# sidebar
st.sidebar.title("⚡ GridPilot AI")
st.sidebar.caption("Grid Intelligence Platform")
st.sidebar.divider()

st.sidebar.markdown("**SYSTEM STATUS**")
with st.sidebar:
    live_status()

# navigation
pages = {
    "MONITORING": [overview, monitoring, maps],
    "ACTIONS": [alerts],
    "ANALYTICS": [reports],
    "CONFIGURATION": [settings]
}

pg = st.navigation(
    pages, 
    # position = "top"
)

pg.run()
