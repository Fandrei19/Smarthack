from datetime import datetime, timezone
from subprocess import CalledProcessError, TimeoutExpired
from threading import Lock
from time import monotonic

import streamlit as st

from src.live_pipeline import refresh_live_data
from src.alert_service import load_alerts


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
    manual = st.button("Refresh now", key="refresh_live_now")
    st.caption("Open-Meteo forecast · refresh every 15 min while the app is open · UTC")
    with state["lock"]:
        now = monotonic()
        if manual or state["attempted"] is None or now - state["attempted"] >= 15 * 60:
            state["attempted"] = now
            try:
                with st.spinner("Refreshing forecast and agent decisions…"):
                    refresh_live_data()
            except (CalledProcessError, OSError, TimeoutExpired):
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

@st.fragment(run_every="1min")
def alert_notifications():
    alerts = load_alerts()
    acknowledged = st.session_state.setdefault("acknowledged_alerts", set())
    pending = [item for item in alerts if item["id"] not in acknowledged]
    ids = {item["id"] for item in alerts}
    previous = st.session_state.get("notified_alert_ids", set())
    new = ids - previous - acknowledged
    if new:
        st.toast(f"{len(new)} new alerts need human review. Open Alerts.", icon="⚠️")
    st.session_state["notified_alert_ids"] = ids
    if pending:
        st.warning(f"{len(pending)} alerts awaiting human review")
    else:
        st.caption("No unacknowledged alerts")
    st.page_link("pages/alerts.py", label="Open operator alerts", icon=":material/notifications:")

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

history = st.Page("pages/history.py", title="Weather History", icon=":material/history:")

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
    "ANALYTICS": [history]
}

pg = st.navigation(
    pages,
    # position = "top"
)

with st.sidebar:
    alert_notifications()

pg.run()
