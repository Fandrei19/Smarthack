import streamlit as st


st.set_page_config(
    page_title = "GridPilot AI",
    page_icon = "⚡",
    layout = "wide",
    initial_sidebar_state = "expanded"
)

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

agent = st.Page(
    "pages/ai_agent.py",
    title = "AI Agent",
    icon = ":material/smart_toy:"
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
st.sidebar.success("Monitoring active")

# navigation
pages = {
    "MONITORING": [overview, monitoring, maps],
    "ACTIONS": [agent, alerts],
    "ANALYTICS": [reports],
    "CONFIGURATION": [settings]
}

pg = st.navigation(
    pages, 
    # position = "top"
)

pg.run()