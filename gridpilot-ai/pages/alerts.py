import pandas as pd
import streamlit as st
from src.alert_service import load_alerts

st.title("Operator alerts")
st.caption("Notifications requiring attention, based on the live forecast and agent decisions. Times are UTC.")
st.info("Review indicates uncertainty or data issues, not a confirmed electrical fault. Acknowledging an alert does not approve a rating or resolve the underlying issue.")

@st.fragment(run_every="1min")
def alert_center():
    alerts = load_alerts()
    acknowledged = st.session_state.setdefault("acknowledged_alerts", set())
    pending = [a for a in alerts if a["id"] not in acknowledged]
    a,b,c = st.columns(3)
    a.metric("Awaiting review", len(pending))
    b.metric("Critical (agent HIGH risk)", sum(x["severity"] == "Critical" for x in alerts))
    c.metric("Acknowledged this session", sum(x["id"] in acknowledged for x in alerts))
    show_ack = st.checkbox("Include acknowledged alerts", value=True)
    visible = alerts if show_ack else pending
    if not alerts:
        st.success("No alerts requiring human review in the available forecast.")
    elif not visible:
        st.info("All current alerts have been acknowledged in this session.")
    for item in visible:
        with st.container(border=True):
            acknowledged_item = item["id"] in acknowledged
            title = f"{item['severity']} · {item['category']} · {item['segment']}"
            (st.error if item["severity"] == "Critical" else st.warning)(title)
            st.write(item["reason"])
            st.caption(f"Forecast time: {item['timestamp_utc']} · Agent action / next step: {item['action']}")
            if acknowledged_item:
                st.caption("Acknowledged in this browser session — condition still active.")
            elif st.button("Acknowledge — I will review", key="ack_" + item["id"]):
                acknowledged.add(item["id"])
                st.rerun()
    if alerts:
        export = pd.DataFrame([{**x, "acknowledged_this_session": x["id"] in acknowledged} for x in alerts])
        st.download_button("Download alert report", export.to_csv(index=False), "operator_alerts.csv", "text/csv")
    st.caption("Checks every minute while this page is open. Acknowledgements are local to this session.")

alert_center()
