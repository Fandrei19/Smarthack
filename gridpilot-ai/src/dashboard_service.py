import pandas as pd
import streamlit as st

from src.data_loader import load_weather, load_predictions, load_agent
from src.data_processor import combine, add_derived, focus_hour
from src.live_pipeline import OUTPUT_DIR


def get_dashboard_data():
    weather = load_weather(OUTPUT_DIR / "weather_live.csv")
    predictions = load_predictions(OUTPUT_DIR / "predictions_live.csv")
    combined = add_derived(combine(weather, predictions))
    current = pd.Timestamp.now(tz="UTC").floor("h")
    combined = combined[combined["timestamp"] >= current]
    age = (pd.Timestamp.now(tz="UTC").timestamp() - (OUTPUT_DIR / "weather_live.csv").stat().st_mtime) / 60
    if age > 30:
        st.warning(f"Forecast download is {age:.0f} minutes old. Refresh before using recommendations.")
    if combined.empty:
        raise ValueError("No current or future forecast is available. Refresh now, or open Weather History.")
    hour = focus_hour(combined)
    decisions = load_agent(OUTPUT_DIR / "agent_results_live.json")
    return {
        "combined": combined,
        "hour": hour,
        "latest": combined[combined["timestamp"] == hour].set_index("segment_id"),
        "agent": decisions,
    }