from src.data_loader import load_weather, load_predictions, load_agent
from src.data_processor import combine, add_derived, focus_hour
from src.live_pipeline import OUTPUT_DIR


def get_dashboard_data():
    weather = load_weather(OUTPUT_DIR / "weather_live.csv")
    predictions = load_predictions(OUTPUT_DIR / "predictions_live.csv")
    combined = add_derived(combine(weather, predictions))
    hour = focus_hour(combined)
    decisions = load_agent(OUTPUT_DIR / "agent_results_live.json")
    return {
        "combined": combined,
        "hour": hour,
        "latest": combined[combined["timestamp"] == hour].set_index("segment_id"),
        "agent": decisions,
    }