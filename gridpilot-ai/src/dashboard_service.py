from src.data_loader import load_weather, load_predictions, load_agent
from src.data_processor import combine, add_derived
from src.live_pipeline import OUTPUT_DIR

# acts as an orchestrator that gets all the necessary datasets for a dashboard in a single call
def get_dashboard_data():
    weather = load_weather(OUTPUT_DIR / "weather_live.csv")
    predictions = load_predictions(OUTPUT_DIR / "predictions_live.csv")
    combined = add_derived(combine(weather, predictions))
    first_hour = combined["timestamp"].min()
    decisions = load_agent(OUTPUT_DIR / "agent_results_live.json")
    return {
        "combined": combined,
        "latest": combined[combined["timestamp"] == first_hour].set_index("segment_id"),
        "agent": decisions,
    }
