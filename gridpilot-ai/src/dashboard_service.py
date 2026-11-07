from src.data_loader import load_weather, load_predictions, load_agent
from src.data_processor import combine, add_derived, latest_snapshot

# acts as an orchestrator that gets all the necessary datasets for a dashboard in a single call
def get_dashboard_data():
    combined = add_derived(combine(load_weather(), load_predictions()))
    return {
        "combined": combined,
        "latest": latest_snapshot(combined),
        "agent": load_agent(),
    }