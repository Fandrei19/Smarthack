import json
from io import StringIO
from pathlib import Path
import matplotlib.pyplot as plt  # type: ignore[reportMissingModuleSource]
import pandas as pd


def run_investment_calculator(
    agent_results_path: str = "data/processed/agent_results.json",
    voltage_kv: float = 110.0,
    power_factor: float = 0.95,
    wholesale_price_eur_mwh: float = 90.0,  # ~450 RON/MWh Romania OPCOM average baseline
    static_rating_a: float = 762.0,        # Baseline static rating for the corridor
    output_plot_path: str = "reports/investment_chart.png"
) -> pd.DataFrame:
    """
    Loads agent_results.json, chunks the timeline into 30-day blocks starting 
    from the first timestamp, calculates financial returns, and plots a monthly chart.
    """
    path = Path(agent_results_path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find results file at {path}. Run the agent first!")

    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    df = pd.DataFrame(raw_data)
    if df.empty:
        raise ValueError("The agent_results.json file is empty.")

    # Parse timestamps
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    start_time = df["timestamp_utc"].min()
    end_time = df["timestamp_utc"].max()
    print(f"Dataset Timeline: {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')} ({len(df)} total hours)")

    # Calculate extra ampacity headroom for hours where dynamic rating is recommended
    valid_mask = (df["action"] == "RECOMMEND_DYNAMIC_RATING") & (df["line_rating_q05_a"].notna())
    
    extra_amps = pd.Series(0.0, index=df.index)
    extra_amps[valid_mask] = df.loc[valid_mask, "line_rating_q05_a"] - static_rating_a
    extra_amps = extra_amps.clip(lower=0.0)

    # Convert Amps to Megawatts (MW) for a 3-phase AC line: P = (sqrt(3) * V * I * cos_phi) / 1000
    df["extra_mw"] = (1.73205 * voltage_kv * extra_amps * power_factor) / 1000.0

    # 1 row = 1 hour -> MWh per hour row
    df["saved_energy_mwh"] = df["extra_mw"] * 1.0
    df["financial_benefit_eur"] = df["saved_energy_mwh"] * wholesale_price_eur_mwh

    # Group into exact 30-day blocks starting from the very first timestamp
    df["period_days_index"] = (df["timestamp_utc"] - start_time).dt.days // 30

    period_summaries = []
    for period_idx, group in df.groupby("period_days_index"):
        p_start = start_time + pd.Timedelta(days=int(period_idx * 30))
        p_end = p_start + pd.Timedelta(days=30)
        
        total_mwh = group["saved_energy_mwh"].sum()
        total_eur = group["financial_benefit_eur"].sum()
        active_hours = int((group["action"] == "RECOMMEND_DYNAMIC_RATING").sum())
        fallback_hours = int((group["action"] == "FALLBACK_STATIC").sum())

        period_summaries.append({
            "period_number": int(period_idx + 1),
            "start_date": p_start.strftime("%Y-%m-%d"),
            "end_date": p_end.strftime("%Y-%m-%d"),
            "dynamic_active_hours": active_hours,
            "safety_fallback_hours": fallback_hours,
            "extra_clean_energy_mwh": round(total_mwh, 2),
            "estimated_revenue_eur": round(total_eur, 2)
        })

    summary_df = pd.DataFrame(period_summaries)
    
    # Generate the Financial Chart
    generate_investment_chart(summary_df, output_plot_path)

    return summary_df


def generate_investment_chart(summary_df: pd.DataFrame, output_path: str) -> None:
    """
    Generates a professional bar chart showing money made (€) for each 30-day period.
    """
    if summary_df.empty:
        return

    # Format labels for the x-axis
    labels = [f"{row['period_number']}\n({row['start_date']} to {row['end_date']})" for _, row in summary_df.iterrows()]
    values = summary_df["estimated_revenue_eur"] / 1000.0  # Convert to Thousands of Euros (€k) for cleaner axis labels

    chart_width = max(14, len(summary_df) * 3.2)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(chart_width, 6))

    bars = plt.bar(labels, values, color="#1f77b4", edgecolor="#0b3c5d", width=0.5, alpha=0.85)

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + max(values)*0.01,
            f"€{height:,.1f}k",
            ha='center', va='bottom', fontsize=10, fontweight='bold'
        )

    plt.title("GridPilot AI — Estimated 30-Day Unlocked Renewable Value (€ Thousands)", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("30-Day Operational Windows", fontsize=11, labelpad=10)
    plt.ylabel("Avoided Curtailment Value (€k)", fontsize=11, labelpad=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig(output_file, dpi=300)

    svg_buffer = StringIO()
    plt.savefig(svg_buffer, format="svg", bbox_inches="tight")
    svg_markup = svg_buffer.getvalue()
    svg_markup = svg_markup[svg_markup.find("<svg"):]

    html_path = output_file.with_suffix(".html")
    html_path.write_text(
        f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GridPilot AI Investment Chart</title>
  <style>
    body {{ font-family: sans-serif; margin: 1rem; }}
    .chart-scroll {{ overflow-x: auto; border: 1px solid #ddd; }}
    .chart-scroll svg {{
      display: block;
      width: {chart_width * 96}px;
      max-width: none;
      height: auto;
    }}
  </style>
</head>
<body>
  <h1>GridPilot AI Investment Chart</h1>
  <p>Scroll horizontally to view all 30-day periods.</p>
  <div class="chart-scroll">{svg_markup}</div>
</body>
</html>
""",
        encoding="utf-8",
    )
    print(f"\nInvestment bar chart saved to {output_file}")
    print(f"Scrollable investment chart saved to {html_path}")
    plt.close()


if __name__ == "__main__":
    print("Running GridPilot AI 30-Day Investment & Financial Analysis...")
    summary = run_investment_calculator()
    print("\n--- Financial Return per 30-Day Window ---")
    print(summary.to_string(index=False))
    
    # Save CSV summary
    summary.to_csv("data/processed/investment_summary_30days.csv", index=False)
    print("\nSaved CSV summary to data/processed/investment_summary_30days.csv")