import json
import time

import altair as alt
import pandas as pd
import streamlit as st

from src.live_pipeline import OUTPUT_DIR

STALE_AFTER_MIN = 30  # the pipeline refreshes every 15 min by default

ACTIONS = {
    "RECOMMEND_DYNAMIC_RATING": ("success", "🟢", "The line can carry more than usual"),
    "KEEP_STATIC": ("warning", "🟡", "Stick to the standard rating"),
    "OPERATOR_REVIEW": ("error", "🔴", "Operator review required"),
    "FALLBACK_STATIC": ("warning", "🟡", "Fallback to static rating"),
}


def _fmt(value, unit="", decimals=0):
    return "Unavailable" if pd.isna(value) else f"{value:.{decimals}f}{unit}"


def _read_live():
    weather = pd.read_csv(OUTPUT_DIR / "weather_live.csv")
    preds = pd.read_csv(OUTPUT_DIR / "predictions_live.csv")
    agent = json.loads((OUTPUT_DIR / "agent_results_live.json").read_text(encoding="utf-8"))
    for df in (weather, preds):
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = preds.merge(
        weather[["timestamp_utc", "segment_id", "wind_speed_ms", "air_temp_c"]],
        on=["timestamp_utc", "segment_id"], how="left")
    return df.dropna(subset=["timestamp_utc"]), agent


def _line_by_hour(df):
    """Line rating per hour = lowest q05 among segments; unknown if any segment is missing."""
    n = df["segment_id"].nunique()
    rows = []
    for ts, g in df.groupby("timestamp_utc"):
        q = g.set_index("segment_id")["q05_ampacity_a"]
        ok = len(q) == n and q.notna().all()
        rows.append({"timestamp_utc": ts,
                     "line_rating": q.min() if ok else float("nan"),
                     "limiting": q.idxmin() if ok else None,
                     "static": g["static_rating_a"].min()})
    return pd.DataFrame(rows)


@st.fragment(run_every="1min")
def live_monitor():
    st.subheader("🔴 Live monitoring")
    try:
        df, agent = _read_live()
    except (OSError, ValueError, KeyError):
        st.info("No live data yet. Start the pipeline: `python -m src.live_pipeline`")
        return

    # freshness
    age = (time.time() - (OUTPUT_DIR / "agent_results_live.json").stat().st_mtime) / 60
    if age > STALE_AFTER_MIN:
        st.error(f"Live data is {age:.0f} min old. Is the live pipeline running?")
    else:
        st.success(f"Updated {'just now' if age < 1 else f'{age:.0f} min ago'} · checks for new data every minute")

    # next forecast hour
    times = df["timestamp_utc"].drop_duplicates().sort_values()
    upcoming = times[times >= pd.Timestamp.now(tz="UTC").floor("h")]
    if upcoming.empty:
        st.warning("The live files contain only past hours.")
        return
    ts = upcoming.iloc[0]
    snap = df[df["timestamp_utc"] == ts].set_index("segment_id")
    st.caption(f"Forecast for {ts:%d %b, %H:%M} UTC. Weather is a forecast, while the model was trained on "
               "historical reanalysis; the q05–q95 range covers model error only, not forecast error.")

    # agent decision for that hour
    decision = next((d for d in agent
                     if pd.to_datetime(d.get("timestamp_utc"), utc=True) == ts), {}) \
        if isinstance(agent, list) else agent
    if decision:
        kind, icon, headline = ACTIONS.get(decision.get("action"),
                                           ("info", "ℹ️", str(decision.get("action", "N/A"))))
        getattr(st, kind)(f"**{icon} {headline}**  \n{decision.get('reason', '')}")
        if decision.get("requires_human_review"):
            st.warning("👤 Human operator confirmation required for this hour.")
    else:
        st.info("No agent decision for this hour.")

    # line + segments
    q05 = snap["q05_ampacity_a"]
    static = snap["static_rating_a"].min()
    complete = not q05.isna().any()
    rating = q05.min() if complete else float("nan")
    c1, c2, c3 = st.columns(3)
    c1.metric("Forecast line rating", _fmt(rating, " A"),
              None if pd.isna(rating) or pd.isna(static) else f"{rating - static:+.0f} A vs. static",
              border=True)
    c2.metric("Static limit", _fmt(static, " A"), border=True)
    c3.metric("Limiting segment", q05.idxmin() if complete else "Unavailable", border=True)

    cols = st.columns(len(snap))
    for col, (seg, r) in zip(cols, snap.iterrows()):
        pct = (r["q05_ampacity_a"] - r["static_rating_a"]) / r["static_rating_a"] * 100
        with col.container(border=True):
            st.markdown(f"**{seg}**")
            st.metric("Ampacity (q05)", _fmt(r["q05_ampacity_a"], " A"),
                      None if pd.isna(pct) else f"{pct:+.0f}% vs. static")
            st.caption(f"Wind {_fmt(r['wind_speed_ms'], ' m/s', 1)} · {_fmt(r['air_temp_c'], ' °C', 1)}")

    # next 24 h
    line_df = _line_by_hour(df)
    base = alt.Chart(line_df).encode(
        x=alt.X("timestamp_utc:T", title="UTC Time", scale=alt.Scale(type="utc")))
    area = base.mark_area(opacity=0.2, color="#2ca02c").encode(
        y=alt.Y("line_rating:Q", title="Ampacity (A)", scale=alt.Scale(zero=False)),
        y2="static:Q")
    line = base.mark_line(color="#2ca02c", point=True).encode(
        y="line_rating:Q",
        tooltip=["timestamp_utc:T", alt.Tooltip("line_rating:Q", format=".0f"), "limiting:N"])
    static_line = base.mark_line(strokeDash=[4, 4], color="#7f7f7f").encode(y="static:Q")
    st.altair_chart((area + line + static_line).properties(height=260), use_container_width=True)
    st.caption("Next 24 hours. Green = extra capacity above the static limit (dashed). "
               "Gaps = missing predictions.")