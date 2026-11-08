import pandas as pd
from src.alert_service import build_alerts, load_alerts


def sample():
    decisions = [dict(timestamp_utc="2026-11-08T10:00:00Z", requires_human_review=True,
        action="FALLBACK_STATIC", risk_level="UNCERTAIN", reason="High uncertainty", bottleneck_segment="SEG_2")]
    weather = pd.DataFrame([dict(timestamp_utc=decisions[0]["timestamp_utc"], segment_id=s, data_quality="ok",
        air_temp_c=10, wind_speed_ms=2, wind_direction_deg=90, solar_radiation=100) for s in ["SEG_1","SEG_2","SEG_3"]])
    predictions=weather.assign(q05_ampacity_a=800, q50_ampacity_a=900, q95_ampacity_a=1000)
    return decisions,weather,predictions


def test_uncertainty_is_review_not_confirmed_critical_fault():
    result=build_alerts(*sample(), now="2026-11-08T09:30Z", updated_at="2026-11-08T09:25Z")
    assert len(result)==1
    assert result[0]["severity"]=="Review"
    assert result[0]["segment"]=="SEG_2"
    assert result==build_alerts(*sample(), now="2026-11-08T09:31Z", updated_at="2026-11-08T09:25Z")


def test_missing_segment_and_stale_data():
    d,w,p=sample()
    result=build_alerts(d,w.iloc[:2],p,now="2026-11-08T09:30Z",updated_at="2026-11-08T08:00Z")
    assert any(a["category"]=="Stale data" for a in result)
    assert any(a["segment"]=="SEG_3" and "missing" in a["reason"] for a in result)


def test_expired_decisions_are_not_current_alerts():
    result=build_alerts(*sample(),now="2026-11-09T09:30Z",updated_at="2026-11-09T09:25Z")
    assert [a["category"] for a in result]==["Expired forecast"]


def test_missing_files_generate_notification(tmp_path):
    assert load_alerts(tmp_path)[0]["category"]=="Data unavailable"
