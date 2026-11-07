import math


UNCERTAINTY_ELEVATED_THRESHOLD = 0.05
UNCERTAINTY_HIGH_THRESHOLD = 0.10


def classify_uncertainty(uncertainty_ratio: float) -> dict:
    """
    Classify prediction uncertainty using prototype thresholds.

    High uncertainty triggers a static fallback recommendation.
    Elevated uncertainty recommends retaining the static reference.
    """
    try:
        uncertainty = float(uncertainty_ratio)
    except (TypeError, ValueError):
        uncertainty = float("nan")

    if not math.isfinite(uncertainty) or uncertainty < 0:
        return {
            "uncertainty_status": "INVALID",
            "reason_code": "UNCERTAINTY_INVALID",
            "risk_level": "UNCERTAIN",
            "action": "FALLBACK_STATIC",
            "requires_human_review": True,
            "reason": "Prediction uncertainty is missing or invalid."
        }

    if uncertainty > UNCERTAINTY_HIGH_THRESHOLD:
        return {
            "uncertainty_status": "HIGH",
            "reason_code": "UNCERTAINTY_HIGH",
            "risk_level": "UNCERTAIN",
            "action": "FALLBACK_STATIC",
            "requires_human_review": True,
            "reason": "Prediction uncertainty exceeds the high threshold."
        }

    if uncertainty > UNCERTAINTY_ELEVATED_THRESHOLD:
        return {
            "uncertainty_status": "ELEVATED",
            "reason_code": "UNCERTAINTY_ELEVATED",
            "risk_level": "MEDIUM",
            "action": "KEEP_STATIC",
            "requires_human_review": False,
            "reason": "Prediction uncertainty is elevated."
        }

    return {
        "uncertainty_status": "NORMAL",
        "reason_code": "UNCERTAINTY_NORMAL",
        "risk_level": "LOW",
        "action": "RECOMMEND_DYNAMIC_RATING",
        "requires_human_review": False,
        "reason": "Prediction uncertainty is within the normal threshold."
    }


def calculate_utilization(
    line_rating_q05_a: float,
    static_rating_a: float
) -> float:
    """
    Calculate the ratio between reference static ampacity and conservative dynamic ampacity (Q05).
    """
    if line_rating_q05_a is None or static_rating_a is None:
        raise ValueError("Ampacity values cannot be None.")

    if not math.isfinite(line_rating_q05_a) or not math.isfinite(static_rating_a):
        raise ValueError("Ampacity values must be finite.")

    if line_rating_q05_a <= 0 or static_rating_a <= 0:
        raise ValueError("Ampacity values must be positive.")

    return static_rating_a / line_rating_q05_a


def classify_risk(utilization: float) -> dict:
    """
    Classify operational risk based on the utilization ratio (static / dynamic Q05).
    """
    if not math.isfinite(utilization) or utilization < 0:
        raise ValueError("Utilization must be finite and non-negative.")

    if utilization <= 0.80:
        return {
            "risk_level": "LOW",
            "action": "RECOMMEND_DYNAMIC_RATING",
            "requires_human_review": False
        }
    elif utilization <= 0.95:
        return {
            "risk_level": "MEDIUM",
            "action": "KEEP_STATIC",
            "requires_human_review": False
        }
    else:
        return {
            "risk_level": "HIGH",
            "action": "OPERATOR_REVIEW",
            "requires_human_review": True
        }


def _derive_incomplete_reason_code(monitoring_reason: str | None) -> str:
    if not monitoring_reason:
        return "SEGMENT_DATA_INCOMPLETE"
    text = monitoring_reason.lower()
    if "timestamp" in text:
        return "INVALID_TIMESTAMP"
    if "missing segments" in text:
        return "MISSING_SEGMENT_PREDICTIONS"
    if "duplicate" in text:
        return "DUPLICATE_SEGMENT_PREDICTIONS"
    if "data quality" in text or "weather" in text:
        return "INVALID_WEATHER_DATA_QUALITY"
    if "intervals" in text or "quantile" in text:
        return "INCONSISTENT_PREDICTION_INTERVALS"
    if "ampacity" in text:
        return "INVALID_AMPACITY_VALUES"
    return "SEGMENT_DATA_INCOMPLETE"


def evaluate_line_decision(
    line_rating_q05_a: float | None,
    static_rating_a: float | None,
    monitoring_status: str,
    uncertainty_ratio: float | None = None,
    planned_load_a: float | None = None,
    monitoring_reason: str | None = None,
    is_stale: bool = False
) -> dict:
    """
    Evaluate the simulated transmission line and generate a deterministic recommendation.
    """
    if is_stale:
        return {
            "risk_level": "UNCERTAIN",
            "utilization": None,
            "action": "FALLBACK_STATIC",
            "requires_human_review": True,
            "reason": "Weather or prediction data is stale.",
            "reason_code": "STALE_WEATHER_DATA",
            "uncertainty_status": "NOT_EVALUATED"
        }

    if monitoring_status == "INCOMPLETE":
        reason_text = (
            monitoring_reason
            if monitoring_reason
            else "Monitoring data is incomplete."
        )
        return {
            "risk_level": "UNCERTAIN",
            "utilization": None,
            "action": "FALLBACK_STATIC",
            "requires_human_review": True,
            "reason": reason_text,
            "reason_code": _derive_incomplete_reason_code(monitoring_reason),
            "uncertainty_status": "NOT_EVALUATED"
        }

    if monitoring_status != "VALID":
        raise ValueError("Invalid monitoring status.")

    try:
        utilization = calculate_utilization(
            line_rating_q05_a=line_rating_q05_a,
            static_rating_a=static_rating_a
        )
    except ValueError as exc:
        return {
            "risk_level": "UNCERTAIN",
            "utilization": None,
            "action": "FALLBACK_STATIC",
            "requires_human_review": True,
            "reason": str(exc),
            "reason_code": "INVALID_RATING_INPUTS",
            "uncertainty_status": "NOT_EVALUATED"
        }

    uncertainty_status = "NOT_PROVIDED"

    if uncertainty_ratio is not None:
        uncertainty_result = classify_uncertainty(uncertainty_ratio)
        uncertainty_status = uncertainty_result["uncertainty_status"]

        if uncertainty_status in ("HIGH", "INVALID"):
            return {
                "risk_level": "UNCERTAIN",
                "utilization": utilization,
                "action": "FALLBACK_STATIC",
                "requires_human_review": True,
                "reason": uncertainty_result["reason"],
                "reason_code": uncertainty_result["reason_code"],
                "uncertainty_status": uncertainty_status
            }

    if planned_load_a is not None:
        try:
            planned_load = float(planned_load_a)
        except (TypeError, ValueError):
            planned_load = float("nan")

        if not math.isfinite(planned_load) or planned_load < 0:
            return {
                "risk_level": "UNCERTAIN",
                "utilization": utilization,
                "action": "FALLBACK_STATIC",
                "requires_human_review": True,
                "reason": "Planned load data is invalid.",
                "reason_code": "PLANNED_LOAD_INVALID",
                "uncertainty_status": uncertainty_status
            }

        if planned_load > line_rating_q05_a:
            return {
                "risk_level": "HIGH",
                "utilization": utilization,
                "action": "OPERATOR_REVIEW",
                "requires_human_review": True,
                "reason": (
                    "Simulated planned load exceeds the conservative "
                    "dynamic line rating."
                ),
                "reason_code": "PLANNED_LOAD_EXCEEDS_DYNAMIC_RATING",
                "uncertainty_status": uncertainty_status
            }

    if static_rating_a > line_rating_q05_a:
        return {
            "risk_level": "HIGH",
            "utilization": utilization,
            "action": "OPERATOR_REVIEW",
            "requires_human_review": True,
            "reason": (
                "Static rating exceeds the conservative dynamic rating "
                "(thermal derating risk)."
            ),
            "reason_code": "STATIC_EXCEEDS_DYNAMIC_RATING",
            "uncertainty_status": uncertainty_status
        }

    if uncertainty_status == "ELEVATED":
        return {
            "risk_level": "MEDIUM",
            "utilization": utilization,
            "action": "KEEP_STATIC",
            "requires_human_review": False,
            "reason": "Prediction uncertainty is elevated.",
            "reason_code": "UNCERTAINTY_ELEVATED",
            "uncertainty_status": uncertainty_status
        }

    risk = classify_risk(utilization)

    if risk["risk_level"] == "LOW":
        reason = "Line utilization is within the low-risk threshold."
        reason_code = "DYNAMIC_RATING_ACCEPTABLE"
    elif risk["risk_level"] == "MEDIUM":
        reason = "Line utilization is within the medium-risk threshold."
        reason_code = "UTILIZATION_MEDIUM"
    else:
        reason = "Line utilization exceeds the high-risk threshold."
        reason_code = "UTILIZATION_HIGH"

    return {
        "risk_level": risk["risk_level"],
        "utilization": utilization,
        "action": risk["action"],
        "requires_human_review": risk["requires_human_review"],
        "reason": reason,
        "reason_code": reason_code,
        "uncertainty_status": uncertainty_status
    }