
def calculate_utilization(
    line_rating_q05_a: float,
    static_rating_a: float
) -> float:
    """
    Calculate the utilization ratio of the transmission line.

    The ratio compares the reference static ampacity against
    the conservative dynamic ampacity (Q05).

    A higher ratio indicates that the available dynamic capacity
    is closer to the static reference rating.

    Args:
        line_rating_q05_a: Minimum conservative ampacity across all
            monitored line segments, in amperes.
        static_rating_a: Reference static ampacity, in amperes.

    Returns:
        Utilization ratio as a float.

    Raises:
        ValueError: If either ampacity is missing or non-positive.

    Note:
        This is a simulation indicator, not a measurement of
        actual electrical current flowing through the line.
    """

    # Validate input values before performing the calculation.
    if line_rating_q05_a is None or static_rating_a is None:
        raise ValueError("Ampacity values cannot be None.")

    if line_rating_q05_a <= 0 or static_rating_a <= 0:
        raise ValueError("Ampacity values must be positive.")

    # Calculate the ratio between static and dynamic ampacity.
    return static_rating_a / line_rating_q05_a


def classify_risk(utilization: float) -> dict:
    """
    Classify the operational risk based on the utilization ratio.

    Three risk categories are used:
        LOW: Utilization <= 0.80
        MEDIUM: 0.80 < Utilization <= 0.95
        HIGH: Utilization > 0.95

    Args:
        utilization: Calculated utilization ratio.

    Returns:
        Dictionary containing:
            risk_level: LOW, MEDIUM, or HIGH.
            action: Recommended monitoring or review action.
            requires_human_review: Whether operator review is required.

    Raises:
        ValueError: If utilization is negative.

    Note:
        Thresholds are initial project assumptions and are not
        certified electrical operating limits.
    """

    if utilization < 0:
        raise ValueError("Utilization cannot be negative.")

    # Low utilization: continue normal monitoring.
    if utilization <= 0.80:
        return {
            "risk_level": "LOW",
            "action": "NORMAL_MONITORING",
            "requires_human_review": False
        }

    # Medium utilization: recommend more frequent monitoring.
    elif utilization <= 0.95:
        return {
            "risk_level": "MEDIUM",
            "action": "INCREASE_MONITORING",
            "requires_human_review": False
        }

    # High utilization: request human operator review.
    else:
        return {
            "risk_level": "HIGH",
            "action": "REQUEST_OPERATOR_REVIEW",
            "requires_human_review": True
        }


def evaluate_line_decision(
    line_rating_q05_a: float | None,
    static_rating_a: float,
    monitoring_status: str
) -> dict:
    """
    Evaluate the transmission line and generate a risk-based decision.

    This function connects the multi-point monitoring module
    with the risk classification logic.

    If monitoring data is incomplete, the function avoids calculating
    utilization and returns an UNKNOWN risk level.

    Args:
        line_rating_q05_a: Conservative ampacity of the entire line,
            calculated by the multi-point monitoring module.
        static_rating_a: Reference static ampacity, in amperes.
        monitoring_status: Monitoring result status (VALID or INCOMPLETE).

    Returns:
        Dictionary containing:
            risk_level: LOW, MEDIUM, HIGH, or UNKNOWN.
            utilization: Calculated utilization ratio, or None.
            action: Recommended action.
            requires_human_review: Whether operator review is required.
            reason: Explanation of the decision.

    Raises:
        ValueError: If monitoring status is invalid or ampacity
            inputs are invalid.

    Note:
        The function generates recommendations only. It does not
        perform physical grid control or automatically change
        transmission line operating conditions.
    """

    # Incomplete monitoring data must not produce a risk calculation.
    if monitoring_status == "INCOMPLETE":
        return {
            "risk_level": "UNKNOWN",
            "utilization": None,
            "action": "REQUEST_DATA_VALIDATION",
            "requires_human_review": True,
            "reason": "Monitoring data is incomplete."
        }

    # Reject unexpected monitoring statuses.
    if monitoring_status != "VALID":
        raise ValueError("Invalid monitoring status.")

    # Calculate utilization using the conservative line ampacity.
    utilization = calculate_utilization(
        line_rating_q05_a=line_rating_q05_a,
        static_rating_a=static_rating_a
    )

    # Convert utilization into a risk category and recommendation.
    risk = classify_risk(utilization)

    # Generate a human-readable explanation for the decision.
    if risk["risk_level"] == "LOW":
        reason = "Line utilization is within the low-risk threshold."
    elif risk["risk_level"] == "MEDIUM":
        reason = "Line utilization requires increased monitoring."
    else:
        reason = "Line utilization exceeds the high-risk threshold."

    # Return the complete decision result for the main agent.
    return {
        "risk_level": risk["risk_level"],
        "utilization": utilization,
        "action": risk["action"],
        "requires_human_review": risk["requires_human_review"],
        "reason": reason
    }
