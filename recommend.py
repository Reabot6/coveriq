"""Apply consent and timing safeguards to the synthetic cash-flow estimate."""

from __future__ import annotations

import pandas as pd

from products import GOAL_LABELS, PRODUCTS
from synthetic_data import FEATURE_COLUMNS

MIN_SURPLUS_PERSISTENCE_ESTIMATE = 0.55
MIN_MONTHS_AT_NEW_INCOME_LEVEL = 6


def recommend(
    features: dict[str, float],
    model,
    consent: bool,
    preferred_goals: list[str] | None = None,
) -> dict:
    """Return recommend, wait, or suppress with traceable reasons."""
    if not consent:
        return {
            "action": "suppress",
            "message": "No analysis or offer is produced without the demo opt-in.",
            "probability_surplus_persists": None,
            "products": [],
            "reason": "Consent was not provided.",
        }

    if not preferred_goals:
        return {
            "action": "suppress",
            "message": "No offer is produced because the customer selected no products.",
            "probability_surplus_persists": None,
            "products": [],
            "reason": "The customer did not ask to consider a product.",
        }

    goal_by_label = {label: goal for goal, label in GOAL_LABELS.items()}
    selected_goals = [
        goal_by_label[label]
        for label in preferred_goals
        if label in goal_by_label
    ]
    selected_products = [
        product for product in PRODUCTS.values() if product["goal"] in selected_goals
    ]
    if not selected_products:
        raise ValueError("Choose at least one customer goal from the demo catalogue")

    model_input = pd.DataFrame(
        [[features[column] for column in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    )
    positive_class_index = list(model.classes_).index(1)
    probability = float(model.predict_proba(model_input)[0][positive_class_index])

    if features["mean_monthly_surplus"] <= 0:
        action = "suppress"
        reason = "Recent monthly outflows meet or exceed account credits, so CoverIQ holds back the offer."
    elif (
        features["recent_income_change_pct"] >= 0.25
        and features["months_since_income_change"] < MIN_MONTHS_AT_NEW_INCOME_LEVEL
    ):
        action = "wait"
        months = int(features["months_since_income_change"])
        reason = (
            f"Income is {features['recent_income_change_pct']:.0%} above its earlier level, "
            f"but has held there for only {months} months. CoverIQ is waiting for more history."
        )
    elif probability < MIN_SURPLUS_PERSISTENCE_ESTIMATE:
        action = "wait"
        reason = (
            f"The model estimates a {probability:.0%} chance of positive cash flow in each "
            f"of the next three months, below the {MIN_SURPLUS_PERSISTENCE_ESTIMATE:.0%} "
            "review threshold."
        )
    else:
        action = "recommend"
        reason = (
            f"Income has held at the higher level for "
            f"{int(features['months_since_income_change'])} months, and the model estimates "
            f"a {probability:.0%} chance of positive cash flow in each of the next three months."
        )

    return {
        "action": action,
        "message": {
            "recommend": "Show product information for the customer to review.",
            "wait": "Wait for more evidence, then review again.",
            "suppress": "Do not show an offer in this scenario.",
        }[action],
        "probability_surplus_persists": probability,
        "products": [product["name"] for product in selected_products],
        "matched_goals": [GOAL_LABELS[goal] for goal in selected_goals],
        "reason": reason,
        "model_target": "Positive net cash flow in each of the next three simulated months",
    }
