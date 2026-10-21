"""Apply consent and timing safeguards to the synthetic cash-flow estimate."""

from __future__ import annotations

import pandas as pd

from products import GOAL_LABELS, PRODUCTS
from synthetic_data import FEATURE_COLUMNS

MIN_SURPLUS_PERSISTENCE_ESTIMATE = 0.55


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

    if features["average_monthly_surplus"] <= 0:
        action = "suppress"
        reason = "Recent credits do not exceed recorded debits in this fictional history."
    elif features["months_since_income_change"] < 6:
        action = "wait"
        reason = "The higher income level is recent; wait for more observed history."
    elif probability < MIN_SURPLUS_PERSISTENCE_ESTIMATE:
        action = "wait"
        reason = (
            "The synthetic model estimate is below the demo threshold for showing this product."
        )
    else:
        action = "recommend"
        reason = (
            "The observed pattern and synthetic forward cash-flow estimate support showing "
            "the selected product information for customer review."
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
        "model_target": "Positive monthly net cash flow in at least five of six future synthetic months",
    }
