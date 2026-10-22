"""Combine catalogue retrieval and premium sustainability scores."""

from __future__ import annotations

import pandas as pd

from affordability import add_premium_features
from catalogue import retrieve_offers
from products import PRODUCTS
from synthetic_data import FEATURE_COLUMNS

MIN_SUSTAINABILITY_ESTIMATE = 0.60
MIN_MONTHS_AT_NEW_INCOME_LEVEL = 6


def recommend_catalogue(
    financial_features: dict[str, float],
    premiums: dict[str, float],
    payment_frequency: str,
    model,
    consent: bool,
    selected_interests: list[str] | None = None,
) -> list[dict]:
    """Score every catalogue item and rank requested offers by fit and affordability."""
    if not consent or not selected_interests:
        return []

    retrieved = [
        offer for offer in retrieve_offers(selected_interests)
        if offer["selected"]
    ]
    ranked = []
    for offer in retrieved:
        product_key = offer["key"]
        model_features = add_premium_features(
            financial_features, premiums[product_key], payment_frequency
        )
        model_input = pd.DataFrame(
            [[model_features[column] for column in FEATURE_COLUMNS]],
            columns=FEATURE_COLUMNS,
        )
        positive_class = list(model.classes_).index(1)
        probability = float(model.predict_proba(model_input)[0][positive_class])
        monthly_premium = premiums[product_key] / 12 if payment_frequency == "annual" else premiums[product_key]
        baseline_surplus = financial_features["baseline_monthly_surplus"]
        relies_on_new_income = (
            monthly_premium > max(baseline_surplus, 0)
            and financial_features["recent_income_change_pct"] >= 0.25
        )

        if financial_features["mean_monthly_surplus"] <= 0:
            action = "hold"
            reason = "Recent outflows use all observed credits, leaving no average monthly surplus."
        elif (
            relies_on_new_income
            and financial_features["months_since_income_change"] < MIN_MONTHS_AT_NEW_INCOME_LEVEL
        ):
            action = "wait"
            reason = (
                f"This ₦{monthly_premium:,.0f}/month premium is above the prior-income surplus "
                f"of ₦{baseline_surplus:,.0f}/month, so it relies on the recent raise. "
                f"Recent money left is ₦{financial_features['mean_monthly_surplus']:,.0f}/month, "
                f"but the higher salary has appeared for only "
                f"{int(financial_features['months_since_income_change'])} months."
            )
        elif probability < MIN_SUSTAINABILITY_ESTIMATE:
            action = "wait"
            reason = (
                f"The premium is ₦{monthly_premium:,.0f}/month and recent money left after "
                f"spending is ₦{financial_features['mean_monthly_surplus']:,.0f}/month. "
                f"The model estimates a {probability:.0%} chance of maintaining a one-month "
                "recurring-bills reserve for the next year."
            )
        else:
            action = "recommend"
            reason = (
                f"The premium is ₦{monthly_premium:,.0f}/month against an observed monthly "
                f"surplus of ₦{financial_features['mean_monthly_surplus']:,.0f}; "
                f"the model estimates {probability:.0%} sustainability."
            )

        # The customer-interest retrieval and financial estimate both affect rank.
        combined_score = 0.45 * offer["relevance_score"] + 0.55 * probability
        ranked.append({
            **offer,
            "action": action,
            "probability_sustainable": probability,
            "premium_amount": premiums[product_key],
            "payment_frequency": payment_frequency,
            "monthly_premium": monthly_premium,
            "baseline_monthly_surplus": baseline_surplus,
            "relies_on_new_income": relies_on_new_income,
            "reason": reason,
            "combined_score": combined_score,
        })

    actionable = [offer for offer in ranked if offer["action"] == "recommend"]
    if actionable:
        best = max(actionable, key=lambda offer: offer["combined_score"])
        best["action"] = "top_match"
        best["reason"] = "Top catalogue match. " + best["reason"]
        for offer in actionable:
            if offer is not best:
                offer["action"] = "also_fits"
    action_order = {"top_match": 0, "also_fits": 1, "wait": 2, "hold": 3}
    return sorted(
        ranked,
        key=lambda offer: (action_order[offer["action"]], -offer["combined_score"]),
    )
