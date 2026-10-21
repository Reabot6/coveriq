"""Turn ranked catalogue candidates into one simulated next step per customer."""

from __future__ import annotations

from typing import Any


def build_next_best_actions(
    customer_profiles: list[dict[str, Any]],
    offers: list[dict[str, Any]],
    evaluation: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Choose at most one offer to show each opted-in customer for review."""
    offers_by_id = {offer["offer_id"]: offer for offer in offers}
    actions = {}

    for customer in customer_profiles:
        customer_id = customer["customer_id"]
        ranked = evaluation["ranked_candidates"].get(customer_id, [])

        if customer.get("consent_given") is not True:
            actions[customer_id] = {
                "action": "suppress",
                "recommended_offer": None,
                "reason": "Consent is off, so no customer activity was analyzed.",
                "full_ranking": [],
                "delivery": "not_sent_demo_only",
            }
            continue

        if not ranked:
            actions[customer_id] = {
                "action": "suppress",
                "recommended_offer": None,
                "reason": "No catalogue offer passed the current fit checks.",
                "full_ranking": [],
                "delivery": "not_sent_demo_only",
            }
            continue

        full_ranking = [
            {
                "rank": candidate["rank"],
                "offer_id": candidate["offer_id"],
                "name": offers_by_id[candidate["offer_id"]]["name"],
                "fit_points": candidate["fit_points"],
                "model_score": candidate["model_score"],
                "matched_signals": candidate["matched_signals"],
            }
            for candidate in ranked
        ]
        best = ranked[0]
        offer = offers_by_id[best["offer_id"]]
        actions[customer_id] = {
            "action": "show_for_review",
            "recommended_offer": {
                "offer_id": offer["offer_id"],
                "name": offer["name"],
                "provider": offer["provider"],
                "message": offer["message"],
                "fit_points": best["fit_points"],
                "model_score": best["model_score"],
                "matched_signals": best["matched_signals"],
            },
            "reason": (
                "This offer had the highest simulated-response score among candidates "
                "that passed the product-fit checks."
            ),
            "full_ranking": full_ranking,
            "delivery": "not_sent_demo_only",
        }

    return actions
