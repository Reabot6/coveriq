"""Run the catalogue fit checks across a set of demo customer profiles."""

from __future__ import annotations

from collections import Counter
from typing import Any

from decision_layer import build_next_best_actions
from matching_model import get_matching_model, score_offer_pair
from offer_matching import evaluate_offer_match
from ranking import rank_candidates


def evaluate_catalogue(
    customer_profiles: list[dict[str, Any]],
    offers: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate each customer against every offer and summarize the outcomes."""
    evaluations = [
        evaluate_offer_match(customer, offer)
        for customer in customer_profiles
        for offer in offers
    ]
    matching_model, model_report = get_matching_model()
    customer_by_id = {profile["customer_id"]: profile for profile in customer_profiles}
    offer_by_id = {offer["offer_id"]: offer for offer in offers}
    for result in evaluations:
        if result["status"] == "candidate":
            result["model_score"] = score_offer_pair(
                matching_model,
                customer_by_id[result["customer_id"]],
                offer_by_id[result["offer_id"]],
                result,
            )
    counts = Counter(result["status"] for result in evaluations)

    result = {
        "total_pairs": len(evaluations),
        "counts": {
            "candidate": counts["candidate"],
            "not_candidate": counts["not_candidate"],
            "suppressed": counts["suppressed"],
        },
        "evaluations": evaluations,
        "ranked_candidates": rank_candidates(evaluations),
        "model_report": model_report,
    }
    result["next_best_actions"] = build_next_best_actions(
        customer_profiles,
        offers,
        result,
    )
    return result
