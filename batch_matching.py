"""Run the catalogue fit checks across a set of demo customer profiles."""

from __future__ import annotations

from collections import Counter
from typing import Any

from offer_matching import evaluate_offer_match


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
    counts = Counter(result["status"] for result in evaluations)

    return {
        "total_pairs": len(evaluations),
        "counts": {
            "candidate": counts["candidate"],
            "not_candidate": counts["not_candidate"],
            "suppressed": counts["suppressed"],
        },
        "evaluations": evaluations,
    }
