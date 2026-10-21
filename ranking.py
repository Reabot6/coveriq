"""Order catalogue candidates by the type of evidence supporting each match."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


EVIDENCE_POINTS = {
    "customer_need": 4,
    "declared_asset": 3,
    "cash_flow": 2,
    "observed_activity": 1,
}


def rank_candidates(
    evaluations: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Sort candidates per customer and keep the point breakdown visible."""
    by_customer: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for result in evaluations:
        if result["status"] != "candidate":
            continue

        evidence_types = result["evidence_types"]
        point_breakdown = {
            evidence: EVIDENCE_POINTS[evidence]
            for evidence in evidence_types
        }
        by_customer[result["customer_id"]].append(
            {
                "offer_id": result["offer_id"],
                "fit_points": sum(point_breakdown.values()),
                "point_breakdown": point_breakdown,
                "model_score": result["model_score"],
                "matched_signals": result["matched_signals"],
            }
        )

    ranked = {}
    for customer_id, candidates in by_customer.items():
        candidates.sort(key=lambda item: (-item["model_score"], item["offer_id"]))
        for position, candidate in enumerate(candidates, start=1):
            candidate["rank"] = position
        ranked[customer_id] = candidates

    return ranked
