"""Join fictional customer details to activity under a consent check."""

from __future__ import annotations

from typing import Any

from profile_features import summarize_activity


def build_customer_profiles(
    customers: list[dict[str, Any]],
    activity_by_customer: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build one demo profile per ID, summarizing activity only after opt-in."""
    profiles = []
    seen_ids = set()

    for customer in customers:
        customer_id = customer.get("customer_id")
        if not customer_id:
            raise ValueError("Every demo customer needs a customer ID")
        if customer_id in seen_ids:
            raise ValueError(f"Duplicate demo customer ID: {customer_id}")
        seen_ids.add(customer_id)

        has_consent = customer.get("consent_given") is True

        profile = {
            "customer_id": customer_id,
            "name": customer["name"],
            "customer_type": customer["customer_type"],
            "stated_needs": list(customer.get("stated_needs", [])),
            "declared_assets": list(customer.get("declared_assets", [])),
            "consent_given": has_consent,
            "activity_summary": None,
        }

        if has_consent:
            if customer_id not in activity_by_customer:
                raise ValueError(f"No demo activity found for {customer_id}")

            activity = activity_by_customer[customer_id]
            profile["activity_summary"] = summarize_activity(
                activity["transactions"],
                opening_balance=activity.get("opening_balance", 0.0),
            )

        profiles.append(profile)

    return profiles
