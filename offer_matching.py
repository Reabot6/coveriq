"""Check whether an offer is a reasonable candidate for ranking."""

from __future__ import annotations

from typing import Any


def evaluate_offer_match(
    customer: dict[str, Any],
    offer: dict[str, Any],
) -> dict[str, Any]:
    """Apply catalogue fit checks without claiming product eligibility."""
    offer_id = offer["offer_id"]
    if customer.get("consent_given") is not True:
        return {
            "customer_id": customer["customer_id"],
            "offer_id": offer_id,
            "status": "suppressed",
            "matched_signals": [],
            "reasons": ["Consent is off; customer signals were not evaluated."],
        }

    rules = offer.get("match_rules", {})
    if not rules:
        raise ValueError(f"Offer {offer_id} is missing machine-readable match rules")
    allowed_rules = {
        "required_need",
        "required_asset",
        "required_credit_category",
        "minimum_positive_cashflow_month_share",
    }
    unknown_rules = set(rules) - allowed_rules
    if unknown_rules:
        unknown_names = ", ".join(sorted(unknown_rules))
        raise ValueError(f"Offer {offer_id} has unsupported match rules: {unknown_names}")

    needs = set(customer.get("stated_needs", []))
    assets = set(customer.get("declared_assets", []))
    summary = customer.get("activity_summary") or {}
    matched_signals = []
    reasons = []

    required_need = rules.get("required_need")
    if required_need:
        if required_need in needs:
            matched_signals.append(f"selected need: {required_need}")
        else:
            reasons.append(f"The customer has not selected the {required_need} need.")

    required_asset = rules.get("required_asset")
    if required_asset:
        if required_asset in assets:
            matched_signals.append(f"declared asset: {required_asset}")
        else:
            reasons.append(f"No matching declared asset ({required_asset}) is on the profile.")

    required_category = rules.get("required_credit_category")
    if required_category:
        category_shares = summary.get("credit_category_shares", {})
        if category_shares.get(required_category, 0) > 0:
            matched_signals.append(f"observed credit category: {required_category}")
        else:
            reasons.append(f"No {required_category} credits appear in the permitted history.")

    minimum_cashflow_share = rules.get("minimum_positive_cashflow_month_share")
    if minimum_cashflow_share is not None:
        minimum_cashflow_share = float(minimum_cashflow_share)
        if not 0 <= minimum_cashflow_share <= 1:
            raise ValueError("minimum positive cash-flow share must be between 0 and 1")
        observed_share = summary.get("positive_cashflow_month_share")
        if observed_share is not None and observed_share >= minimum_cashflow_share:
            matched_signals.append("positive cash flow in most observed months")
        else:
            reasons.append("The observed positive cash-flow pattern does not meet this demo check.")

    is_candidate = not reasons
    return {
        "customer_id": customer["customer_id"],
        "offer_id": offer_id,
        "status": "candidate" if is_candidate else "not_candidate",
        "matched_signals": matched_signals,
        "reasons": (
            ["Passed the catalogue fit checks; this is not an eligibility decision."]
            if is_candidate
            else reasons
        ),
    }
