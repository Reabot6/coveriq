"""Prototype offer ranker trained on simulated customer-offer interactions."""

from __future__ import annotations

import random
import math
from functools import lru_cache
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


OFFER_FAMILIES = (
    "savings_and_protection",
    "education_savings",
    "motor_insurance",
    "marine_hull",
    "marine_cargo",
)
FEATURE_COLUMNS = [
    "history_months",
    "average_monthly_credits_log",
    "surplus_ratio",
    "positive_cashflow_month_share",
    "cashflow_stability",
    "balance_change_6m_ratio",
    "need_match",
    "asset_match",
    "activity_match",
    *[f"offer_{family}" for family in OFFER_FAMILIES],
]


def _simulate_interactions(sample_count: int = 8000, seed: int = 19) -> pd.DataFrame:
    """Make fictional impression/response events for a prototype training set."""
    rng = random.Random(seed)
    rows = []
    for _ in range(sample_count):
        family = rng.choice(OFFER_FAMILIES)
        need_match = int(rng.random() < 0.25)
        asset_match = int(rng.random() < 0.30)
        activity_match = int(rng.random() < 0.40)
        surplus_ratio = rng.uniform(-0.35, 0.55)
        positive_month_share = rng.random()
        stability = rng.random()
        balance_change = rng.uniform(-0.4, 1.0)
        history_months = rng.randint(3, 36)
        average_credits = rng.uniform(80_000, 8_000_000)

        # These assumptions generate simulated outcomes; they are not learned
        # from Zenith customers and must be replaced with observed events.
        logit = (
            -2.2
            + 2.2 * need_match
            + 1.0 * asset_match
            + 0.8 * activity_match
            + 1.4 * max(surplus_ratio, 0)
            + 0.6 * positive_month_share
            + 0.4 * stability
            + 0.2 * max(balance_change, 0)
            + min(history_months, 24) / 120
        )
        response_chance = 1 / (1 + math.exp(-logit))

        row = {
            "history_months": history_months,
            "average_monthly_credits_log": math.log1p(average_credits),
            "surplus_ratio": surplus_ratio,
            "positive_cashflow_month_share": positive_month_share,
            "cashflow_stability": stability,
            "balance_change_6m_ratio": balance_change,
            "need_match": need_match,
            "asset_match": asset_match,
            "activity_match": activity_match,
            "response": int(rng.random() < response_chance),
        }
        row.update({f"offer_{item}": int(item == family) for item in OFFER_FAMILIES})
        rows.append(row)

    return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def get_matching_model() -> tuple[RandomForestClassifier, dict[str, Any]]:
    """Train once per app process and return the model plus synthetic metrics."""
    data = _simulate_interactions()
    train, test = train_test_split(
        data,
        test_size=0.25,
        random_state=19,
        stratify=data["response"],
    )
    model = RandomForestClassifier(
        n_estimators=180,
        min_samples_leaf=8,
        max_features=0.85,
        class_weight="balanced_subsample",
        random_state=19,
        n_jobs=-1,
    )
    model.fit(train[FEATURE_COLUMNS], train["response"])
    probabilities = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    report = {
        "training_events": len(train),
        "held_out_events": len(test),
        "held_out_auc": float(roc_auc_score(test["response"], probabilities)),
        "training_source": "simulated impressions and responses",
    }
    return model, report


def build_pair_features(
    customer: dict[str, Any],
    offer: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, float]:
    """Combine permitted financial summary and offer-fit signals for scoring."""
    summary = customer.get("activity_summary") or {}
    categories = summary.get("credit_category_shares", {})
    family = offer.get("category", "")
    features = {
        "history_months": float(summary.get("history_months", 0)),
        "average_monthly_credits_log": math.log1p(
            max(0.0, float(summary.get("average_monthly_credits", 0)))
        ),
        "surplus_ratio": float(summary.get("recent_surplus_ratio", 0)),
        "positive_cashflow_month_share": float(
            summary.get("positive_cashflow_month_share", 0)
        ),
        "cashflow_stability": float(summary.get("cashflow_stability", 0)),
        "balance_change_6m_ratio": float(
            summary.get("balance_change_6m_ratio") or 0
        ),
        "need_match": float(
            any(signal.startswith("selected need:") for signal in evaluation["matched_signals"])
        ),
        "asset_match": float(
            any(signal.startswith("declared asset:") for signal in evaluation["matched_signals"])
        ),
        "activity_match": float(
            any(
                signal.startswith("observed credit category:")
                for signal in evaluation["matched_signals"]
            )
        ),
    }
    features.update(
        {f"offer_{item}": float(item == family) for item in OFFER_FAMILIES}
    )
    return features


def score_offer_pair(
    model: RandomForestClassifier,
    customer: dict[str, Any],
    offer: dict[str, Any],
    evaluation: dict[str, Any],
) -> float:
    """Return the model's simulated-response score for one candidate pair."""
    features = build_pair_features(customer, offer, evaluation)
    row = pd.DataFrame(
        [[features[name] for name in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    )
    positive_index = list(model.classes_).index(1)
    return float(model.predict_proba(row)[0][positive_index])
