"""Generate transaction histories and premium-sustainability training labels."""

from __future__ import annotations

import random
from calendar import monthrange
from datetime import date
from typing import Any

import pandas as pd

from affordability import add_premium_features, premium_schedule
from features import calculate_features


HISTORY_MONTHS = 36
FORECAST_MONTHS = 12
LABEL_COLUMN = "premium_sustainable_next_12_months"
FEATURE_COLUMNS = [
    "income_presence_rate",
    "income_variability",
    "typical_monthly_income",
    "current_balance",
    "typical_monthly_recurring_outflow",
    "mean_monthly_surplus",
    "baseline_monthly_surplus",
    "lower_quartile_surplus",
    "positive_surplus_rate",
    "surplus_variability",
    "obligation_to_income_ratio",
    "liquidity_buffer_months",
    "balance_growth_months_of_income",
    "debit_count_per_month",
    "early_month_outflow_share",
    "history_months",
    "premium_to_income_ratio",
    "premium_payment_to_balance_ratio",
    "annual_payment",
]

PATTERNS = {
    # Weight, spend ratio, shock chance, historical missed-pay chance,
    # and future missed-pay chance. These are simulator assumptions.
    "steady": (0.30, 0.68, 0.05, 0.01, 0.04),
    "recent_raise": (0.23, 0.66, 0.06, 0.02, 0.08),
    "gradual_growth": (0.18, 0.74, 0.07, 0.03, 0.08),
    "variable_income": (0.17, 0.82, 0.15, 0.10, 0.15),
    "declining_income": (0.12, 0.88, 0.09, 0.04, 0.12),
}


def _month_for(index: int) -> tuple[int, int, int]:
    year, zero_based_month = divmod(2024 * 12 + index, 12)
    month = zero_based_month + 1
    return year, month, monthrange(year, month)[1]


def _transaction(
    year: int,
    month: int,
    day: int,
    direction: str,
    category: str,
    amount: float,
) -> dict[str, Any]:
    days_in_month = monthrange(year, month)[1]
    return {
        "date": date(year, month, min(day, days_in_month)).isoformat(),
        "direction": direction,
        "category": category,
        "amount": round(max(0.0, amount), 2),
    }


def _customer_history(
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[float]]:
    """Return observed transactions and the following year's net cash flows."""
    pattern = rng.choices(
        list(PATTERNS), weights=[values[0] for values in PATTERNS.values()]
    )[0]
    _, spend_ratio, shock_chance, past_gap_chance, future_gap_chance = PATTERNS[pattern]
    base_income = rng.uniform(300_000, 1_500_000)
    base_obligations_ratio = rng.uniform(0.10, 0.24)
    raise_age = rng.choice((3, 6, 9, 12))
    raise_start = HISTORY_MONTHS - raise_age
    raise_multiplier = rng.uniform(1.3, 2.1)
    raise_continues = True
    if pattern == "recent_raise":
        reversion_chance = {3: 0.35, 6: 0.22, 9: 0.10, 12: 0.04}[raise_age]
        raise_continues = rng.random() >= reversion_chance

    transactions: list[dict[str, Any]] = []
    future_net_flows: list[float] = []

    for month_index in range(HISTORY_MONTHS + FORECAST_MONTHS):
        year, month, _ = _month_for(month_index)
        is_future = month_index >= HISTORY_MONTHS

        if pattern == "recent_raise" and month_index >= raise_start:
            income_level = base_income * raise_multiplier
            if is_future and not raise_continues:
                income_level = base_income
        elif pattern == "gradual_growth":
            growth_rate = rng.uniform(0.003, 0.014)
            income_level = base_income * (1 + month_index * growth_rate)
        elif pattern == "variable_income":
            income_level = base_income * rng.uniform(0.50, 1.50)
        elif pattern == "declining_income" and month_index >= HISTORY_MONTHS - 9:
            income_level = base_income * rng.uniform(0.50, 0.90)
        else:
            income_level = base_income * rng.uniform(0.94, 1.06)

        gap_chance = future_gap_chance if is_future else past_gap_chance
        missed_income = rng.random() < gap_chance
        income = 0.0 if missed_income else max(0.0, income_level)
        total_outflow = max(
            60_000,
            income_level * rng.uniform(spend_ratio - 0.12, spend_ratio + 0.12)
            + base_income * rng.uniform(-0.08, 0.10),
        )
        if rng.random() < shock_chance:
            total_outflow *= rng.uniform(1.2, 1.8)

        obligations = min(
            total_outflow * 0.65,
            base_income * rng.uniform(
                max(0.04, base_obligations_ratio - 0.04),
                base_obligations_ratio + 0.04,
            ),
        )
        variable_outflow = max(0.0, total_outflow - obligations)
        housing = obligations * 0.62
        utilities = obligations * 0.16
        loan_payment = obligations - housing - utilities
        food = variable_outflow * rng.uniform(0.38, 0.52)
        transport = variable_outflow * rng.uniform(0.14, 0.24)
        other = max(0.0, variable_outflow - food - transport)
        transfer = rng.uniform(5_000, 40_000) if rng.random() < 0.25 else 0.0
        net_flow = income - total_outflow - transfer

        if is_future:
            future_net_flows.append(net_flow)
            continue

        if income:
            transactions.append(
                _transaction(year, month, rng.randint(22, 28), "credit", "salary", income)
            )

        for day, category, amount in (
            (1, "housing", housing),
            (7, "utilities", utilities),
            (14, "loan_payment", loan_payment),
        ):
            if amount > 0:
                transactions.append(_transaction(year, month, day, "debit", category, amount))

        for category, amount, low, high in (
            ("food", food, 2, 5),
            ("transport", transport, 2, 6),
            ("other", other, 1, 5),
        ):
            transaction_count = rng.randint(low, high)
            weights = [rng.uniform(0.2, 1.0) for _ in range(transaction_count)]
            weight_total = sum(weights)
            for weight in weights:
                transactions.append(
                    _transaction(
                        year,
                        month,
                        rng.randint(1, 28),
                        "debit",
                        category,
                        amount * weight / weight_total,
                    )
                )

        if transfer:
            transactions.append(
                _transaction(year, month, 28, "debit", "savings_transfer", transfer)
            )

    return transactions, future_net_flows


def create_training_data(n_samples: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Create one profile/premium example per row and label from future balances."""
    rng = random.Random(seed)
    rows = []
    for customer_id in range(n_samples):
        transactions, future_net_flows = _customer_history(rng)
        opening_balance = rng.uniform(100_000, 1_500_000)
        financial_features = calculate_features(
            transactions,
            opening_balance=opening_balance,
        )

        monthly_premium = financial_features["typical_monthly_income"] * rng.choice(
            (0.02, 0.04, 0.07, 0.10, 0.15, 0.22, 0.30)
        )
        frequency = rng.choice(("monthly", "annual"))
        premium_amount = monthly_premium if frequency == "monthly" else monthly_premium * 12
        model_features = add_premium_features(
            financial_features,
            premium_amount,
            frequency,
        )

        projected_balance = financial_features["current_balance"]
        reserve_floor = financial_features["typical_monthly_recurring_outflow"]
        payment_schedule = premium_schedule(premium_amount, frequency)
        # A premium is sustainable in this scenario only if one month's typical
        # recurring commitments remain available throughout the forecast year.
        projected_balances = []
        projected_balance = financial_features["current_balance"]
        for cash_flow, payment in zip(future_net_flows, payment_schedule):
            projected_balance += cash_flow - payment
            projected_balances.append(projected_balance)
        label = int(all(balance >= reserve_floor for balance in projected_balances))

        rows.append(
            {
                "customer_id": customer_id,
                **{name: model_features[name] for name in FEATURE_COLUMNS},
                "premium_amount": round(premium_amount, 2),
                "payment_frequency": frequency,
                LABEL_COLUMN: label,
            }
        )
    return pd.DataFrame(rows)
