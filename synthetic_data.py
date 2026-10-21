"""Generate synthetic histories and forward cash-flow persistence labels."""

from __future__ import annotations

import random
from calendar import monthrange
from datetime import date

import pandas as pd

from features import calculate_features


HISTORY_MONTHS = 36
FORECAST_MONTHS = 6
FEATURE_COLUMNS = [
    "income_stability",
    "recent_income_change_pct",
    "months_since_income_change",
    "average_monthly_surplus",
    "surplus_ratio",
    "balance_growth_ratio",
    "history_months",
]


def _month_for(index: int) -> tuple[int, int, int]:
    absolute = 2024 * 12 + index
    year, month_zero = divmod(absolute, 12)
    month = month_zero + 1
    return year, month, monthrange(year, month)[1]


def _profile(rng: random.Random) -> tuple[list[dict], list[dict]]:
    base_salary = rng.randint(300_000, 1_500_000)
    pattern = rng.choices(
        ["steady", "recent_raise", "gradual", "volatile", "declining"],
        weights=[0.28, 0.23, 0.18, 0.18, 0.13],
    )[0]
    raise_age = rng.choice((3, 6, 9, 12))
    raise_at = HISTORY_MONTHS - raise_age
    base_cost_ratio = rng.uniform(0.52, 1.02)
    past_transactions: list[dict] = []
    future_months: list[tuple[float, float]] = []

    for month_index in range(HISTORY_MONTHS + FORECAST_MONTHS):
        if pattern == "recent_raise" and month_index >= raise_at:
            salary_level = base_salary * rng.uniform(1.3, 1.9)
        elif pattern == "gradual":
            salary_level = base_salary * (1 + month_index * rng.uniform(0.003, 0.014))
        elif pattern == "volatile":
            salary_level = base_salary * rng.uniform(0.55, 1.45)
        elif pattern == "declining" and month_index >= HISTORY_MONTHS - 9:
            salary_level = base_salary * rng.uniform(0.5, 0.9)
        else:
            salary_level = base_salary * rng.uniform(0.94, 1.06)

        salary = float(max(0, int(salary_level)))
        # Future income interruptions are sampled independently of the past.
        if month_index >= HISTORY_MONTHS and rng.random() < 0.08:
            salary = 0.0

        costs = max(80_000, salary * base_cost_ratio + base_salary * rng.uniform(-0.08, 0.12))
        costs = float(int(costs))

        if month_index < HISTORY_MONTHS:
            year, month, last_day = _month_for(month_index)
            if salary > 0:
                past_transactions.append(
                    {
                        "date": date(year, month, 25).isoformat(),
                        "direction": "credit",
                        "category": "salary",
                        "amount": int(salary),
                    }
                )
            past_transactions.append(
                {
                    "date": date(year, month, 8).isoformat(),
                    "direction": "debit",
                    "category": "living_costs",
                    "amount": int(costs),
                }
            )
        else:
            future_months.append((salary, costs))

    positive_months = sum(salary - costs > 0 for salary, costs in future_months)
    future_is_positive = positive_months >= 5
    return past_transactions, [{"future_surplus_persisted": int(future_is_positive)}]


def create_training_data(n_samples: int = 2400, seed: int = 42) -> pd.DataFrame:
    """Create unique history/future profiles with forward-looking labels."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n_samples):
        transactions, future = _profile(rng)
        features = calculate_features(
            transactions,
            stated_goal="long_term_savings",
            opening_balance=rng.randint(100_000, 1_500_000),
        )
        row = {name: features[name] for name in FEATURE_COLUMNS}
        row.update(future[0])
        rows.append(row)
    return pd.DataFrame(rows)
