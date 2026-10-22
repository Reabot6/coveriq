"""Turn permitted transaction history into cash-flow model features."""

from __future__ import annotations

from collections import defaultdict
from statistics import median, pstdev
from typing import Any

import pandas as pd


RECURRING_OUTFLOW_CATEGORIES = {
    "housing",
    "utilities",
    "loan_payment",
    "insurance",
    "subscription",
    "education",
}


def calculate_features(
    transactions: list[dict[str, Any]],
    stated_goal: str | None = None,
    opening_balance: float = 0.0,
) -> dict[str, float]:
    """Summarize transaction patterns; product interest stays outside the model."""
    del stated_goal  # A declared goal is a matching input, not a cash-flow feature.
    if not transactions:
        raise ValueError("At least one transaction is required")

    monthly: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "income": 0.0,
            "credits": 0.0,
            "debits": 0.0,
            "recurring_outflows": 0.0,
            "credit_count": 0.0,
            "debit_count": 0.0,
            "early_month_debits": 0.0,
        }
    )
    net_flow_by_month: dict[str, float] = defaultdict(float)
    for transaction in sorted(transactions, key=lambda item: item["date"]):
        month = str(transaction["date"])[:7]
        amount = float(transaction["amount"])
        direction = transaction["direction"]
        category = str(transaction.get("category", "other")).lower()
        if amount < 0:
            raise ValueError("Transaction amounts must be non-negative")

        if direction == "credit":
            monthly[month]["credits"] += amount
            monthly[month]["credit_count"] += 1
            net_flow_by_month[month] += amount
            if category in {"salary", "wages", "employment_income"}:
                monthly[month]["income"] += amount
        elif direction == "debit":
            monthly[month]["debits"] += amount
            monthly[month]["debit_count"] += 1
            if int(str(transaction["date"])[8:10]) <= 10:
                monthly[month]["early_month_debits"] += amount
            net_flow_by_month[month] -= amount
            if category in RECURRING_OUTFLOW_CATEGORIES:
                monthly[month]["recurring_outflows"] += amount
        else:
            raise ValueError("Transaction direction must be 'credit' or 'debit'")

    first_month, last_month = min(monthly), max(monthly)
    months = [str(month) for month in pd.period_range(first_month, last_month, freq="M")]
    for month in months:
        monthly[month]
        net_flow_by_month[month]

    if len(months) < 6:
        raise ValueError("At least six months of transaction history are required")

    # Carry a balance through months with no transactions.
    balance = float(opening_balance)
    ending_balances = []
    for month in months:
        balance += net_flow_by_month[month]
        ending_balances.append(balance)

    incomes = [monthly[month]["income"] for month in months]
    window_size = min(12, len(months))
    recent_incomes = incomes[-window_size:]
    positive_incomes = [income for income in recent_incomes if income > 0]
    typical_income = median(positive_incomes) if positive_incomes else 0.0
    income_mad = (
        median(abs(income - typical_income) for income in positive_incomes)
        if positive_incomes
        else 0.0
    )

    baseline_incomes = incomes[:-3]
    baseline_income_values = [income for income in baseline_incomes if income > 0]
    baseline_income = median(baseline_income_values) if baseline_income_values else 0.0
    recent_income = median(incomes[-3:])
    income_change = (
        recent_income / baseline_income - 1.0 if baseline_income > 0 else 0.0
    )

    months_at_higher_level = 0
    if baseline_income > 0:
        for income in reversed(incomes):
            if income < baseline_income * 1.25:
                break
            months_at_higher_level += 1

    surplus = [net_flow_by_month[month] for month in months]
    recent_surplus = surplus[-window_size:]
    typical_outflow = median(
        [monthly[month]["debits"] for month in months[-6:]]
    )
    typical_balance = median(ending_balances[-6:])
    earlier_balance = median(ending_balances[-12:-6]) if len(months) >= 12 else 0.0
    later_balance = median(ending_balances[-6:])

    features = {
        "income_presence_rate": sum(income > 0 for income in recent_incomes) / window_size,
        "income_variability": income_mad / typical_income if typical_income else 1.0,
        "recent_income_change_pct": income_change,
        "months_since_income_change": float(months_at_higher_level),
        "mean_monthly_surplus": sum(recent_surplus) / len(recent_surplus),
        "lower_quartile_surplus": float(pd.Series(recent_surplus).quantile(0.25)),
        "positive_surplus_rate": sum(value > 0 for value in recent_surplus) / len(recent_surplus),
        "surplus_variability": pstdev(recent_surplus) / max(typical_income, 1.0),
        "obligation_to_income_ratio": (
            sum(monthly[month]["recurring_outflows"] for month in months[-window_size:])
            / max(sum(recent_incomes), 1.0)
        ),
        "liquidity_buffer_months": max(0.0, typical_balance) / max(typical_outflow, 1.0),
        "balance_growth_months_of_income": (
            (later_balance - earlier_balance) / max(typical_income, 1.0)
            if len(months) >= 12
            else 0.0
        ),
        "debit_count_per_month": (
            sum(monthly[month]["debit_count"] for month in months[-window_size:])
            / window_size
        ),
        "early_month_outflow_share": (
            sum(monthly[month]["early_month_debits"] for month in months[-window_size:])
            / max(sum(monthly[month]["debits"] for month in months[-window_size:]), 1.0)
        ),
        "history_months": float(len(months)),
    }
    return {name: round(float(value), 4) for name, value in features.items()}


def feature_rows(
    transactions: list[dict[str, Any]], opening_balance: float = 0.0
) -> list[dict[str, float | str]]:
    """Build the monthly account timeline shown in the demo."""
    monthly: dict[str, dict[str, float]] = defaultdict(
        lambda: {"salary": 0.0, "credits": 0.0, "debits": 0.0}
    )
    for transaction in sorted(transactions, key=lambda item: item["date"]):
        month = str(transaction["date"])[:7]
        amount = float(transaction["amount"])
        if transaction["direction"] == "credit":
            monthly[month]["credits"] += amount
            if str(transaction.get("category", "")).lower() in {
                "salary", "wages", "employment_income"
            }:
                monthly[month]["salary"] += amount
        else:
            monthly[month]["debits"] += amount

    balance = float(opening_balance)
    rows: list[dict[str, float | str]] = []
    for month in sorted(monthly):
        totals = monthly[month]
        balance += totals["credits"] - totals["debits"]
        rows.append(
            {
                "Month": month,
                "Salary credit": round(totals["salary"]),
                "Other credits": round(totals["credits"] - totals["salary"]),
                "Debits": round(totals["debits"]),
                "End balance": round(balance),
            }
        )
    return rows
