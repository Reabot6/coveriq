"""Premium inputs and repeatable cash-flow stress scenarios."""

from __future__ import annotations

from collections import defaultdict
from statistics import median
from typing import Any


FORECAST_MONTHS = 12
PAYMENT_FREQUENCIES = ("monthly", "annual")


def premium_schedule(
    amount: float,
    frequency: str,
    months: int = FORECAST_MONTHS,
) -> list[float]:
    """Return the premium charged in each forecast month."""
    if amount <= 0:
        raise ValueError("Premium amount must be greater than zero")
    if months < 1:
        raise ValueError("Forecast must include at least one month")
    if frequency == "monthly":
        return [float(amount)] * months
    if frequency == "annual":
        return [float(amount)] + [0.0] * (months - 1)
    raise ValueError("Payment frequency must be 'monthly' or 'annual'")


def add_premium_features(
    financial_features: dict[str, float],
    amount: float,
    frequency: str,
) -> dict[str, float]:
    """Add premium burden and payment schedule to model inputs."""
    schedule = premium_schedule(amount, frequency)
    monthly_income = max(financial_features["typical_monthly_income"], 1.0)
    current_balance = max(financial_features["current_balance"], 0.0)
    balance_floor = max(current_balance, monthly_income * 0.1, 1.0)
    equivalent_monthly_amount = (
        amount if frequency == "monthly" else amount / 12.0
    )
    return {
        **financial_features,
        "premium_to_income_ratio": equivalent_monthly_amount / monthly_income,
        "premium_payment_to_balance_ratio": min(max(schedule) / balance_floor, 20.0),
        "annual_payment": float(frequency == "annual"),
    }


def stress_test(
    transactions: list[dict[str, Any]],
    opening_balance: float,
    premium_amount: float,
    payment_frequency: str,
) -> list[dict[str, float | str]]:
    """Repeat recent monthly flows for a baseline and a downside scenario."""
    premium_by_month = premium_schedule(premium_amount, payment_frequency)
    monthly: dict[str, dict[str, float]] = defaultdict(
        lambda: {"credits": 0.0, "debits": 0.0, "recurring": 0.0}
    )
    recurring_categories = {
        "housing", "utilities", "loan_payment", "insurance", "subscription", "education"
    }
    for transaction in transactions:
        month = str(transaction["date"])[:7]
        category = str(transaction.get("category", "other")).lower()
        amount = float(transaction["amount"])
        if transaction["direction"] == "credit":
            monthly[month]["credits"] += amount
        else:
            monthly[month]["debits"] += amount
            if category in recurring_categories:
                monthly[month]["recurring"] += amount

    months = sorted(monthly)
    recent_months = months[-12:]
    if not recent_months:
        raise ValueError("Transactions are required for a stress test")

    recent_recurring = [monthly[month]["recurring"] for month in recent_months]
    reserve_floor = median(recent_recurring)
    current_balance = opening_balance + sum(
        monthly[month]["credits"] - monthly[month]["debits"] for month in months
    )
    scenarios = (
        ("Recent pattern", 1.0, 1.0),
        ("Income down 20%, spending up 15%", 0.8, 1.15),
    )
    results: list[dict[str, float | str]] = []

    for name, income_factor, debit_factor in scenarios:
        balance = current_balance
        lowest_balance = balance
        months_below_reserve = 0
        for forecast_month in range(FORECAST_MONTHS):
            source_month = recent_months[forecast_month % len(recent_months)]
            flow = monthly[source_month]
            balance += (
                flow["credits"] * income_factor
                - flow["debits"] * debit_factor
                - premium_by_month[forecast_month]
            )
            lowest_balance = min(lowest_balance, balance)
            if balance < reserve_floor:
                months_below_reserve += 1
        results.append(
            {
                "scenario": name,
                "lowest_balance": round(lowest_balance),
                "months_below_reserve": float(months_below_reserve),
                "reserve_floor": round(reserve_floor),
            }
        )
    return results
