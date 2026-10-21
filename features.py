"""Summarize transaction history as a small set of customer signals."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, pstdev
from typing import Any


GOALS = ("long_term_savings", "education", "family_protection")


def calculate_features(
    transactions: list[dict[str, Any]],
    stated_goal: str,
    opening_balance: float = 0.0,
) -> dict[str, float]:
    """Aggregate categorized transactions; the goal is customer-provided."""
    if stated_goal not in GOALS:
        raise ValueError(f"stated_goal must be one of {GOALS}")
    if not transactions:
        raise ValueError("At least one transaction is required")

    monthly: dict[str, dict[str, float]] = defaultdict(
        lambda: {"salary": 0.0, "credits": 0.0, "debits": 0.0}
    )
    balance = float(opening_balance)
    month_end_balances: dict[str, float] = {}

    for transaction in sorted(transactions, key=lambda item: item["date"]):
        month = transaction["date"][:7]
        amount = float(transaction["amount"])
        direction = transaction["direction"]

        if amount < 0:
            raise ValueError("Transaction amounts must be non-negative")
        if direction == "credit":
            monthly[month]["credits"] += amount
            balance += amount
            if transaction["category"] == "salary":
                monthly[month]["salary"] += amount
        elif direction == "debit":
            monthly[month]["debits"] += amount
            balance -= amount
        else:
            raise ValueError("Transaction direction must be 'credit' or 'debit'")

        month_end_balances[month] = balance

    months = sorted(monthly)
    salaries = [monthly[month]["salary"] for month in months]
    if len(salaries) < 6 or any(salary <= 0 for salary in salaries):
        raise ValueError("Need at least 6 months with a salary credit")

    average_salary = mean(salaries)
    salary_variation = pstdev(salaries) / average_salary if average_salary else 1.0
    income_stability = max(0.0, min(1.0, 1.0 - salary_variation))

    earlier_salary = mean(salaries[:-3]) if len(salaries) > 3 else average_salary
    recent_salary = mean(salaries[-3:])
    income_change = (
        max(0.0, (recent_salary - earlier_salary) / earlier_salary)
        if earlier_salary
        else 0.0
    )

    changed_level = earlier_salary * 1.25
    months_since_change = 0
    for salary in reversed(salaries):
        if salary < changed_level:
            break
        months_since_change += 1

    monthly_surplus = [
        monthly[month]["credits"] - monthly[month]["debits"] for month in months
    ]
    recent_surplus = mean(monthly_surplus[-3:])
    recent_credits = mean([monthly[month]["credits"] for month in months[-3:]])
    surplus_ratio = recent_surplus / recent_credits if recent_credits else 0.0
    surplus_ratio = max(-1.0, min(1.0, surplus_ratio))

    balances = [month_end_balances[month] for month in months]
    previous_balance = mean(balances[-6:-3]) if len(balances) >= 6 else balances[0]
    recent_balance = mean(balances[-3:])
    balance_growth = (recent_balance - previous_balance) / max(
        abs(previous_balance), 100_000.0
    )
    balance_growth = max(-2.0, min(2.0, balance_growth))

    features = {
        "income_stability": income_stability,
        "recent_income_change_pct": income_change,
        "months_since_income_change": float(months_since_change),
        "average_monthly_surplus": recent_surplus,
        "surplus_ratio": surplus_ratio,
        "balance_growth_ratio": balance_growth,
        "history_months": float(len(months)),
    }
    features.update({f"goal_{goal}": float(stated_goal == goal) for goal in GOALS})
    return {name: round(value, 4) for name, value in features.items()}


def feature_rows(
    transactions: list[dict[str, Any]], opening_balance: float = 0.0
) -> list[dict[str, float | str]]:
    """Build the month-by-month rows shown in the demo."""
    monthly: dict[str, dict[str, float]] = defaultdict(
        lambda: {"salary": 0.0, "credits": 0.0, "debits": 0.0}
    )
    for transaction in sorted(transactions, key=lambda item: item["date"]):
        month = transaction["date"][:7]
        amount = float(transaction["amount"])
        if transaction["direction"] == "credit":
            monthly[month]["credits"] += amount
            if transaction["category"] == "salary":
                monthly[month]["salary"] += amount
        else:
            monthly[month]["debits"] += amount

    balance = float(opening_balance)
    rows: list[dict[str, float | str]] = []
    for month in sorted(monthly):
        totals = monthly[month]
        salary = totals["salary"]
        balance += totals["credits"] - totals["debits"]
        rows.append(
            {
                "Month": month,
                "Salary credit": round(salary),
                "Other credits": round(totals["credits"] - salary),
                "Debits": round(totals["debits"]),
                "End balance": round(balance),
            }
        )
    return rows
