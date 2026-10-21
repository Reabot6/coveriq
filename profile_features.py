

from __future__ import annotations

from collections import defaultdict
from statistics import mean, pstdev
from typing import Any


def summarize_activity(
    transactions: list[dict[str, Any]],
    opening_balance: float = 0.0,
) -> dict[str, Any]:
    """Return monthly cash-flow signals without making a product decision."""
    if not transactions:
        raise ValueError("At least one transaction is required")

    monthly: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "credits": 0.0,
            "debits": 0.0,
            "credit_categories": defaultdict(float),
        }
    )
    balance = float(opening_balance)
    ending_balances: dict[str, float] = {}

    for transaction in sorted(transactions, key=lambda item: item["date"]):
        month = str(transaction["date"])[:7]
        amount = float(transaction["amount"])
        direction = transaction["direction"]

        if amount < 0:
            raise ValueError("Transaction amounts must be non-negative")
        if direction == "credit":
            category = str(transaction["category"])
            monthly[month]["credits"] += amount
            monthly[month]["credit_categories"][category] += amount
            balance += amount
        elif direction == "debit":
            monthly[month]["debits"] += amount
            balance -= amount
        else:
            raise ValueError("Transaction direction must be 'credit' or 'debit'")

        ending_balances[month] = balance

    months = sorted(monthly)
    credits = [monthly[month]["credits"] for month in months]
    debits = [monthly[month]["debits"] for month in months]
    net_flows = [credit - debit for credit, debit in zip(credits, debits)]
    average_credit = mean(credits)
    average_absolute_net = mean(abs(net) for net in net_flows)
    net_variation = pstdev(net_flows) / max(average_absolute_net, 1.0)
    balances = [ending_balances[month] for month in months]

    recent_months = min(6, len(months))
    recent_net = mean(net_flows[-recent_months:])
    recent_credits = mean(credits[-recent_months:])
    balance_change = None
    if len(balances) >= 12:
        earlier_balance = mean(balances[-12:-6])
        latest_balance = mean(balances[-6:])
        balance_change = (latest_balance - earlier_balance) / max(
            abs(earlier_balance), 100_000.0
        )

    credit_categories: dict[str, float] = defaultdict(float)
    for month in months:
        for category, amount in monthly[month]["credit_categories"].items():
            credit_categories[category] += amount

    salary_signals = _salary_signals(monthly, months)
    return {
        "history_months": len(months),
        "average_monthly_credits": round(average_credit, 2),
        "average_monthly_debits": round(mean(debits), 2),
        "average_monthly_surplus": round(mean(net_flows), 2),
        "recent_monthly_surplus": round(recent_net, 2),
        "recent_surplus_ratio": round(
            recent_net / recent_credits if recent_credits else 0.0,
            4,
        ),
        "positive_cashflow_month_share": round(
            sum(net > 0 for net in net_flows) / len(net_flows),
            4,
        ),
        "cashflow_stability": round(max(0.0, min(1.0, 1.0 - net_variation)), 4),
        "balance_change_6m_ratio": (
            round(balance_change, 4) if balance_change is not None else None
        ),
        "credit_category_shares": {
            category: round(amount / sum(credit_categories.values()), 4)
            for category, amount in credit_categories.items()
        },
        "salary": salary_signals,
    }


def _salary_signals(
    monthly: dict[str, dict[str, Any]],
    months: list[str],
) -> dict[str, float | int | None]:
    """Summarize salary history when the account has salary credits."""
    salaries = [monthly[month]["credit_categories"].get("salary", 0.0) for month in months]
    paid_months = [salary for salary in salaries if salary > 0]
    if len(paid_months) < 6:
        return {
            "months_observed": len(paid_months),
            "recent_change_pct": None,
            "months_at_higher_level": None,
            "recent_consistency": None,
        }

    baseline_period = salaries[:-12] if len(salaries) > 12 else salaries[:-3]
    baseline_salaries = [salary for salary in baseline_period if salary > 0]
    if not baseline_salaries:
        baseline_salaries = paid_months[:-3]

    baseline = mean(baseline_salaries)
    recent_salary = mean(salaries[-3:])
    recent_paid = [salary for salary in salaries[-3:] if salary > 0]
    salary_variation = pstdev(recent_paid) / mean(recent_paid) if len(recent_paid) > 1 else 0.0

    months_at_higher_level = 0
    for salary in reversed(salaries):
        if salary < baseline * 1.25:
            break
        months_at_higher_level += 1

    return {
        "months_observed": len(paid_months),
        "recent_change_pct": round((recent_salary - baseline) / baseline, 4),
        "months_at_higher_level": months_at_higher_level,
        "recent_consistency": round(max(0.0, min(1.0, 1.0 - salary_variation)), 4),
    }
