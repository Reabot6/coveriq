"""Fictional salary history used in the CoverIQ demo."""

from __future__ import annotations

from calendar import monthrange
from datetime import date


CUSTOMER = {
    "name": "Chinedu",
    "stated_goal": "long_term_savings",
    "opening_balance": 450_000,
}

HISTORY_MONTHS = 36


def _month_start(month_offset: int) -> date:
    """Return a date relative to the current month."""
    today = date.today()
    absolute_month = today.year * 12 + today.month - 1 + month_offset
    year, zero_based_month = divmod(absolute_month, 12)
    return date(year, zero_based_month + 1, 1)


def build_transactions(stable_months_after_raise: int = 3) -> list[dict[str, object]]:
    """Create three years of fictional transactions through the last full month."""
    if stable_months_after_raise not in (3, 6, 9):
        raise ValueError("stable_months_after_raise must be 3, 6, or 9")

    transactions: list[dict[str, object]] = []
    for month_index in range(HISTORY_MONTHS):
        month_date = _month_start(month_index - HISTORY_MONTHS)
        days_in_month = monthrange(month_date.year, month_date.month)[1]
        month_end = month_date.replace(day=days_in_month)
        months_after_raise = month_index - (HISTORY_MONTHS - stable_months_after_raise)
        salary = 500_000 if months_after_raise < 0 else 1_000_000
        salary_variation = [0, 12_000, -8_000, 5_000, -10_000, 7_000][month_index % 6]
        salary += salary_variation

        expenses = [
            ("housing", 150_000),
            ("food", 125_000 + (month_index % 3) * 10_000),
            ("transport", 55_000 + (month_index % 2) * 8_000),
            ("utilities", 35_000),
            ("other", 65_000 + (month_index % 4) * 5_000),
        ]
        transactions.append(
            {
                "date": month_date.replace(day=25).isoformat(),
                "direction": "credit",
                "category": "salary",
                "amount": salary,
            }
        )

        for expense_index, (category, amount) in enumerate(expenses, start=2):
            day = min(expense_index * 4, days_in_month)
            transactions.append(
                {
                    "date": month_date.replace(day=day).isoformat(),
                    "direction": "debit",
                    "category": category,
                    "amount": amount,
                }
            )

        savings_transfer = 20_000 if month_index % 3 == 0 else 0
        if savings_transfer:
            transactions.append(
                {
                    "date": month_end.isoformat(),
                    "direction": "debit",
                    "category": "savings_transfer",
                    "amount": savings_transfer,
                }
            )

    return sorted(transactions, key=lambda transaction: transaction["date"])


