"""Generate fictional account activity for each batch-demo customer."""

from __future__ import annotations

from calendar import monthrange
from datetime import date

from chinedu_data import HISTORY_MONTHS, build_transactions
from demo_batch import CUSTOMERS


def _month_start(month_offset: int) -> date:
    """Return the first day of a month relative to the current month."""
    today = date.today()
    current_month = today.year * 12 + today.month - 1
    year, month_index = divmod(current_month + month_offset, 12)
    return date(year, month_index + 1, 1)


def _build_marine_business_transactions() -> list[dict[str, object]]:
    """Create three years of fictional freight income and operating costs."""
    transactions: list[dict[str, object]] = []
    monthly_revenue_changes = [0, 250_000, -180_000, 420_000, -90_000, 160_000]

    for month_index in range(HISTORY_MONTHS):
        month_date = _month_start(month_index - HISTORY_MONTHS)
        days_in_month = monthrange(month_date.year, month_date.month)[1]
        revenue = 4_200_000 + monthly_revenue_changes[month_index % 6]

        transactions.extend(
            [
                {
                    "date": month_date.replace(day=10).isoformat(),
                    "direction": "credit",
                    "category": "freight_income",
                    "amount": int(revenue * 0.65),
                },
                {
                    "date": month_date.replace(day=24).isoformat(),
                    "direction": "credit",
                    "category": "freight_income",
                    "amount": revenue - int(revenue * 0.65),
                },
            ]
        )

        operating_costs = [
            ("fuel", 1_150_000),
            ("crew", 780_000),
            ("port_fees", 420_000),
            ("repairs_and_supplies", 360_000),
        ]
        for day, (category, amount) in zip((5, 12, 19, 27), operating_costs):
            transactions.append(
                {
                    "date": month_date.replace(day=min(day, days_in_month)).isoformat(),
                    "direction": "debit",
                    "category": category,
                    "amount": amount,
                }
            )

    return sorted(transactions, key=lambda transaction: transaction["date"])


def build_demo_activity(
    customer_ids: set[str] | None = None,
) -> dict[str, dict[str, object]]:
    """Build fictional histories for the requested demo customer IDs."""
    requested_ids = customer_ids
    if requested_ids is None:
        requested_ids = {customer["customer_id"] for customer in CUSTOMERS}

    activity = {}
    if "CUST-001" in requested_ids:
        activity["CUST-001"] = {
            "opening_balance": 450_000,
            "transactions": build_transactions(stable_months_after_raise=9),
        }
    if "CUST-002" in requested_ids:
        activity["CUST-002"] = {
            "opening_balance": 8_500_000,
            "transactions": _build_marine_business_transactions(),
        }

    return activity
