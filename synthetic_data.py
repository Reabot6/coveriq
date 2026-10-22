"""Create fictional bank histories for the CoverIQ insurance-routing demo.

The generator creates 36 months of account history followed by three future
months. The future months are used only to create training outcomes; model
features are built from the 36 earlier months.
"""

from __future__ import annotations

from calendar import monthrange

import numpy as np
import pandas as pd


HISTORY_MONTHS = 36
FUTURE_MONTHS = 3
MONTHS_PER_PROFILE = HISTORY_MONTHS + FUTURE_MONTHS
CONTRIBUTION_SCENARIOS = (5_000, 15_000, 30_000, 60_000, 120_000)


def _profile_settings(rng: np.random.Generator, persona: str) -> dict:
    """Draw a customer's underlying income and spending pattern."""
    income = float(rng.uniform(220_000, 1_500_000))
    settings = {
        "persona": persona,
        "base_income": income,
        "income_noise": 0.035,
        "income_reliability": 0.98,
        "core_ratio": float(rng.uniform(0.48, 0.70)),
        "debt_ratio": float(rng.uniform(0.02, 0.10)),
        "discretionary_ratio": float(rng.uniform(0.08, 0.18)),
        "saving_ratio": float(rng.uniform(0.00, 0.08)),
        "annual_growth": float(rng.uniform(0.00, 0.08)),
        "jump_persists": bool(rng.integers(0, 2)),
    }

    if persona == "variable_income":
        settings["income_noise"] = 0.28
        settings["income_reliability"] = 0.88
        settings["core_ratio"] = float(rng.uniform(0.50, 0.76))
    elif persona == "high_commitments":
        settings["core_ratio"] = float(rng.uniform(0.68, 0.86))
        settings["debt_ratio"] = float(rng.uniform(0.10, 0.24))
        settings["discretionary_ratio"] = float(rng.uniform(0.10, 0.20))
    elif persona == "recent_income_change":
        settings["core_ratio"] = float(rng.uniform(0.50, 0.72))
        settings["income_noise"] = 0.045
    elif persona == "income_disruption":
        settings["income_noise"] = 0.12
        settings["income_reliability"] = 0.78
        settings["core_ratio"] = float(rng.uniform(0.58, 0.78))
    elif persona == "steady_saver":
        settings["core_ratio"] = float(rng.uniform(0.38, 0.57))
        settings["saving_ratio"] = float(rng.uniform(0.05, 0.14))
        settings["discretionary_ratio"] = float(rng.uniform(0.06, 0.14))

    return settings


def _income_for_month(
    settings: dict,
    month_index: int,
    rng: np.random.Generator,
    *,
    chinedu: bool = False,
) -> tuple[float, str]:
    """Return the main income credit and its transaction category."""
    if chinedu:
        level = 1_000_000 if month_index >= 33 else 500_000
        return max(0.0, level * rng.normal(1.0, 0.012)), "salary"

    persona = settings["persona"]
    base = settings["base_income"]
    reliability = settings["income_reliability"]

    if persona == "recent_income_change":
        level = base * (0.72 if month_index < 33 else 1.38)
        if month_index >= HISTORY_MONTHS and not settings["jump_persists"]:
            level = base * 0.78
    elif persona == "income_disruption" and month_index >= 27:
        level = base * (0.48 if month_index >= HISTORY_MONTHS else 0.76)
    elif persona == "variable_income":
        level = base * rng.lognormal(mean=-0.5 * settings["income_noise"] ** 2,
                                     sigma=settings["income_noise"])
    else:
        level = base * (1 + settings["annual_growth"]) ** (month_index / 12)

    if rng.random() > reliability:
        level = 0.0
    elif persona != "variable_income":
        level *= max(0.0, rng.normal(1.0, settings["income_noise"]))

    category = "business_income" if persona == "variable_income" else "salary"
    return float(level), category


def _date_for_month(month: pd.Period, day: int) -> str:
    last_day = monthrange(month.year, month.month)[1]
    return str(month.start_time.date() + pd.Timedelta(days=min(day, last_day) - 1))


def _make_profile(
    profile_id: str,
    display_name: str,
    persona: str,
    rng: np.random.Generator,
    *,
    chinedu: bool = False,
) -> tuple[list[dict], list[dict], dict]:
    settings = _profile_settings(rng, persona)
    base = 500_000.0 if chinedu else settings["base_income"]
    if chinedu:
        settings.update(
            {
                "base_income": base,
                "core_ratio": 0.72,
                "debt_ratio": 0.16,
                "discretionary_ratio": 0.18,
                "saving_ratio": 0.06,
            }
        )

    start_month = pd.Period("2023-01", freq="M")
    opening_balance = base * (3.2 if chinedu else rng.uniform(0.25, 1.25))
    transactions: list[dict] = []
    months: list[dict] = []

    for month_index in range(MONTHS_PER_PROFILE):
        month = start_month + month_index
        balance = opening_balance
        income, income_category = _income_for_month(
            settings, month_index, rng, chinedu=chinedu
        )
        credits: list[tuple[str, float]] = []
        if income > 0:
            credits.append((income_category, income))
        if not chinedu and rng.random() < 0.24:
            extra = settings["base_income"] * rng.uniform(0.025, 0.16)
            credits.append(("other_income", extra))

        spend_scale = rng.lognormal(mean=-0.5 * 0.06**2, sigma=0.06)
        core_total = settings["base_income"] * settings["core_ratio"] * spend_scale
        housing = core_total * 0.44
        food = core_total * 0.28 * rng.uniform(0.88, 1.12)
        utilities = core_total * 0.10 * rng.uniform(0.85, 1.18)
        transport = core_total * 0.12 * rng.uniform(0.82, 1.20)
        debt = settings["base_income"] * settings["debt_ratio"] * rng.uniform(0.92, 1.08)
        other_spend = (
            settings["base_income"]
            * settings["discretionary_ratio"]
            * rng.uniform(0.62, 1.38)
        )
        savings_transfer = (
            max(0.0, income) * settings["saving_ratio"] * rng.uniform(0.70, 1.25)
        )
        debits = [
            ("housing", housing),
            ("food", food),
            ("utilities", utilities),
            ("transport", transport),
            ("debt_payment", debt),
            ("other_spend", other_spend),
            ("savings_transfer", savings_transfer),
        ]

        month_income = sum(amount for _, amount in credits)
        month_outflow = sum(amount for _, amount in debits)
        events = []
        for direction, entries in (("in", credits), ("out", debits)):
            for category, amount in entries:
                if amount <= 1:
                    continue
                day = int(rng.integers(1, 26))
                events.append(
                    (_date_for_month(month, day), category, direction, round(float(amount), 2))
                )
        events.sort(key=lambda event: event[0])

        for event_date, category, direction, amount in events:
            balance += amount if direction == "in" else -amount
            transactions.append(
                {
                    "profile_id": profile_id,
                    "date": event_date,
                    "month": str(month),
                    "month_index": month_index,
                    "category": category,
                    "direction": direction,
                    "amount": amount,
                    "balance_after": round(balance, 2),
                }
            )

        essential_outflow = housing + food + utilities + transport + debt
        months.append(
            {
                "profile_id": profile_id,
                "month": str(month),
                "month_index": month_index,
                "income": round(month_income, 2),
                "total_outflow": round(month_outflow, 2),
                "essential_outflow": round(essential_outflow, 2),
                "net_flow": round(month_income - month_outflow, 2),
                "closing_balance": round(balance, 2),
            }
        )
        opening_balance = balance

    profile = {
        "profile_id": profile_id,
        "display_name": display_name,
        "persona": persona,
        "is_demo": profile_id.startswith("demo_"),
    }
    return transactions, months, profile


def generate_population(
    training_profiles: int = 1_000,
    demo_accounts: int = 11,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Build the repeatable model population and the visible demo accounts."""
    rng = np.random.default_rng(seed)
    personas = (
        "steady_salary",
        "variable_income",
        "high_commitments",
        "recent_income_change",
        "income_disruption",
    )
    all_transactions: list[dict] = []
    all_months: list[dict] = []
    profiles: list[dict] = []

    for index in range(training_profiles):
        persona = str(rng.choice(personas))
        rows, month_rows, profile = _make_profile(
            f"train_{index + 1:04d}", "", persona, rng
        )
        all_transactions.extend(rows)
        all_months.extend(month_rows)
        profiles.append(profile)

    rows, month_rows, profile = _make_profile(
        "demo_chinedu",
        "Chinedu",
        "recent_income_change",
        rng,
        chinedu=True,
    )
    all_transactions.extend(rows)
    all_months.extend(month_rows)
    profiles.append(profile)

    demo_personas = [
        "steady_salary",
        "variable_income",
        "high_commitments",
        "recent_income_change",
        "income_disruption",
    ]
    for index in range(demo_accounts):
        persona = demo_personas[index % len(demo_personas)]
        rows, month_rows, profile = _make_profile(
            f"demo_{index + 1:02d}",
            f"Account {index + 1:02d}",
            persona,
            rng,
        )
        all_transactions.extend(rows)
        all_months.extend(month_rows)
        profiles.append(profile)

    transactions = pd.DataFrame(all_transactions)
    monthly = pd.DataFrame(all_months)
    profile_frame = pd.DataFrame(profiles)
    transactions["date"] = pd.to_datetime(transactions["date"])
    return {
        "transactions": transactions,
        "monthly": monthly,
        "profiles": profile_frame,
    }
