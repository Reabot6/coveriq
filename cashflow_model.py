"""Train and run CoverIQ's simulated cash-flow sustainability model."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from synthetic_data import (
    CONTRIBUTION_SCENARIOS,
    FUTURE_MONTHS,
    HISTORY_MONTHS,
    generate_population,
)


ROUTING_THRESHOLD = 0.65
FEATURE_COLUMNS = (
    "average_monthly_income",
    "income_variation",
    "income_month_share",
    "recent_income_level_ratio",
    "recent_higher_income_months",
    "average_essential_outflow",
    "essential_outflow_ratio",
    "average_monthly_surplus",
    "positive_surplus_month_share",
    "surplus_variation",
    "balance_trend_ratio",
    "current_balance",
    "cash_buffer_months",
    "proposed_contribution",
    "contribution_to_income",
    "contribution_to_surplus",
)

FEATURE_LABELS = {
    "average_monthly_income": "Average monthly income",
    "income_variation": "Income variation",
    "income_month_share": "Months with income",
    "recent_income_level_ratio": "Recent income vs earlier level",
    "recent_higher_income_months": "Months at higher income level",
    "average_essential_outflow": "Typical essential outflow",
    "essential_outflow_ratio": "Essential outflows vs income",
    "average_monthly_surplus": "Average monthly remainder",
    "positive_surplus_month_share": "Months with positive remainder",
    "surplus_variation": "Remainder variation",
    "balance_trend_ratio": "Balance trend vs income",
    "current_balance": "Latest balance",
    "cash_buffer_months": "Cash buffer in months",
    "proposed_contribution": "Illustrative monthly contribution",
    "contribution_to_income": "Contribution vs income",
    "contribution_to_surplus": "Contribution vs remainder",
}


def _history_for(monthly: pd.DataFrame, profile_id: str) -> pd.DataFrame:
    return (
        monthly.loc[
            (monthly["profile_id"] == profile_id)
            & (monthly["month_index"] < HISTORY_MONTHS)
        ]
        .sort_values("month_index")
        .reset_index(drop=True)
    )


def customer_features(history: pd.DataFrame, contribution: float) -> dict[str, float]:
    """Summarize the 36-month history and a candidate monthly contribution."""
    history = history.sort_values("month_index")
    income = history["income"].astype(float)
    outflow = history["total_outflow"].astype(float)
    surplus = income - outflow
    essential = history["essential_outflow"].astype(float)
    balance = history["closing_balance"].astype(float)

    average_income = max(float(income.mean()), 1.0)
    average_essential = max(float(essential.mean()), 1.0)
    average_surplus = float(surplus.mean())
    earlier_income = float(income.iloc[-12:-3].mean())
    recent_income = float(income.iloc[-3:].mean())
    recent_ratio = recent_income / max(earlier_income, 1.0)
    higher_months = int((income.iloc[-6:] > max(earlier_income * 1.15, 1.0)).sum())
    balance_slope = float(np.polyfit(np.arange(len(balance)), balance, 1)[0])

    return {
        "average_monthly_income": average_income,
        "income_variation": float(income.std(ddof=0) / average_income),
        "income_month_share": float((income > 0).mean()),
        "recent_income_level_ratio": recent_ratio,
        "recent_higher_income_months": float(higher_months),
        "average_essential_outflow": average_essential,
        "essential_outflow_ratio": average_essential / average_income,
        "average_monthly_surplus": average_surplus,
        "positive_surplus_month_share": float((surplus > 0).mean()),
        "surplus_variation": float(surplus.std(ddof=0) / max(abs(average_surplus), 1.0)),
        "balance_trend_ratio": balance_slope / average_income,
        "current_balance": float(balance.iloc[-1]),
        "cash_buffer_months": float(balance.iloc[-1] / average_essential),
        "proposed_contribution": float(contribution),
        "contribution_to_income": float(contribution / average_income),
        "contribution_to_surplus": float(contribution / max(average_surplus, 1.0)),
    }


def _future_sustainability_label(
    history: pd.DataFrame,
    future: pd.DataFrame,
    contribution: float,
) -> int:
    """Label whether a simulated contribution preserves a basic cash buffer."""
    if len(future) < FUTURE_MONTHS:
        raise ValueError("Each simulated history needs three future outcome months.")

    balance = float(history.iloc[-1]["closing_balance"])
    buffer_floor = max(float(history["essential_outflow"].mean()) * 0.25, 0.0)
    for row in future.head(FUTURE_MONTHS).itertuples(index=False):
        balance += float(row.net_flow) - contribution
        if balance < buffer_floor:
            return 0
    return 1


def _training_examples(monthly: pd.DataFrame, profile_ids: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[dict[str, float]] = []
    labels: list[int] = []
    by_profile = {key: frame for key, frame in monthly.groupby("profile_id", sort=False)}

    for profile_id in profile_ids:
        account = by_profile[profile_id].sort_values("month_index")
        history = account[account["month_index"] < HISTORY_MONTHS]
        future = account[account["month_index"] >= HISTORY_MONTHS]
        for contribution in CONTRIBUTION_SCENARIOS:
            rows.append(customer_features(history, contribution))
            labels.append(_future_sustainability_label(history, future, contribution))

    return pd.DataFrame(rows, columns=FEATURE_COLUMNS), np.asarray(labels, dtype=int)


def fit_model(population: dict[str, pd.DataFrame]) -> dict:
    """Train on generated profiles and evaluate on separate generated accounts."""
    profiles = population["profiles"]
    monthly = population["monthly"]
    training_ids = profiles.loc[profiles["profile_id"].str.startswith("train_"), "profile_id"]
    train_ids, test_ids = train_test_split(
        training_ids.to_list(), test_size=0.20, random_state=42
    )
    train_x, train_y = _training_examples(monthly, train_ids)
    test_x, test_y = _training_examples(monthly, test_ids)

    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(max_iter=1_500, random_state=42),
    )
    model.fit(train_x, train_y)
    test_scores = model.predict_proba(test_x)[:, 1]
    metrics = {
        "training_customers": len(train_ids),
        "held_out_customers": len(test_ids),
        "training_examples": len(train_y),
        "held_out_examples": len(test_y),
        "held_out_positive_rate": float(test_y.mean()),
        "roc_auc": float(roc_auc_score(test_y, test_scores))
        if len(np.unique(test_y)) > 1
        else None,
        "average_precision": float(average_precision_score(test_y, test_scores)),
        "brier_score": float(brier_score_loss(test_y, test_scores)),
        "label": (
            "On generated customer histories: did the closing balance stay above "
            "25% of typical essential outflows for each of the next three months?"
        ),
    }
    return {
        "model": model,
        "metrics": metrics,
        "feature_columns": list(FEATURE_COLUMNS),
        "training_profile_ids": set(train_ids),
        "held_out_profile_ids": set(test_ids),
    }


def score_contribution(
    bundle: dict,
    history: pd.DataFrame,
    contribution: float,
) -> float:
    row = pd.DataFrame([customer_features(history, contribution)], columns=FEATURE_COLUMNS)
    return float(bundle["model"].predict_proba(row)[0, 1])


def route_profile(bundle: dict, history: pd.DataFrame) -> dict:
    """Choose the highest illustrative amount clearing the review threshold."""
    scenarios = [
        {
            "contribution": amount,
            "score": score_contribution(bundle, history, amount),
        }
        for amount in CONTRIBUTION_SCENARIOS
    ]
    routed = [item for item in scenarios if item["score"] >= ROUTING_THRESHOLD]
    selected = max(routed, key=lambda item: item["contribution"]) if routed else None
    top_scenario = max(scenarios, key=lambda item: item["score"])
    return {
        "scenarios": scenarios,
        "selected": selected,
        "top_scenario": top_scenario,
        "action": "surface_for_review" if selected else "hold_back",
    }


def explain_score(bundle: dict, history: pd.DataFrame, contribution: float) -> list[dict]:
    """Show local positive and negative linear-model signals for one scenario."""
    raw = pd.DataFrame(
        [customer_features(history, contribution)], columns=FEATURE_COLUMNS
    )
    imputer = bundle["model"].named_steps["simpleimputer"]
    scaler = bundle["model"].named_steps["standardscaler"]
    classifier = bundle["model"].named_steps["logisticregression"]
    transformed = scaler.transform(imputer.transform(raw))[0]
    names = imputer.get_feature_names_out(FEATURE_COLUMNS)
    contributions = transformed * classifier.coef_[0]
    ranked = np.argsort(np.abs(contributions))[::-1]

    output = []
    for index in ranked:
        name = str(names[index])
        if name not in FEATURE_LABELS or len(output) >= 5:
            continue
        effect = float(contributions[index])
        output.append(
            {
                "signal": FEATURE_LABELS[name],
                "direction": "supports" if effect > 0 else "reduces",
                "relative_effect": effect,
            }
        )
    return output


def build_demo(progress: Callable[[str], None] | None = None) -> dict:
    """Prepare the population, train once, and score the visible demo batch."""
    notify = progress or (lambda _message: None)
    notify("Generating 36-month account histories and three future outcome months")
    population = generate_population()

    notify("Building cash-flow features and training the model on generated accounts")
    bundle = fit_model(population)

    notify("Scoring each demo account against the contribution scenarios")
    monthly = population["monthly"]
    demo_profiles = population["profiles"].loc[
        population["profiles"]["is_demo"]
    ].copy()
    routes = []
    for profile in demo_profiles.itertuples(index=False):
        history = _history_for(monthly, profile.profile_id)
        result = route_profile(bundle, history)
        selected = result["selected"]
        top_scenario = selected or result["top_scenario"]
        routes.append(
            {
                "profile_id": profile.profile_id,
                "customer": profile.display_name,
                "action": result["action"],
                "selected_contribution": (
                    selected["contribution"] if selected else None
                ),
                "scored_contribution": top_scenario["contribution"],
                "model_score": top_scenario["score"],
                "scenarios": result["scenarios"],
            }
        )

    return {
        **population,
        "bundle": bundle,
        "routes": pd.DataFrame(routes),
    }


def profile_metrics(monthly: pd.DataFrame, profile_id: str) -> dict[str, float]:
    history = _history_for(monthly, profile_id)
    income = history["income"].astype(float)
    outflow = history["total_outflow"].astype(float)
    earlier = float(income.iloc[-12:-3].mean())
    recent = float(income.iloc[-3:].mean())
    higher_months = int((income.iloc[-6:] > max(earlier * 1.15, 1.0)).sum())
    return {
        "average_income": float(income.mean()),
        "average_outflow": float(outflow.mean()),
        "average_surplus": float((income - outflow).mean()),
        "positive_surplus_share": float(((income - outflow) > 0).mean()),
        "current_balance": float(history.iloc[-1]["closing_balance"]),
        "cash_buffer_months": float(
            history.iloc[-1]["closing_balance"]
            / max(float(history["essential_outflow"].mean()), 1.0)
        ),
        "recent_income_ratio": recent / max(earlier, 1.0),
        "higher_income_months": float(higher_months),
    }


def monthly_chart_data(monthly: pd.DataFrame, profile_id: str) -> pd.DataFrame:
    history = _history_for(monthly, profile_id).copy()
    history["month"] = pd.to_datetime(history["month"])
    return history.set_index("month")[["income", "total_outflow", "closing_balance"]]


def transaction_timeline(transactions: pd.DataFrame, profile_id: str) -> pd.DataFrame:
    history = transactions.loc[
        (transactions["profile_id"] == profile_id)
        & (transactions["month_index"] < HISTORY_MONTHS)
    ].copy()
    return history.sort_values(["date", "category"])
