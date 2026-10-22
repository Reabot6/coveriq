"""Interactive CoverIQ product-matching demo."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from chinedu_data import CUSTOMER, build_transactions
from features import calculate_features, feature_rows
from products import GOAL_LABELS
from recommend import recommend
from train_model import train_model


st.set_page_config(page_title="CoverIQ", page_icon="C", layout="wide")

HIGH_INCOME_DURATIONS = (3, 6, 9, 12)
SESSION_KEYS = ("scene", "settings", "result", "consent", "goals", "raise_duration")


@st.cache_resource
def load_model():
    """Fit once and reuse the model while the demo session is running."""
    return train_model()


def restart_demo() -> None:
    for key in SESSION_KEYS:
        st.session_state.pop(f"coveriq_{key}", None)


def customer_transactions(settings: dict[str, Any]) -> list[dict[str, Any]]:
    return build_transactions(
        stable_months_after_raise=settings["high_income_months"]
    )


def show_customer_setup() -> None:
    st.title("CoverIQ")
    st.caption("Chinedu · Customer profile")

    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.subheader("Offer preferences")
        consent = st.checkbox(
            "Allow CoverIQ to review this account's activity for the offers selected here.",
            key="coveriq_consent",
        )
        st.multiselect(
            "Which offers would Chinedu like to explore?",
            options=list(GOAL_LABELS.values()),
            default=[GOAL_LABELS[CUSTOMER["stated_goal"]]],
            disabled=not consent,
            key="coveriq_goals",
        )

    with right:
        st.subheader("Income change")
        high_income_months = st.select_slider(
            "How many months has the higher salary continued?",
            options=HIGH_INCOME_DURATIONS,
            value=3,
            disabled=not consent,
            key="coveriq_raise_duration",
        )
        st.caption(
            "This changes the salary history CoverIQ reviews. The rest of the account history stays the same."
        )

    if st.button("Review account activity", type="primary", disabled=not consent):
        st.session_state["coveriq_settings"] = {
            "consent": consent,
            "selected_goals": list(st.session_state.get("coveriq_goals", [])),
            "high_income_months": high_income_months,
        }
        st.session_state["coveriq_scene"] = "review"
        st.rerun()


def run_review(settings: dict[str, Any]) -> dict[str, Any]:
    transactions = customer_transactions(settings)
    status_area = st.empty()
    with status_area.status("CoverIQ is reviewing the account", expanded=True) as status:
        status.write("Reading 36 months of credits and debits")
        features = calculate_features(
            transactions,
            opening_balance=CUSTOMER["opening_balance"],
        )
        status.write("Calculating income regularity, spending, surplus, and balance trends")
        status.write("Loading the cash-flow model")
        model, _ = load_model()
        status.write("Estimating whether positive monthly cash flow will continue")
        decision = recommend(
            features,
            model,
            consent=settings["consent"],
            preferred_goals=settings["selected_goals"],
        )
        status.write("Matching Chinedu's selected interests to the product catalogue")
        status.update(label="Account review complete", state="complete", expanded=False)

    return {
        "transactions": transactions,
        "features": features,
        "timeline": feature_rows(transactions, CUSTOMER["opening_balance"]),
        "decision": decision,
    }


def show_review(settings: dict[str, Any]) -> None:
    st.title("CoverIQ · Background review")
    st.caption("Chinedu · 36 months of account activity")

    transactions = customer_transactions(settings)
    features = calculate_features(
        transactions,
        opening_balance=CUSTOMER["opening_balance"],
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Salary received",
        f"{features['income_presence_rate']:.0%} of months",
    )
    metric_columns[1].metric(
        "Months with surplus",
        f"{features['positive_surplus_rate']:.0%} of recent months",
    )
    metric_columns[2].metric(
        "Average monthly surplus",
        f"₦{features['mean_monthly_surplus']:,.0f}",
    )
    metric_columns[3].metric(
        "Balance buffer",
        f"{features['liquidity_buffer_months']:.1f} months",
    )

    st.markdown("#### Account activity")
    st.dataframe(
        pd.DataFrame(feature_rows(transactions, CUSTOMER["opening_balance"])),
        hide_index=True,
        width="stretch",
        height=310,
        column_config={
            "Salary credit": st.column_config.NumberColumn(format="₦%,.0f"),
            "Other credits": st.column_config.NumberColumn(format="₦%,.0f"),
            "Debits": st.column_config.NumberColumn(format="₦%,.0f"),
            "End balance": st.column_config.NumberColumn(format="₦%,.0f"),
        },
    )

    if st.button("Run CoverIQ review", type="primary", width="stretch"):
        result = run_review(settings)
        st.session_state["coveriq_result"] = result
        st.rerun()

    if st.button("Back to Chinedu's preferences"):
        st.session_state["coveriq_scene"] = "setup"
        st.rerun()


def show_result(result: dict[str, Any]) -> None:
    features = result["features"]
    decision = result["decision"]

    st.title("CoverIQ · Recommendation")
    st.caption("Chinedu · Cash-flow review and selected interests")

    selected_products = decision.get("products", [])
    if selected_products:
        st.markdown("**Selected for consideration**")
        st.write(" · ".join(selected_products))

    score = decision.get("probability_surplus_persists")
    score_label = "Not calculated" if score is None else f"{score:.0%}"
    score_columns = st.columns([1, 2])
    with score_columns[0]:
        st.metric(
            "Positive cash-flow estimate",
            score_label,
            help="Estimated chance that monthly credits exceed debits in each of the next three months.",
        )
    with score_columns[1]:
        st.markdown("#### CoverIQ's action")
        action_labels = {
            "recommend": "Show the selected offer for review",
            "wait": "Wait and review again later",
            "suppress": "Hold back the offer",
        }
        if decision["action"] == "recommend":
            st.success(action_labels[decision["action"]])
        elif decision["action"] == "wait":
            st.warning(action_labels[decision["action"]])
        else:
            st.info(action_labels[decision["action"]])
        st.write(decision["reason"])

    st.markdown("#### Signals behind the decision")
    signal_columns = st.columns(4)
    signal_columns[0].metric(
        "Income regularity", f"{features['income_presence_rate']:.0%}"
    )
    signal_columns[1].metric(
        "Positive-surplus months", f"{features['positive_surplus_rate']:.0%}"
    )
    signal_columns[2].metric(
        "Average monthly surplus", f"₦{features['mean_monthly_surplus']:,.0f}"
    )
    signal_columns[3].metric(
        "Recurring outflows", f"{features['obligation_to_income_ratio']:.0%} of income"
    )

    with st.expander("View the 36-month account history"):
        st.dataframe(
            pd.DataFrame(result["timeline"]),
            hide_index=True,
            width="stretch",
            height=420,
            column_config={
                "Salary credit": st.column_config.NumberColumn(format="₦%,.0f"),
                "Other credits": st.column_config.NumberColumn(format="₦%,.0f"),
                "Debits": st.column_config.NumberColumn(format="₦%,.0f"),
                "End balance": st.column_config.NumberColumn(format="₦%,.0f"),
            },
        )

    buttons = st.columns(2)
    with buttons[0]:
        if st.button("Change preferences"):
            st.session_state.pop("coveriq_result", None)
            st.session_state["coveriq_scene"] = "setup"
            st.rerun()
    with buttons[1]:
        st.button("Restart", on_click=restart_demo)


def show_app() -> None:
    scene = st.session_state.get("coveriq_scene", "setup")
    settings = st.session_state.get("coveriq_settings")
    result = st.session_state.get("coveriq_result")

    if scene == "setup":
        show_customer_setup()
    elif scene == "review" and settings:
        if result is None:
            show_review(settings)
        else:
            show_result(result)
    else:
        st.session_state["coveriq_scene"] = "setup"
        st.rerun()


show_app()
