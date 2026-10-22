"""Streamlit demo for CoverIQ's cash-flow based offer timing."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from chinedu_data import CUSTOMER, build_transactions
from features import calculate_features, feature_rows
from products import GOAL_LABELS
from recommend import recommend
from train_model import train_model


st.set_page_config(page_title="CoverIQ", page_icon="◈", layout="wide")

MINIMUM_HIGH_INCOME_MONTHS = (3, 6, 9)
STATE_KEYS = (
    "scene",
    "settings",
    "result",
    "consent",
    "goals",
    "raise_duration",
)


@st.cache_resource
def load_model():
    """Train once per app session and reuse the fitted model."""
    return train_model()


def restart_demo() -> None:
    for key in STATE_KEYS:
        st.session_state.pop(f"coveriq_{key}", None)


def show_intro() -> None:
    st.title("CoverIQ")
    st.subheader("Relevant offers. Better timing.")

    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.markdown("#### Product discovery")
            st.write("Customers may see an offer before knowing whether it fits their situation.")
    with right:
        with st.container(border=True):
            st.markdown("#### CoverIQ")
            st.write("With permission, it reviews financial patterns and decides whether to show, wait, or hold back.")

    st.caption("Insurance is the example. The same matching layer can support other products.")
    if st.button("Start demo", type="primary", width="stretch"):
        st.session_state["coveriq_scene"] = "setup"
        st.rerun()


def show_setup() -> None:
    st.title("Chinedu")
    st.caption("Fictional customer · Prudential Zenith Life example")

    consent = st.checkbox(
        "I agree to use Chinedu’s fictional bank history for this demo analysis.",
        key="coveriq_consent",
    )
    selected_goals = st.multiselect(
        "What offers does Chinedu want to hear about?",
        options=list(GOAL_LABELS.values()),
        default=[GOAL_LABELS[CUSTOMER["stated_goal"]]],
        disabled=not consent,
        key="coveriq_goals",
    )
    high_income_months = st.select_slider(
        "Months at the higher income level",
        options=MINIMUM_HIGH_INCOME_MONTHS,
        value=3,
        disabled=not consent,
        key="coveriq_raise_duration",
    )

    st.caption("Connects a  account ")
    if st.button("Connect demo account", type="primary", disabled=not consent):
        st.session_state["coveriq_settings"] = {
            "consent": consent,
            "selected_goals": selected_goals,
            "high_income_months": high_income_months,
        }
        st.session_state["coveriq_scene"] = "review"
        st.rerun()


def run_review(settings: dict[str, Any]) -> dict[str, Any]:
    transactions = build_transactions(
        stable_months_after_raise=settings["high_income_months"]
    )
    status_area = st.empty()
    with status_area.status("Reviewing the demo history", expanded=True) as status:
        status.write("Reading 36 months of  credits and debits.")
        features = calculate_features(
            transactions,
            stated_goal=CUSTOMER["stated_goal"],
            opening_balance=CUSTOMER["opening_balance"],
        )
        status.write("Calculating income, spending, surplus, and balance signals.")
        status.write("Fitting the cash-flow model on simulated training histories.")
        model, model_report = load_model()
        status.write("Forecasting whether positive cash flow persists over the next three months.")
        decision = recommend(
            features,
            model,
            consent=settings["consent"],
            preferred_goals=settings["selected_goals"],
        )
        status.write("Checking the selected interests against the offer catalogue.")
        status.update(label="Review complete · no offer sent", state="complete", expanded=False)

    return {
        "transactions": transactions,
        "features": features,
        "timeline": feature_rows(transactions, CUSTOMER["opening_balance"]),
        "decision": decision,
        "model_report": model_report,
    }


def show_result(result: dict[str, Any]) -> None:
    features = result["features"]
    decision = result["decision"]
    st.title("CoverIQ · Review")
    st.caption("Chinedu · 36 months analyzed · no offer sent")

    metric_columns = st.columns(4)
    metric_columns[0].metric("Income received", f"{features['income_presence_rate']:.0%} of months")
    metric_columns[1].metric("Positive surplus", f"{features['positive_surplus_rate']:.0%} of months")
    metric_columns[2].metric("Average monthly surplus", f"₦{features['mean_monthly_surplus']:,.0f}")
    metric_columns[3].metric("Liquidity buffer", f"{features['liquidity_buffer_months']:.1f} months")

    action_titles = {
        "recommend": "Suggested for review",
        "wait": "Wait for more history",
        "suppress": "Hold back this offer",
    }
    if decision["action"] == "recommend":
        st.success(action_titles[decision["action"]])
        for product_name in decision["products"]:
            st.subheader(product_name)
    elif decision["action"] == "wait":
        st.warning(action_titles[decision["action"]])
    else:
        st.info(action_titles[decision["action"]])

    if decision.get("probability_surplus_persists") is not None:
        score_column, reason_column = st.columns([1, 2])
        with score_column:
            st.metric(
                "Cash-flow persistence estimate",
                f"{decision['probability_surplus_persists']:.0%}",
                help="Model estimate for positive net cash flow in all of the next three simulated months.",
            )
        with reason_column:
            st.markdown("**Why this action?**")
            st.write(decision["reason"])
    else:
        st.write(decision["reason"])

    st.markdown("#### Transaction history")
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

    with st.expander("Model details"):
        report = result["model_report"]
        st.write(
            f"The logistic regression was trained on {report['training_profiles']:,} "
            f"simulated profiles and evaluated on {report['held_out_profiles']:,} held-out profiles."
        )
        st.write(
            f"Held-out ROC-AUC: {report['roc_auc']:.3f} · "
            f"majority baseline accuracy: {report['baseline_accuracy']:.3f} · "
            f"model accuracy: {report['accuracy']:.3f}"
        )
        st.caption(
            "Training outcomes are simulated cash-flow patterns.  "
            
        )

    if st.button("Change consent or interests"):
        st.session_state.pop("coveriq_result", None)
        st.session_state["coveriq_scene"] = "setup"
        st.rerun()
    st.button("Restart demo", on_click=restart_demo)


def show_review() -> None:
    settings = st.session_state.get("coveriq_settings")
    if not settings:
        st.session_state["coveriq_scene"] = "setup"
        st.rerun()
        return

    result = st.session_state.get("coveriq_result")
    if result is None:
        st.title("CoverIQ · Account review")
        if st.button("Analyze Chinedu’s history", type="primary", width="stretch"):
            result = run_review(settings)
            st.session_state["coveriq_result"] = result
            st.rerun()
        if st.button("Back to consent and interests"):
            st.session_state["coveriq_scene"] = "setup"
            st.rerun()
        return

    show_result(result)


scene = st.session_state.get("coveriq_scene", "intro")
if scene == "setup":
    show_setup()
elif scene == "review":
    show_review()
else:
    show_intro()
