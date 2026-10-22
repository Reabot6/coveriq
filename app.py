"""Interactive CoverIQ catalogue-matching demo."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from affordability import stress_test
from chinedu_data import CUSTOMER, build_transactions
from features import calculate_features, feature_rows
from products import GOAL_LABELS, PRODUCTS
from recommend import recommend_catalogue
from train_model import train_model


st.set_page_config(page_title="CoverIQ", page_icon="C", layout="wide")

HIGH_INCOME_DURATIONS = (3, 6, 9, 12)
PRODUCT_KEYS = list(PRODUCTS)
OFFER_OPTIONS = list(GOAL_LABELS.values())
DEFAULT_PREMIUMS = {"my_smart_save": 5_000, "my_edusave": 100_000, "my_life_plan": 50_000}
SESSION_KEYS = (
    "scene", "settings", "result", "consent", "raise_duration", "offer",
    "frequency", "premium_my_smart_save", "premium_my_edusave", "premium_my_life_plan",
)


@st.cache_resource
def load_model():
    """Train once on the documented simulator and reuse during the session."""
    return train_model()


def restart_demo() -> None:
    for key in SESSION_KEYS:
        st.session_state.pop(f"coveriq_{key}", None)


def customer_transactions(settings: dict[str, Any]) -> list[dict[str, Any]]:
    return build_transactions(stable_months_after_raise=settings["high_income_months"])


def analyze(settings: dict[str, Any]) -> dict[str, Any]:
    transactions = customer_transactions(settings)
    features = calculate_features(transactions, opening_balance=CUSTOMER["opening_balance"])
    model, report = load_model()
    matches = recommend_catalogue(
        features,
        settings["premiums"],
        settings["payment_frequency"],
        model,
        consent=settings["consent"],
        selected_interests=settings["selected_interests"],
    )
    scenarios = {
        offer["key"]: stress_test(
            transactions,
            CUSTOMER["opening_balance"],
            settings["premiums"][offer["key"]],
            settings["payment_frequency"],
        )
        for offer in matches
        if offer["action"] in {"top_match", "also_fits", "wait", "hold"}
    }
    return {
        "transactions": transactions,
        "features": features,
        "timeline": feature_rows(transactions, CUSTOMER["opening_balance"]),
        "matches": matches,
        "scenarios": scenarios,
        "evaluation": report,
    }


def show_customer_setup() -> None:
    st.title("CoverIQ")
    st.caption(f"{CUSTOMER['name']} · Customer profile")

    left, right = st.columns([1, 1.2], gap="large")
    with left:
        st.subheader("Offer preferences")
        consent = st.checkbox(
            "Allow CoverIQ to review this account activity for selected offers.",
            key="coveriq_consent",
        )
        selected_interests = st.multiselect(
            "Insurance interests to check",
            OFFER_OPTIONS,
            default=[GOAL_LABELS[CUSTOMER["stated_goal"]]],
            disabled=not consent,
            key="coveriq_offer",
            help="Choose one interest, add a second, or select all three.",
        )
        payment_frequency = st.radio(
            "Payment frequency",
            ("monthly", "annual"),
            horizontal=True,
            disabled=not consent,
            key="coveriq_frequency",
        )

    selected_product_keys = [
        key for key in PRODUCT_KEYS
        if GOAL_LABELS[PRODUCTS[key]["goal"]] in selected_interests
    ]
    premium_columns = st.columns(max(1, len(selected_product_keys)))
    premiums: dict[str, float] = dict(DEFAULT_PREMIUMS)
    for column, product_key in zip(premium_columns, selected_product_keys):
        product = PRODUCTS[product_key]
        with column:
            premiums[product_key] = float(st.number_input(
                f"{product['name']} test premium (₦)",
                min_value=1_000,
                max_value=10_000_000,
                value=DEFAULT_PREMIUMS[product_key],
                step=1_000,
                disabled=not consent,
                key=f"coveriq_premium_{product_key}",
            ))
    if selected_product_keys:
        st.caption("Test premiums are comparison inputs, not insurer quotes.")

    with right:
        st.subheader("Recent income change")
        high_income_months = st.select_slider(
            "Months at the higher salary",
            options=HIGH_INCOME_DURATIONS,
            value=3,
            disabled=not consent,
            key="coveriq_raise_duration",
        )
        st.caption("This changes the salary history used in the review.")

    if st.button(
        "Review account activity",
        type="primary",
        disabled=not consent or not selected_interests,
    ):
        st.session_state["coveriq_settings"] = {
            "consent": consent,
            "selected_interests": list(selected_interests),
            "premiums": premiums,
            "payment_frequency": payment_frequency,
            "high_income_months": high_income_months,
        }
        st.session_state["coveriq_scene"] = "review"
        st.rerun()


def show_review(settings: dict[str, Any]) -> None:
    st.title("CoverIQ · Account activity")
    st.caption(f"{CUSTOMER['name']} · 36 months reviewed")
    transactions = customer_transactions(settings)
    features = calculate_features(transactions, opening_balance=CUSTOMER["opening_balance"])

    metrics = st.columns(4)
    metrics[0].metric("Income received", f"{features['income_presence_rate']:.0%} of months")
    metrics[1].metric("Months with surplus", f"{features['positive_surplus_rate']:.0%}")
    metrics[2].metric("Average monthly money left", f"₦{features['mean_monthly_surplus']:,.0f}")
    metrics[3].metric("Current balance", f"₦{features['current_balance']:,.0f}")

    st.dataframe(
        pd.DataFrame(feature_rows(transactions, CUSTOMER["opening_balance"])),
        hide_index=True,
        width="stretch",
        height=300,
        column_config={
            "Salary credit": st.column_config.NumberColumn(format="₦%,.0f"),
            "Other credits": st.column_config.NumberColumn(format="₦%,.0f"),
            "Debits": st.column_config.NumberColumn(format="₦%,.0f"),
            "End balance": st.column_config.NumberColumn(format="₦%,.0f"),
        },
    )

    if st.button("Run catalogue review", type="primary", width="stretch"):
        with st.status("Checking the selected offers", expanded=True) as status:
            st.write("Summarizing income, recurring bills, and money left after spending")
            st.write("Retrieving catalogue offers that match the selected interests")
            result = analyze(settings)
            st.write("Testing each proposed premium against recent and stressed cash flow")
            status.update(label="Catalogue review complete", state="complete", expanded=False)
        st.session_state["coveriq_result"] = result
        st.rerun()

    if st.button("Change interests or test premiums"):
        st.session_state["coveriq_scene"] = "setup"
        st.rerun()


def show_result(result: dict[str, Any], settings: dict[str, Any]) -> None:
    features = result["features"]
    matches = result["matches"]
    best = next((offer for offer in matches if offer["action"] == "top_match"), None)

    st.title("CoverIQ · Catalogue match")
    st.caption(f"{CUSTOMER['name']} · {len(settings['selected_interests'])} selected interests")
    metrics = st.columns(3)
    metrics[0].metric("Typical monthly income", f"₦{features['typical_monthly_income']:,.0f}")
    metrics[1].metric("Average monthly money left", f"₦{features['mean_monthly_surplus']:,.0f}")
    metrics[2].metric("Surplus before recent raise", f"₦{features['baseline_monthly_surplus']:,.0f}")

    if best:
        st.success(f"Top match for review: {best['name']}")
        st.write(best["reason"])
    elif settings["selected_interests"]:
        st.warning("No selected offer passed the current cash-flow checks.")
    else:
        st.info("Select at least one insurance interest to search the catalogue.")

    st.subheader("Catalogue results")
    status_names = {
        "top_match": "Top match",
        "also_fits": "Also fits",
        "wait": "Wait",
        "hold": "Hold back",
        "not_requested": "Not selected",
    }
    for offer in matches:
        with st.container(border=True):
            title_col, status_col = st.columns([3, 1])
            title_col.markdown(f"**{offer['name']}**")
            status_col.markdown(f"**{status_names[offer['action']]}**")
            st.caption(offer["summary"])

            details = st.columns(3)
            details[0].metric("Catalogue relevance", f"{offer['relevance_score']:.0%}")
            amount = offer["premium_amount"]
            period = "year" if settings["payment_frequency"] == "annual" else "month"
            details[1].metric("Test premium", f"₦{amount:,.0f} / {period}")
            probability = offer["probability_sustainable"]
            details[2].metric(
                "12-month estimate",
                "Not checked" if probability is None else f"{probability:.0%}",
            )
            st.write(offer["reason"])

            if offer["key"] in result["scenarios"]:
                with st.expander("See projected balances"):
                    scenario_frame = pd.DataFrame(result["scenarios"][offer["key"]])
                    for column in ("lowest_balance", "reserve_floor"):
                        scenario_frame[column] = scenario_frame[column].map(
                            lambda value: f"₦{value:,.0f}"
                        )
                    scenario_frame = scenario_frame.rename(columns={
                        "scenario": "Cash-flow scenario",
                        "lowest_balance": "Lowest balance",
                        "months_below_reserve": "Months below bills reserve",
                        "reserve_floor": "One-month bills reserve",
                    })
                    st.dataframe(scenario_frame, hide_index=True, width="stretch")

    st.caption(
        "The model learns from generated scenarios. Catalogue retrieval compares the selected "
        "interests with the available offer descriptions."
    )

    with st.expander("Model evaluation on held-out generated profiles"):
        report = result["evaluation"]
        st.write(
            f"ROC-AUC {report['roc_auc']:.2f} · balanced accuracy "
            f"{report['balanced_accuracy']:.2f} · Brier score {report['brier_score']:.2f}"
        )
        st.caption(report["training_data_note"])

    with st.expander("View 36-month account history"):
        st.dataframe(pd.DataFrame(result["timeline"]), hide_index=True, width="stretch", height=380)

    buttons = st.columns(2)
    with buttons[0]:
        if st.button("Adjust catalogue review"):
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
            show_result(result, settings)
    else:
        st.session_state["coveriq_scene"] = "setup"
        st.rerun()


show_app()
