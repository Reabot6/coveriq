"""CoverIQ product-matching demo using fictional customer data."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from chinedu_data import CUSTOMER, build_transactions
from features import calculate_features, feature_rows
from products import GOAL_LABELS, PRODUCTS
from recommend import recommend
from train_model import train_model


st.set_page_config(page_title="CoverIQ | Product matching", page_icon="◈", layout="wide")


@st.cache_resource
def get_model():
    return train_model()


STAGES = ["Why CoverIQ", "Before · Current journey", "After · CoverIQ"]


def reset_coveriq() -> None:
    """Reset customer-visible CoverIQ state before Streamlit reruns the page."""
    st.session_state["coveriq_step"] = 0
    st.session_state["coveriq_opted_in"] = False
    st.session_state["coveriq_demo_consent"] = False
    analysis_keys = (
        "coveriq_analysis_result",
        "coveriq_analysis_features",
        "coveriq_model_report",
        "coveriq_analysis_signature",
    )
    for key in analysis_keys:
        st.session_state.pop(key, None)


def intro_page() -> None:
    st.title("CoverIQ")
    st.subheader("Relevant products. Better timing. Fewer unwanted offers.")
    st.write(
        "The Zecathon challenge describes customers being presented with multiple banking "
        "products without knowing which is most relevant to their needs. CoverIQ is a proposed "
        "recommendation service that uses customer-permitted information and financial behaviour "
        "to decide whether to recommend a product for review, wait, or suppress an offer."
    )
    customer, institution = st.columns(2)
    with customer:
        with st.container(border=True):
            st.markdown("#### Customer value")
        st.write(
            "More relevant product information, a clear reason for the offer, "
            "and a choice to opt in or decline."
        )
    with institution:
        with st.container(border=True):
            st.markdown("#### Potential business value")
            st.write(
                "Test whether better timed, better qualified referrals improve customer response "
                "and reduce irrelevant outreach. This is a hypothesis to measure, not a result we claim today."
            )
    st.markdown("#### What CoverIQ does")
    st.write(
        "It checks consent and customer interest, summarizes permitted signals, estimates whether "
        "positive cash flow may persist, applies safeguards, then returns a reasoned action."
    )
    st.info(
        "This worked example uses My Smart Save from Prudential Zenith Life. Zenith General Insurance "
        "is a separate insurer; a broader catalogue would need distinct signals and fit checks for each product."
    )
    st.caption(
        "The challenge statement supports the problem framing. It does not prove Zenith’s current "
        "campaigns are untargeted or that CoverIQ improves conversion."
    )


def before_page() -> None:
    st.title("Before · Product discovery can be hard to navigate")
    st.warning(
        "The challenge identifies the risk of multiple, irrelevant, or excessive offers. "
        "This is an illustrative baseline, not a verified copy of Zenith’s current process."
    )
    step = st.session_state.setdefault("manual_step", 0)
    st.progress(
        (step + 1) / 4,
        text=f"Illustrative step {step + 1} of 4",
    )
    if step == 0:
        st.subheader("1 · An offer appears")
        st.write("A customer may see a product message before knowing whether it fits their situation.")
        st.info("Customer question: “Why am I seeing this, and is it meant for me?”")
    elif step == 1:
        st.subheader("2 · The customer enters details")
        st.write(
            "A product journey may ask the customer to provide personal details, "
            "goals, income, or payment preferences."
        )
        left, right = st.columns(2)
        left.text_input("Name", value="Chinedu Okonkwo", disabled=True)
        right.text_input("Product of interest", value="My Smart Save", disabled=True)
        left.text_input("Monthly income entered", value="₦1,000,000", disabled=True)
        right.text_input("Time at higher salary", value="3 months", disabled=True)
        st.caption("These are fictional demo values, not confirmed application requirements.")
    elif step == 2:
        st.subheader("3 · The answers are customer-declared")
        st.write("This illustration does not compare the answers with the customer’s account history.")
        st.table(
            pd.DataFrame([
                {"Signal": "Income", "Source here": "Customer-entered"},
                {"Signal": "Time at new salary", "Source here": "Customer-entered"},
                {"Signal": "Surplus trend", "Source here": "Not checked"},
                {"Signal": "Offer relevance", "Source here": "Not assessed"},
            ]),
            hide_index=True,
            alt="Signals available in the illustrative product discovery baseline.",
        )
    else:
        st.subheader("4 · The customer decides whether to continue")
        outcome = st.radio(
            "Illustrative outcome",
            [
                "Customer pauses after an income change",
                "Customer decides it is not a fit",
                "Customer continues to separate review",
            ],
            key="manual_outcome",
        )
        st.write({
            "Customer pauses after an income change": "The form does not monitor the fictional account after the customer enters answers.",
            "Customer decides it is not a fit": "The customer has to recognize the mismatch and stop or seek another option.",
            "Customer continues to separate review": "Continuing does not mean the policy is approved or purchased.",
        }[outcome])
    left, right = st.columns(2)
    with left:
        if st.button("Back", disabled=step == 0, key="manual_back"):
            st.session_state["manual_step"] = step - 1
            st.rerun()
    with right:
        if step < 3:
            if st.button("Next", type="primary", width="stretch", key="manual_next"):
                st.session_state["manual_step"] = step + 1
                st.rerun()
        elif st.button("Restart this scene", key="manual_restart"):
            st.session_state["manual_step"] = 0
            st.rerun()


def after_page() -> None:
    st.title("After · CoverIQ checks before it suggests")
    step = st.session_state.setdefault("coveriq_step", 0)
    st.progress((step + 1) / 3, text=f"CoverIQ step {step + 1} of 3")
    if step == 0:
        st.subheader("1 · Ask permission")
        st.write("Chinedu chooses whether CoverIQ may analyze this fictional history.")
        consent = st.checkbox(
            "Allow CoverIQ to analyze this fictional transaction history for the demo.",
            key="coveriq_demo_consent",
        )
        st.caption("Demo control only; no legal consent is recorded and no bank is connected.")
        if not consent:
            st.session_state["coveriq_opted_in"] = False
            for key in ("coveriq_analysis_result", "coveriq_analysis_features", "coveriq_model_report"):
                st.session_state.pop(key, None)
        if st.button("Continue", type="primary", disabled=not consent, key="coveriq_consent_next"):
            st.session_state["coveriq_opted_in"] = True
            st.session_state["coveriq_step"] = 1
            st.rerun()
        return
    if step == 1:
        st.subheader("2 · Tell CoverIQ what matters to you")
        st.write(
            "CoverIQ uses the customer's stated goal to find matching catalogue "
            "entries. Interest alone does not establish affordability or eligibility."
        )
        goal_labels = st.multiselect(
            "Goals Chinedu is open to exploring",
            options=list(GOAL_LABELS.values()),
            default=st.session_state.get("coveriq_selected_goals", ["Long-term savings"]),
            key="coveriq_product_choices",
        )
        left, right = st.columns(2)
        with left:
            if st.button("Back to permission", key="coveriq_back_to_consent"):
                st.session_state["coveriq_step"] = 0
                st.rerun()
        with right:
            if st.button("Review account history", type="primary", disabled=not goal_labels, key="coveriq_products_next"):
                st.session_state["coveriq_selected_goals"] = goal_labels
                st.session_state["coveriq_step"] = 2
                st.rerun()
        return

    st.subheader("3 · Review the history and model evidence")
    st.write("Explore 36 months of fictional account activity and compare a recent salary change with a sustained one.")
    months = st.segmented_control(
        "Time at the higher salary", options=[3, 9],
        format_func=lambda value: f"{value} months", default=3, key="coveriq_raise_timeline",
    ) or 3
    transactions = build_transactions(months)
    history = pd.DataFrame(feature_rows(transactions, opening_balance=CUSTOMER["opening_balance"]))
    end_balance = int(history.iloc[-1]["End balance"])
    opening = int(CUSTOMER["opening_balance"])
    st.metric(
        "Illustrative balance after three years",
        f"₦{end_balance:,.0f}",
        f"₦{end_balance - opening:+,.0f} from starting balance",
    )
    st.line_chart(
        history.set_index("Month")["End balance"],
        y_label="Balance (₦)",
        alt="Fictional balance over 36 months.",
    )
    with st.expander("View monthly transaction summary"):
        st.table(history, hide_index=True, alt="Monthly salary, other credits, debits, and balances for the fictional customer.")
    selected_goals = st.session_state.get("coveriq_selected_goals", ["Long-term savings"])
    st.caption("Customer-stated goals: " + ", ".join(selected_goals))
    signature = (months, tuple(sorted(selected_goals)))
    if st.session_state.get("coveriq_analysis_signature") != signature:
        for key in ("coveriq_analysis_result", "coveriq_analysis_features", "coveriq_model_report"):
            st.session_state.pop(key, None)

    if st.button("Run CoverIQ review", type="primary", width="stretch", key="coveriq_analyze"):
        with st.status("CoverIQ is reviewing the profile", expanded=True) as status:
            st.write("Checking the demo opt-in and loading the permitted fictional history.")
            features = calculate_features(transactions, CUSTOMER["stated_goal"], CUSTOMER["opening_balance"])
            st.write("Calculating salary stability, recent change, net cash flow, and balance trend.")
            st.write("Training a forward cash-flow model on separate synthetic customer profiles.")
            model, report = get_model()
            st.write("Estimating whether monthly positive cash flow persists across at least five of six future months.")
            result = recommend(features, model, st.session_state.get("coveriq_opted_in", False), selected_goals)
            st.write("Applying timing safeguards and preparing a simulated response. No message is sent.")
            status.update(label="Review complete · no message was sent", state="complete", expanded=True)
        st.session_state["coveriq_analysis_signature"] = signature
        st.session_state["coveriq_analysis_result"] = result
        st.session_state["coveriq_analysis_features"] = features
        st.session_state["coveriq_model_report"] = report

    result = st.session_state.get("coveriq_analysis_result")
    features = st.session_state.get("coveriq_analysis_features")
    if result and features:
        st.markdown("#### Signals from the history")
        cols = st.columns(4)
        cols[0].metric("Income stability", f"{features['income_stability']:.0%}")
        cols[1].metric("Recent income change", f"{features['recent_income_change_pct']:.0%}")
        cols[2].metric("Months at new level", f"{features['months_since_income_change']:.0f}")
        cols[3].metric(
            "Recent monthly net flow",
            f"₦{features['average_monthly_surplus']:,.0f}",
        )
        st.markdown("#### CoverIQ action")
        st.write("Catalogue matches for the stated goal: " + ", ".join(result["products"]))
        st.caption("Goal matching uses the customer's stated preference. The ML model forecasts cash-flow persistence and does not rank products.")
        if result["action"] == "recommend":
            st.success(result["message"])
        elif result["action"] == "wait":
            st.warning(result["message"])
        else:
            st.info(result["message"])
        st.write(result["reason"])
        if result["probability_surplus_persists"] is not None:
            st.metric(
                "Synthetic model estimate · six-month cash-flow persistence",
                f"{result['probability_surplus_persists']:.0%}",
            )
        st.caption(
            "The model was trained and held-out-tested on generated profiles only. Its estimate is not "
            "validated on real customers, calibrated confidence, affordability, or product eligibility."
        )
        with st.expander("How the model was evaluated"):
            report = st.session_state.get("coveriq_model_report", {})
            st.write(
                f"Held-out synthetic profiles: {report.get('held_out_profiles', 0)} · "
                f"ROC AUC {report.get('roc_auc', 0):.2f} · Accuracy {report.get('accuracy', 0):.2f} · "
                f"Brier score {report.get('brier_score', 0):.2f}"
            )
            feature_names = {
                "income_stability": "Income stability",
                "recent_income_change_pct": "Recent income change",
                "months_since_income_change": "Months at higher income",
                "average_monthly_surplus": "Recent net cash flow",
                "surplus_ratio": "Surplus ratio",
                "balance_growth_ratio": "Balance trend",
                "history_months": "History length",
            }
            importance_rows = [
                {"Synthetic model feature": feature_names[name], "Relative importance": f"{score:.0%}"}
                for name, score in list(report.get("feature_importance", {}).items())[:4]
            ]
            st.table(pd.DataFrame(importance_rows), hide_index=True,
                     alt="Most important features in the synthetic model across generated training profiles.")
            st.caption("These metrics describe performance on the simulator’s data distribution only.")
        with st.expander("Simulated response to a bank channel"):
            st.json({
                "consent": bool(st.session_state.get("coveriq_opted_in", False)),
                "customer_ref": "DEMO-001",
                "action": result["action"],
                "customer_goals": result["matched_goals"],
                "matching_catalogue_products": result["products"],
                "reason": result["reason"],
                "model_target": result["model_target"],
                "synthetic_estimate": result["probability_surplus_persists"],
                "delivery": "not_sent_demo_only",
            })
    left, right = st.columns(2)
    with left:
        if st.button("Back to product interest", key="coveriq_back_to_products"):
            st.session_state["coveriq_step"] = 1
            st.rerun()
    with right:
        st.button("Restart CoverIQ", key="coveriq_restart", on_click=reset_coveriq)


st.title("CoverIQ · Demo")
st.caption("Customer-permitted product matching, demonstrated with fictional information.")
st.info(
    "No real customer data is used. There is no connection to Zenith Bank, Zenith General Insurance, "
    "or Prudential Zenith Life; no quote, eligibility decision, or message is produced."
)
stage = st.segmented_control("Demo story", options=STAGES, default=STAGES[0], key="demo_stage")
if stage == STAGES[1]:
    before_page()
elif stage == STAGES[2]:
    after_page()
else:
    intro_page()

