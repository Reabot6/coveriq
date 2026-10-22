"""Streamlit interface for the CoverIQ insurance-routing proof of concept."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from cashflow_model import (
    CONTRIBUTION_SCENARIOS,
    ROUTING_THRESHOLD,
    build_demo,
    explain_score,
    monthly_chart_data,
    profile_metrics,
    transaction_timeline,
)


PRODUCT_NAME = "My Smart Save"
PRODUCT_PROVIDER = "Prudential Zenith Life"
PRODUCT_SUMMARY = "A long-term savings plan with life cover."
PRODUCT_SOURCE = "https://campaign.prudentialzenith.com/dream-big-invest-in-your-tomorrow/"

st.set_page_config(page_title="CoverIQ · Insurance routing", page_icon="C", layout="wide")


def money(value: float) -> str:
    return f"₦{value:,.0f}"


def _route_table(routes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for route in routes.itertuples(index=False):
        chosen = route.selected_contribution
        rows.append(
            {
                "Account": route.customer,
                "Top comparison amount": money(route.scored_contribution),
                "Model estimate": f"{route.model_score:.0%}",
                "Offer for review": money(chosen) if pd.notna(chosen) else "—",
                "Bank action": (
                    "Review offer" if route.action == "surface_for_review" else "Hold back"
                ),
            }
        )
    return pd.DataFrame(rows)


def show_batch_summary(demo: dict) -> None:
    routes = demo["routes"]
    routed = routes[routes["action"] == "surface_for_review"]
    held = routes[routes["action"] == "hold_back"]

    st.subheader("Bank review queue")
    first, second, third = st.columns(3)
    first.metric("Accounts scanned", f"{len(routes)}")
    second.metric("Offers to review", f"{len(routed)}")
    third.metric("Held back", f"{len(held)}")
    st.dataframe(_route_table(routes), hide_index=True, width="stretch")

    st.subheader("Insurance partner summary")
    st.caption("Aggregate routing counts; customer-level account mapping stays with the bank.")
    counts = (
        routed.groupby("selected_contribution", dropna=False)
        .size()
        .reindex(CONTRIBUTION_SCENARIOS, fill_value=0)
    )
    partner_rows = [
        {
            "Illustrative monthly contribution": money(amount),
            "Accounts routed for bank review": int(counts.loc[amount]),
        }
        for amount in CONTRIBUTION_SCENARIOS
    ]
    partner_rows.append(
        {"Illustrative monthly contribution": "Held back", "Accounts routed for bank review": len(held)}
    )
    st.dataframe(pd.DataFrame(partner_rows), hide_index=True, width="stretch")


def show_customer_detail(demo: dict) -> None:
    routes = demo["routes"]
    profiles = demo["profiles"].set_index("profile_id")
    display_to_id = profiles.loc[
        profiles["is_demo"], "display_name"
    ].to_dict()
    display_names = list(display_to_id.values())
    default_index = display_names.index("Chinedu") if "Chinedu" in display_names else 0
    selected_name = st.selectbox(
        "Inspect a bank-side account",
        display_names,
        index=default_index,
    )
    profile_id = next(key for key, name in display_to_id.items() if name == selected_name)

    monthly = demo["monthly"]
    history = monthly.loc[
        (monthly["profile_id"] == profile_id)
        & (monthly["month_index"] < 36)
    ].sort_values("month_index")
    details = profile_metrics(monthly, profile_id)
    route = routes.loc[routes["profile_id"] == profile_id].iloc[0]

    st.markdown(f"### {selected_name} · 36-month account review")
    a, b, c, d = st.columns(4)
    a.metric("Typical monthly income", money(details["average_income"]))
    b.metric("Typical monthly outflow", money(details["average_outflow"]))
    c.metric("Average monthly remainder", money(details["average_surplus"]))
    d.metric("Current cash buffer", f"{details['cash_buffer_months']:.1f} months")

    if details["recent_income_ratio"] >= 1.15:
        change_text = (
            f"Recent income is {details['recent_income_ratio']:.1f}× its earlier level; "
            f"the higher level appears in {int(details['higher_income_months'])} of the last 6 months."
        )
    else:
        change_text = "Income has not recently moved above its earlier level."
    st.caption(change_text)

    chart = monthly_chart_data(monthly, profile_id).rename(
        columns={
            "income": "Income",
            "total_outflow": "Outflow",
            "closing_balance": "End balance",
        }
    )
    flow_chart, balance_chart = st.columns(2)
    flow_chart.markdown("**Monthly money in and out**")
    flow_chart.line_chart(chart[["Income", "Outflow"]], height=260, y_label="Naira")
    balance_chart.markdown("**End-of-month balance**")
    balance_chart.line_chart(chart[["End balance"]], height=260, y_label="Naira")

    st.markdown("#### Contribution scenarios")
    st.caption(
        f"The model estimates whether a basic cash buffer persists for the next three months. "
        f"A scenario at or above {ROUTING_THRESHOLD:.0%} is queued for bank review."
    )
    scenario_rows = []
    for item in route.scenarios:
        scenario_rows.append(
            {
                "Illustrative monthly amount": money(item["contribution"]),
                "Estimated buffer sustainability": f"{item['score']:.0%}",
                "Bank action": (
                    "Available for review"
                    if item["score"] >= ROUTING_THRESHOLD
                    else "Hold back"
                ),
            }
        )
    st.dataframe(pd.DataFrame(scenario_rows), hide_index=True, width="stretch")

    selected = route.selected_contribution
    explanation_amount = (
        float(selected)
        if pd.notna(selected)
        else CONTRIBUTION_SCENARIOS[0]
    )
    explanation = explain_score(demo["bundle"], history, explanation_amount)
    with st.expander("Why did CoverIQ route this account?"):
        if route.action == "surface_for_review":
            st.write(
                f"The highest comparison amount clearing the review threshold is "
                f"{money(float(selected))}. These are the strongest model signals for that scenario."
            )
        else:
            st.write(
                "No comparison amount cleared the review threshold. These are the strongest "
                "signals for the smallest scenario."
            )
        supports = [item["signal"] for item in explanation if item["direction"] == "supports"]
        reduces = [item["signal"] for item in explanation if item["direction"] == "reduces"]
        left, right = st.columns(2)
        left.markdown("**Supports the estimate**")
        left.write("\n".join(f"• {item}" for item in supports) or "No positive signal stood out.")
        right.markdown("**Reduces the estimate**")
        right.write("\n".join(f"• {item}" for item in reduces) or "No caution signal stood out.")

    with st.expander("Recent transaction rows"):
        timeline = transaction_timeline(demo["transactions"], profile_id)
        timeline = timeline.tail(18).copy()
        timeline["amount"] = timeline["amount"].map(money)
        timeline["balance_after"] = timeline["balance_after"].map(money)
        timeline = timeline.rename(
            columns={
                "date": "Date",
                "category": "Category",
                "direction": "Money in / out",
                "amount": "Amount",
                "balance_after": "Balance after",
            }
        )
        st.dataframe(
            timeline[["Date", "Category", "Money in / out", "Amount", "Balance after"]],
            hide_index=True,
            width="stretch",
        )


def show_model_check(demo: dict) -> None:
    metrics = demo["bundle"]["metrics"]
    st.subheader("Model check")
    st.caption(
        "Trained on generated histories and checked on separate generated accounts. "
        "This is a prototype result, not a Zenith-customer performance claim."
    )
    cols = st.columns(4)
    cols[0].metric("Training accounts", f"{metrics['training_customers']:,}")
    cols[1].metric("Held-out accounts", f"{metrics['held_out_customers']:,}")
    cols[2].metric(
        "ROC-AUC",
        f"{metrics['roc_auc']:.3f}" if metrics["roc_auc"] is not None else "—",
    )
    cols[3].metric("Brier score", f"{metrics['brier_score']:.3f}")
    st.caption(metrics["label"])
    st.caption(
        f"Held-out average precision: {metrics['average_precision']:.3f} · "
        f"positive outcome rate: {metrics['held_out_positive_rate']:.1%}"
    )


def show_app() -> None:
    st.title("CoverIQ")
    st.caption("Bank-side insurance offer routing")
    steps = st.columns(4)
    for column, number, label in zip(
        steps,
        ("01", "02", "03", "04"),
        (
            "Read transaction history",
            "Build cash-flow profile",
            "Estimate contribution sustainability",
            "Route an offer for review",
        ),
    ):
        column.markdown(f"**{number}**  \n{label}")

    st.divider()
    product, action = st.columns([2, 1], vertical_alignment="center")
    with product:
        st.subheader(f"Catalogue example · {PRODUCT_NAME}")
        st.write(f"{PRODUCT_PROVIDER} · {PRODUCT_SUMMARY}")
    with action:
        st.link_button("View product information", PRODUCT_SOURCE, width="stretch")

    st.caption("Comparison amounts are demo inputs, not insurer prices or quotes.")
    if st.button("Run customer batch", type="primary", width="stretch"):
        with st.status("Preparing the review", expanded=True) as status:
            def progress(message: str) -> None:
                status.write(message)

            demo = build_demo(progress=progress)
            st.session_state["coveriq_demo"] = demo
            status.update(label="Review complete", state="complete", expanded=False)

    demo = st.session_state.get("coveriq_demo")
    if demo is None:
        st.info("Run the customer batch to see the model build the bank review queue.")
        return

    st.divider()
    show_batch_summary(demo)
    st.divider()
    show_customer_detail(demo)
    st.divider()
    show_model_check(demo)


show_app()
