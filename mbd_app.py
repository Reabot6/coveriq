"""Streamlit demo for the MBD-trained CoverIQ product-ranking model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from mbd_training import explain_profile, rank_profile


MODEL_PATH = Path("models/mbd_propensity.joblib")
REPORT_PATH = Path("models/mbd_evaluation.json")

st.set_page_config(page_title="CoverIQ · Product matching", page_icon="C", layout="wide")


@st.cache_resource
def load_model_bundle():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_evaluation():
    with REPORT_PATH.open(encoding="utf-8") as report_file:
        return json.load(report_file)


def show_setup_message() -> None:
    st.title("CoverIQ · Product matching")
    st.write("The MBD model has not been trained on this machine yet.")
    st.code(
        ".\\.venv\\Scripts\\python.exe mbd_training.py "
        "--data-dir data/mbd-mini --customers-per-fold 10000 --test-fold 4",
        language="powershell",
    )
    st.caption("Training reads MBD-mini transaction and product-outcome archives.")


def safe_count(value) -> int:
    return int(value) if pd.notna(value) else 0


def show_model_metrics(report: dict) -> None:
    st.subheader("Held-out model performance")
    rows = []
    for target, metrics in report["products"].items():
        rows.append(
            {
                "Product label": target.replace("_", " ").title(),
                "Outcome rate": metrics["positive_rate"],
                "ROC-AUC": metrics["roc_auc"],
                "Average precision": metrics["average_precision"],
                "Prevalence baseline": metrics["average_precision_baseline"],
                "Top 5% lift": metrics["top_5_percent_lift_over_random"],
            }
        )
    frame = pd.DataFrame(rows)
    st.dataframe(
        frame,
        hide_index=True,
        width="stretch",
        column_config={
            "Outcome rate": st.column_config.NumberColumn(format="percent"),
            "ROC-AUC": st.column_config.NumberColumn(format="%.3f"),
            "Average precision": st.column_config.NumberColumn(format="percent"),
            "Prevalence baseline": st.column_config.NumberColumn(format="percent"),
            "Top 5% lift": st.column_config.NumberColumn(format="%.2fx"),
        },
    )
    st.caption(
        f"{report['training_clients']:,} training accounts · "
        f"{report['held_out_clients']:,} held-out accounts · "
        f"{report['training_rows']:,} training months · "
        f"{report['held_out_rows']:,} held-out months"
    )
    st.caption(report["demo_profile_sampling"])


def main() -> None:
    st.title("CoverIQ · Product matching")
    st.caption("Transaction patterns in · ranked product candidates out")
    st.caption("The source dataset keeps its four product names anonymous.")
    st.caption("Scores rank purchase patterns. Above-baseline scores are routed for review, not treated as eligibility or calibrated affordability estimates.")

    if not MODEL_PATH.exists() or not REPORT_PATH.exists():
        show_setup_message()
        return

    bundle = load_model_bundle()
    report = load_evaluation()
    profiles = bundle["demo_profiles"]
    profile_numbers = profiles["profile_number"].tolist()

    st.subheader("Choose accounts to review")
    chosen = st.multiselect(
        "Held-out account profiles",
        options=profile_numbers,
        default=profile_numbers[:5],
        format_func=lambda number: f"Account {number:02d}",
    )
    if not chosen:
        st.info("Select at least one account to run product matching.")
        show_model_metrics(report)
        return

    if st.button("Run product matching", type="primary", width="stretch"):
        rows = []
        by_number = profiles.set_index("profile_number")
        with st.status("Reviewing account activity", expanded=True) as status:
            st.write("Loading each account's transaction summary")
            st.write("Scoring the four product outcomes with the trained models")
            st.write("Ranking candidates and comparing with the recorded next-month outcome")
            for number in chosen:
                profile = by_number.loc[number]
                ranking = rank_profile(bundle, profile)
                candidates = [item for item in ranking if item["above_training_baseline"]]
                top_match = candidates[0] if candidates else ranking[0]
                rows.append(
                    {
                        "Account": f"Account {number:02d}",
                        "Transactions (12 months)": safe_count(profile.get("transaction_count", 0)),
                        "Active months (of 12)": safe_count(profile.get("active_months", 0)),
                        "Decision": "Recommend for review" if candidates else "Hold back",
                        "Top-ranked label": top_match["product"].replace("_", " ").title(),
                "Model score": top_match["propensity_score"],
                        "Observed next-month outcome": (
                            "Purchased" if top_match["observed_next_month_purchase"] else "Not purchased"
                        ),
                    }
                )
            status.update(label="Matching complete", state="complete", expanded=False)
        st.session_state["mbd_ranked_accounts"] = rows

    rows = st.session_state.get("mbd_ranked_accounts")
    if rows:
        visible_rows = [row for row in rows if int(row["Account"].split()[-1]) in chosen]
        if visible_rows:
            st.subheader("Model routing decisions")
            baseline, ranked = st.columns(2)
            baseline.metric("If all four were shown", f"{len(visible_rows) * 4} offer slots")
            recommendations = sum(row["Decision"] == "Recommend for review" for row in visible_rows)
            ranked.metric("Accounts with a candidate", f"{recommendations} for review")
            st.dataframe(
                pd.DataFrame(visible_rows),
                hide_index=True,
                width="stretch",
                column_config={
                    "Model score": st.column_config.NumberColumn(format="percent"),
                },
            )
            st.caption("The recorded next-month outcome is shown for benchmark comparison after scoring.")

            by_number = profiles.set_index("profile_number")
            for row in visible_rows:
                number = int(row["Account"].split()[-1])
                profile = by_number.loc[number]
                ranking = rank_profile(bundle, profile)
                with st.expander(f"Account {number:02d} · full ranking"):
                    ranking_frame = pd.DataFrame(
                        [
                            {
                                "Product label": item["product"].replace("_", " ").title(),
                                "Model score": item["propensity_score"],
                                "Training baseline": item["training_prevalence"],
                                "Decision": (
                                    "Above baseline"
                                    if item["above_training_baseline"]
                                    else "At/below baseline"
                                ),
                                "Observed next-month outcome": (
                                    "Purchased" if item["observed_next_month_purchase"] else "Not purchased"
                                ),
                            }
                            for item in ranking
                        ]
                    )
                    st.dataframe(
                        ranking_frame,
                        hide_index=True,
                        width="stretch",
                        column_config={
                            "Model score": st.column_config.NumberColumn(format="percent"),
                            "Training baseline": st.column_config.NumberColumn(format="percent"),
                        },
                    )
                    candidates = [item for item in ranking if item["above_training_baseline"]]
                    top_product = (candidates[0] if candidates else ranking[0])["product"]
                    signals = explain_profile(bundle, profile, top_product)
                    st.caption("Largest positive signals in the top product score")
                    if signals:
                        st.dataframe(
                            pd.DataFrame(signals),
                            hide_index=True,
                            width="stretch",
                            column_config={
                                "log_odds_contribution": st.column_config.NumberColumn(
                                    format="%.3f"
                                ),
                            },
                        )
                    else:
                        st.write("No individual signal raised the score above its model average.")
                    st.caption("Structured recommendation result")
                    st.json(
                        {
                            "profile_key": f"account_{number:02d}",
                            "next_best_action": (
                                "recommend_for_review" if candidates else "hold_back"
                            ),
                            "reason": (
                                "At least one model score exceeds its training-fold purchase rate."
                                if candidates
                                else "All model scores are at or below their product baselines."
                            ),
                            "ranked_products": [
                                {
                                    "product_id": item["product"],
                                    "model_score": round(item["propensity_score"], 6),
                                    "training_baseline": round(item["training_prevalence"], 6),
                                    "above_training_baseline": item["above_training_baseline"],
                                    "selected_for_review": bool(
                                        candidates and item["product"] == candidates[0]["product"]
                                    ),
                                }
                                for item in ranking
                            ],
                        },
                        expanded=False,
                    )
                    st.caption(
                        f"{safe_count(profile.get('transaction_count', 0)):,} transactions · "
                        f"{safe_count(profile.get('active_months', 0))} active months · "
                        f"{safe_count(profile.get('distinct_event_types', 0))} observed transaction types"
                    )
            st.caption("These are recommendation results for review; the demo sends no offers.")

    st.divider()
    show_model_metrics(report)
    st.caption(
        "Product labels are anonymous in MBD. Scores rank next-month purchase patterns; "
        "they are not insurance eligibility decisions, and this demo sends no offers."
    )


if __name__ == "__main__":
    main()
