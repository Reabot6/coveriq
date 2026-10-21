# CoverIQ

CoverIQ is a Streamlit prototype for the Zecathon customer product-matching challenge. It demonstrates a consent-gated flow that maps stated customer goals to catalogue entries, forecasts synthetic cash-flow persistence, and returns recommend, wait, or suppress with an explanation.

## Demo flow

1. Why CoverIQ — states the customer problem and the business value CoverIQ aims to test. Business outcomes are hypotheses, not proven results.
2. Before — an illustrative customer-led product discovery flow. The challenge identifies irrelevant or excessive offers as a problem; this screen is not a verified copy of Zenith's current process.
3. After — Chinedu opts in, selects one or more goals, reviews 36 months of fictional activity, and sees the goal-matched catalogue entries, model estimate, decision, and simulated response.

The worked example uses Prudential Zenith Life's My Smart Save catalogue entry. Zenith General Insurance is a separate insurer. Adding its products would require distinct customer signals, product checks, and validated data; this prototype does not claim to match across both insurers.

## What the model does

synthetic_data.py simulates 36-month histories and the following six months. Its label records whether monthly net cash flow is positive in at least five future months. train_model.py fits a Random Forest on generated profiles and evaluates it on held-out generated profiles. At inference, the model estimates the same persistence target for Chinedu.

Product matching is separate and explainable: the customer's stated goal maps to product catalogue goals. recommend.py applies consent, customer interest, net-flow, recent-income-change, and model-threshold safeguards before it shows information for review. The model does not rank products.

This is more meaningful than training a classifier to repeat a label that directly says “recommend at nine months.” However, evaluation is only against our simulator. The estimate is not calibrated or validated on actual bank customers, and does not establish product suitability, affordability, eligibility, conversion, or policy persistence.

## Files

- app.py — three-stage presentation, before/after journeys, transaction history, model evidence, and simulated response.
- chinedu_data.py — fictional Chinedu and his 36-month salary-change scenarios.
- features.py — income stability, recent change, time at new salary, net flow, and balance trend features.
- synthetic_data.py — synthetic 36-month histories plus six future months and the forward cash-flow label.
- train_model.py — Random Forest training and held-out synthetic evaluation.
- recommend.py — goal-to-catalogue mapping, safeguards, and explainable action.
- products.py — demo product catalogue and goal labels.
- PROJECT_GUIDE.md — model explanation, limitations, and presenter checklist.

## Run on Windows PowerShell

From this project directory, run:

.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py

If .venv does not exist yet, create it with python -m venv .venv, then install requirements.

## Claim boundary

All customer records and model-training profiles are generated. Consent is a UI demonstration; no notification is sent, and there is no connection to Zenith Bank, Zenith General Insurance, or Prudential Zenith Life. A real deployment needs approved data access, verified product requirements, consent and privacy controls, and evaluation on authorized real outcomes. No business lift has been demonstrated.
