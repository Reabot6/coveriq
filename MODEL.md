# CoverIQ model and data flow

## Demo objective

The demo asks a narrow question: given a customer's past account activity and an illustrative monthly contribution, how likely is the account to keep a basic cash buffer over the next three months?

That estimate is used to route an insurance offer for bank review or hold it back. It is not an insurance-need prediction, eligibility decision, premium calculation, or financial advice.

## Files

- `synthetic_data.py` creates repeatable fictional transaction histories and monthly account summaries.
- `cashflow_model.py` builds customer features, creates training examples, trains and evaluates the classifier, and scores demo accounts.
- `app.py` runs the Streamlit experience and presents the bank review queue, an account timeline, contribution comparisons and aggregate partner counts.
- `README.md` has the PowerShell run steps.

## Data generated

The generator produces 36 historical months and three later months per profile. The historical period contains transaction rows for salary or business income, other income, housing, food, utilities, transport, debt payments, savings transfers and other spending. The later three months are kept out of the model features and used only to create the training outcome.

There are 1,000 generated training profiles and a separate 12-account demo batch. Chinedu's history has a salary increase in its final three months. The remaining accounts vary in income regularity, fixed outflows, recent income changes and balance patterns.

All generated records use a fixed random seed so the demo can be repeated with the same results. The generator is a simulation with assumptions; it is not a sample of real banking customers.

## Features

For each profile and contribution scenario, the feature builder summarizes the 36 historical months:

- average monthly income, income variation and share of months with income;
- the recent income level compared with the earlier history, and how many recent months show the higher level;
- typical essential outflows and their share of income;
- average monthly remainder, positive-remainder frequency and remainder variation;
- account balance trend, latest balance and cash-buffer months;
- the illustrative contribution amount relative to income and average remainder.

The records are grouped by month before modeling. Customer IDs identify which rows belong together but are not model features.

## Training target

Each profile is paired with five illustrative monthly contribution scenarios: ₦5,000, ₦15,000, ₦30,000, ₦60,000 and ₦120,000. These are demo comparison values; Prudential Zenith has not supplied a pricing schedule for this prototype.

For each pair, the label is `1` if the simulated closing balance stays above 25% of the profile's typical essential monthly outflows in all three later months after subtracting the contribution each month. Otherwise, the label is `0`.

The target is observable in the generated future transactions. It is not an observed insurance purchase or a real customer's demonstrated ability to maintain a policy.

## Model and evaluation

The model is a scikit-learn pipeline: median imputation, feature standardization and logistic regression. It trains on 80% of the 1,000 generated training profiles; the other 20% are held out by profile before expanding them into contribution examples. Chinedu and the visible demo accounts are kept out of both sets and are scored only after training.

The app reports ROC-AUC, average precision and Brier score on held-out generated profiles. Those metrics describe performance on this simulation only. The 65% review threshold is a demo routing policy, not a threshold learned from an insurer's outcomes. Linear feature contributions are shown as signals that move the score up or down; they are not causal explanations.

## What the interface shows

1. A batch summary of accounts routed for review and accounts held back.
2. A bank-side queue with the top contribution scenario and model estimate for each demo account.
3. A detail view with 36-month income, outflow and balance history for the selected account.
4. The model estimate for each contribution scenario and its review/hold action.
5. Aggregate counts by contribution scenario for the insurance partner. Customer-level account mapping remains on the bank side in the proposed flow.

The catalogue example is My Smart Save, described by Prudential Zenith Life as a long-term savings plan with life cover. See the [product information page](https://campaign.prudentialzenith.com/dream-big-invest-in-your-tomorrow/). The demo does not claim to reproduce its pricing or underwriting.

## What would be needed for a real pilot

A bank would need to supply approved transaction data, and the insurer would need to confirm the current catalogue and permitted contribution options. The model would then need labels from relevant future cash-flow or offer outcomes, evaluation on later real customer periods, review by bank and insurer teams, and a controlled pilot. The generated demo does not substitute for those steps.
